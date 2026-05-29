import glfw
import json
import os
import shutil
import socket
import time
import threading
from typing import Optional
from collections import defaultdict
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError
from websockets.sync.server import serve, ServerConnection
from websockets.sync.client import connect, ClientConnection
from .types import *
from .widgets import Widget
from abc import ABC, abstractmethod


class Viewer(ABC):
    """
    Base class for viewer. This class setups up the relevant ImGui callbacks.
    The child class must override the 'show_gui' function to build the GUI.
    It can can also override the 'step' function to perform any per frame
    computations required.
    The '(server|client)_(send|recv)' should be used for message passing between
    the server and client for remote viewer support.
    """
    def __init__(self, mode: ViewerMode):
        if not hasattr(self, "window_title"):
            self.window_title = "Viewer"
        self.should_exit = False
        self.num_connections = 0

        # Only used in client mode
        self.websocket = None
        self.remote_recv_timeout = 0.0
        self.remote_max_queue = 1
        self._last_client_packet = None

        # Mapping of widget_id to widget
        self.widget_id_to_widget = {}

        self.mode = mode

        # Import server specific modules
        if self.mode & LOCAL_SERVER:
            self.import_server_modules()

        # Import client specific modules
        self.parent_import_server_modules_called = False
        if self.mode & LOCAL_CLIENT:
            self.import_client_modules()
            assert self.parent_import_server_modules_called, \
                "Call to `super().import_client_modules()` missing."

    def _setup(self):
        """ Go over all of the widgets and initialize them """
        for _, widget in list(vars(self).items()):
            if isinstance(widget, Widget):
                widget.setup()
                self.widget_id_to_widget[widget.widget_id] = widget

    def _destroy(self):
        """ Go over all of the widgets and free any manually allocated objects """
        for _, widget in vars(self).items():
            if isinstance(widget, Widget):
                widget.destroy()

    def _disconnect_client(self, message: str):
        print(message)
        self._last_client_packet = None
        if self.websocket is not None:
            try:
                self.websocket.close()
            except Exception:
                pass
        self.websocket = None

    def _collect_packet(self, widget_method: str, viewer_method: str):
        metadata = {}
        binaries = []

        for _, widget in vars(self).items():
            if not isinstance(widget, Widget):
                continue
            binary, text = getattr(widget, widget_method)()
            if text is not None:
                metadata[widget.widget_id] = text
            if binary is not None:
                binary_view = memoryview(binary)
                binaries.append((widget.widget_id, binary_view))

        binary, text = getattr(self, viewer_method)()
        if text is not None:
            metadata["viewer"] = text
        if binary is not None:
            binaries.append(("viewer", memoryview(binary)))

        payload = {"metadata": metadata}
        if binaries:
            payload["binaries"] = [
                {"widget_id": widget_id, "length": binary_view.nbytes}
                for widget_id, binary_view in binaries
            ]
        return payload, [binary_view for _, binary_view in binaries]

    def _send_packet(self, websocket, payload: dict, binaries, *, dedupe: bool = False):
        if not payload["metadata"] and not payload.get("binaries"):
            return False

        packet = json.dumps(payload, separators=(",", ":"))
        if dedupe and not binaries and packet == self._last_client_packet:
            return False

        websocket.send(packet, text=True)
        if binaries:
            websocket.send(binaries, text=False)

        if dedupe:
            self._last_client_packet = packet if not binaries else None
        return True

    def _recv_packet(self, websocket, *, timeout: Optional[float] = None):
        payload_message = websocket.recv(timeout=timeout)
        if not isinstance(payload_message, str):
            raise ValueError(
                f"Expected websocket control packet as text, received {type(payload_message).__name__}"
            )
        payload = json.loads(payload_message)
        binary_blob = b""
        if payload.get("binaries"):
            # Once we start consuming a packet, finish it fully before polling again,
            # otherwise a timeout between the header and body can desynchronize the stream.
            binary_blob = websocket.recv()
            if isinstance(binary_blob, str):
                raise ValueError("Expected websocket frame payload as binary, received text")
        return payload, binary_blob

    def _apply_packet(self, payload: dict, binary_blob, *, viewer_recv: str, widget_recv: str):
        all_data = defaultdict(dict)

        for widget_id, metadata in payload.get("metadata", {}).items():
            normalized_widget_id = "viewer" if widget_id == "viewer" else int(widget_id)
            all_data[normalized_widget_id]["metadata"] = metadata

        binary_view = memoryview(binary_blob)
        offset = 0
        for binary_desc in payload.get("binaries", []):
            widget_id = binary_desc["widget_id"]
            normalized_widget_id = "viewer" if widget_id == "viewer" else int(widget_id)
            length = int(binary_desc["length"])
            all_data[normalized_widget_id]["binary"] = binary_view[offset:offset + length]
            offset += length

        if offset != len(binary_view):
            raise ValueError("Corrupted websocket payload: binary sizes do not match frame data")

        for widget_id, data in all_data.items():
            if widget_id == "viewer":
                getattr(self, viewer_recv)(data.get("binary", None), data.get("metadata", None))
            else:
                widget = self.widget_id_to_widget[int(widget_id)]
                getattr(widget, widget_recv)(data.get("binary", None), data.get("metadata", None))

    def _drain_server_recv(self, websocket: ServerConnection):
        while True:
            try:
                self._server_recv(websocket, timeout=self.remote_recv_timeout)
            except TimeoutError:
                break

    def _drain_client_recv(self, websocket: ClientConnection):
        while True:
            try:
                self._client_recv(websocket, timeout=self.remote_recv_timeout)
            except TimeoutError:
                break
    
    def _main(self, websocket=None):
        """
        TODO: Update
        Internal method which handles inputs, resize and calls
        backend computation and then creates the UI.
        """
        if self.mode is SERVER:
            self._drain_server_recv(websocket)
            self.step()
            self._server_send(websocket)
            return

        if self.mode is LOCAL:
            self.step()
            self.show_gui()
            return

        if self.mode is CLIENT and self.websocket is not None:
            try:
                self._drain_client_recv(self.websocket)
            except ConnectionClosed:
                self._disconnect_client("INFO: Server disconnected")

        if self.mode & LOCAL_CLIENT:
            self.show_gui()

        if self.mode is CLIENT and self.websocket is not None:
            try:
                self._client_send(self.websocket)
            except ConnectionClosed:
                self._disconnect_client("INFO: Server disconnected")
    
    def _server_loop(self, websocket: ServerConnection):
        """ Internal method which runs the server loop. """
        # We only allow one client to connect at a time (for now).
        if self.num_connections > 0:
            print("INFO: Client already connected. Only one client is allowed.")
            websocket.close()
            return
        self.num_connections += 1

        glfw.make_context_current(self.window)
        self.onconnect(websocket)

        # Main Loop
        try:
            while True:
                self._main(websocket)
        except ConnectionClosedOK:
            print("INFO: Client disconnected.")
            self.num_connections -= 1
        except ConnectionClosedError as e:
            print(f"ERROR: Connection closed with error: {e}")
            self.num_connections -= 1

        glfw.make_context_current(None)

    def _server_send(self, websocket: ServerConnection):
        """
        Internal method which goes over all of the registered widgets to compile 
        and send the server state to the client.
        """
        payload, binaries = self._collect_packet("server_send", "server_send")
        self._send_packet(websocket, payload, binaries)

    def _server_recv(self, websocket: ServerConnection, *, timeout: Optional[float] = None):
        """
        Internal method which receives state from the client and updates all of
        the widgets.
        """
        payload, binary_blob = self._recv_packet(websocket, timeout=timeout)
        self._apply_packet(payload, binary_blob, viewer_recv="server_recv", widget_recv="server_recv")
    
    def _client_loop(self, ip: str, port: int):
        """
        Internal method which runs the client loop. This loop only deals with
        connecting to the server and handling reconnections. The '_main' method
        is run by the 'immapp.run' function.
        """
        while True:
            # Try to connect to the server
            if self.websocket is None:
                try:
                    websocket = connect(
                        f"ws://{ip}:{port}",
                        max_size=None,
                        max_queue=self.remote_max_queue,
                        compression=None,
                        proxy=None,
                        ping_interval=None,
                    )
                    print("INFO: Connected to server.")
                    self._last_client_packet = None
                    self.onconnect(websocket)
                    # Make websocket available after onconnect finishes to avoid
                    # the main thread from using it
                    self.websocket = websocket
                except Exception as e:
                    print(f"INFO: Failed to connect to server with error: {e}."
                        " Retrying in 2 seconds.")
                    self.websocket = None
            time.sleep(2)

    def _client_send(self, websocket: ClientConnection):
        """
        Internal method which goes over all of the registered widgets to compile
        and send the client state to the server.
        """
        payload, binaries = self._collect_packet("client_send", "client_send")
        self._send_packet(websocket, payload, binaries, dedupe=True)

    def _client_recv(self, websocket: ClientConnection, *, timeout: Optional[float] = None):
        """
        Internal method which receives state from the server and updates all of
        the widgets.
        """
        payload, binary_blob = self._recv_packet(websocket, timeout=timeout)
        self._apply_packet(payload, binary_blob, viewer_recv="client_recv", widget_recv="client_recv")

    def run(self, ip: str = "localhost", port: int = 6009):
        self.create_widgets()
        self.running = True

        # Run the client connection in a different thread, the main thread runs the GUI.
        if self.mode is CLIENT:
            connect_thread = threading.Thread(target=self._client_loop, args=(ip, port))
            # Make the thread a daemon so that it exits when the main thread exits.
            connect_thread.daemon = True
            connect_thread.start()
        if self.mode & LOCAL_CLIENT:
            self._runner_params = hello_imgui.RunnerParams()
            self._runner_params.ini_filename = "viewer_layout.ini"
            self._runner_params.ini_filename_use_app_window_title = False
            self._runner_params.app_window_params.window_geometry.window_size_state = hello_imgui.WindowSizeState.maximized
            self._runner_params.app_window_params.window_title = self.window_title
            self._runner_params.imgui_window_params.show_status_bar = True
            self._runner_params.imgui_window_params.show_status_fps = self.mode is not CLIENT
            self._runner_params.imgui_window_params.show_menu_bar = True
            self._runner_params.callbacks.post_init = self._setup
            # self._runner_params.callbacks.before_exit = self._before_exit
            self._runner_params.callbacks.show_gui = self._main
            self._runner_params.callbacks.show_status = self.show_status
            # Disable VSync
            self._runner_params.callbacks.post_init_add_platform_backend_callbacks = lambda: glfw.swap_interval(0)
            self._runner_params.platform_backend_type = hello_imgui.PlatformBackendType.glfw
            self._addon_params = immapp.AddOnsParams(with_implot=True)

            # This is required to make 'want_capture_*' work. The default value is to create a full screen window,
            # but that would mean the 'want_capture_mouse' variable will always be set.
            self._runner_params.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space

            # Seed the ini file from the bundled default layout if no user ini exists yet.
            ini_path = hello_imgui.ini_settings_location(self._runner_params)
            if not os.path.exists(ini_path):
                default_ini = os.path.join(os.path.dirname(__file__), "default_layout.ini")
                if os.path.exists(default_ini):
                    os.makedirs(os.path.dirname(ini_path) or ".", exist_ok=True)
                    shutil.copy2(default_ini, ini_path)

            immapp.run(self._runner_params, self._addon_params)
        if self.mode is SERVER:
            # Initialize OpenGL and setup widgets
            glfw.init()
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            self.window = glfw.create_window(1920, 1080, "", None, None)
            glfw.make_context_current(self.window)
            self._setup()

            # Release window so that the server thread can use it
            glfw.make_context_current(None)

            # Start server
            with serve(
                self._server_loop,
                ip,
                port,
                max_size=None,
                max_queue=self.remote_max_queue,
                compression=None,
                ping_interval=None,
            ) as server:
                display_ip(ip, port)
                server_thread = threading.Thread(target=server.serve_forever)
                server_thread.start()
                while True:
                    try:
                        time.sleep(1)
                    except KeyboardInterrupt:
                        print("INFO: Shutting down server.")
                        server.shutdown()
                        server_thread.join()
                        break
            
            # Reacquire GLFW context and free resources
            glfw.make_context_current(self.window)
            self._destroy()

        self.running = False

    def step(self):
        """ Your application logic goes here. """
        pass

    def create_widgets(self):
        """ Define stateful widgets here. """
    
    def server_send(self) -> tuple[Optional[bytes],Optional[dict]]:
        """ Send global viewer state to the client. """
        return None, None
    
    def server_recv(self, binary: Optional[bytes], text: Optional[dict]):
        """ Receive and process global viewer state from the client. """

    def client_send(self) -> tuple[Optional[bytes],Optional[dict]]:
        """ Send global viewer state to the server. """
        return None, None

    def client_recv(self, binary: Optional[bytes], text: Optional[dict]):
        """ Receive and process global viewer state from the server. """
        pass

    def show_status(self):
        """ Use this function to render status bar at the bottom. """

    def import_server_modules(self):
        """
        Import server specific modules here. We want the viewer to run without 
        any additional dependancies so we only import the server modules when
        on the server. The modules imported here can only be used in `step` and
        `server*` methods. Don't forget to declare the variables as `global` to 
        ensure that they are globally accesible.
        """
    
    def import_client_modules(self):
        """
        Import client specific modules here. We want the viewer to run without 
        needing to install `imgui_bundle` on the server. The modules imported
        here can only be used in `show_gui` and `client*` methods. Don't forget
        to declare the variables as `global` to  ensure that they are globally
        accesible.
        """
        global immapp
        global hello_imgui
        from imgui_bundle import immapp, hello_imgui

        self.parent_import_server_modules_called = True

    def onconnect(self, websocket: 'ClientConnection|ServerConnection'):
        """ Called when a new connection is made. """
        pass

    @abstractmethod
    def show_gui(self) -> bool:
        """ Define the GUI here. """


def display_ip(bind_ip: str, port: int):
    if bind_ip == "localhost":
        connect_ip = "127.0.0.1"
    elif bind_ip not in {"0.0.0.0", "::"}:
        connect_ip = bind_ip
    else:
        connect_ip = "127.0.0.1"

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("10.255.255.255", 1))
                detected_ip = sock.getsockname()[0]
                if detected_ip and not detected_ip.startswith("127."):
                    connect_ip = detected_ip
        except OSError:
            pass

        if connect_ip == "127.0.0.1":
            try:
                for _, _, _, _, sockaddr in socket.getaddrinfo(
                    socket.gethostname(), None, family=socket.AF_INET, type=socket.SOCK_DGRAM
                ):
                    detected_ip = sockaddr[0]
                    if detected_ip and not detected_ip.startswith("127."):
                        connect_ip = detected_ip
                        break
            except OSError:
                pass

    print(f"INFO: Server listening on ws://{bind_ip}:{port}")
    print(f"INFO: Connect with: python view.py --client {connect_ip} --port {port}")

#
# Throwaway transport probe: connects to the streaming server as a bare
# websocket client (NO imgui, NO GUI thread, single thread) and measures pure
# frame inter-arrival cadence. Compare its [rx] output to the in-app [rx]:
#   - headless smooth + in-app bursty  -> client-side GIL/GUI contention
#   - headless ALSO bursty             -> server or network/transport
#
# Usage:
#   python rx_probe.py <ip> <port> [maxqueue]
#   maxqueue: integer, or "none" for unbounded (default 1, matches the app)
#
import sys
import time
import json
import socket
from websockets.sync.client import connect

ip = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 6666
if len(sys.argv) > 3:
    max_queue = None if sys.argv[3].lower() == "none" else int(sys.argv[3])
else:
    max_queue = 1
duration = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0  # 0 = run forever

ws = connect(
    f"ws://{ip}:{port}",
    max_size=None,
    max_queue=max_queue,
    compression=None,
    ping_interval=None,
)
ws.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
print(f"connected to ws://{ip}:{port}  max_queue={max_queue}", flush=True)

start = time.monotonic()
t0 = last = start
n = 0
worst = 0.0
stalls = 0
while True:
    msg = ws.recv()  # text header
    payload = json.loads(msg) if isinstance(msg, str) else {}
    if payload.get("binaries"):
        ws.recv()  # binary blob

    now = time.monotonic()
    if duration and now - start >= duration:
        break
    gap = now - last
    last = now
    n += 1
    worst = max(worst, gap)
    if gap > 0.033:
        stalls += 1
        print(f"    stall @ {now-start:6.2f}s  gap={gap*1e3:5.0f}ms", flush=True)
    if now - t0 >= 1.0:
        print(f"[rx] {n}/s  worst gap {worst*1e3:5.0f}ms  stalls(>33ms): {stalls}", flush=True)
        t0 = now
        n = 0
        worst = 0.0
        stalls = 0

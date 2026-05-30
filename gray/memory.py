import os
import subprocess
from threading import Thread, Event


class GpuMemoryMonitor:
    # * Polls nvidia-smi (on PATH on both Linux and Windows) for this process's peak
    # *** GPU memory. Per-process memory reads as N/A on Windows consumer GPUs (WDDM mode).

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.pid = os.getpid()
        self.peak_mib = 0.0
        self._stop = Event()
        self._thread = Thread(target=self._run, daemon=True)

    def _sample(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return
        for line in out.strip().splitlines():
            pid, _, used = line.partition(",")
            if pid.strip().isdigit() and int(pid) == self.pid and used.strip().isdigit():
                self.peak_mib = max(self.peak_mib, float(used))

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.interval + 10)
        self._sample()
        return self.peak_mib

    def save(self, model_path: str):
        with open(os.path.join(model_path, "peak_memory_use.csv"), "w") as f:
            f.write("peak_memory_gb\n")
            f.write(f"{round(self.peak_mib / 1024, 2)}\n")

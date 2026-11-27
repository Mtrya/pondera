"""Line-based subprocess transport for an external UCI engine."""

import select
import subprocess
import time


class UciError(RuntimeError):
    pass


class UciTimeout(UciError):
    pass


class UciProcess:
    error_type = UciError
    timeout_type = UciTimeout

    def __init__(self, path, timeout):
        self.process = subprocess.Popen(
            [str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
        )
        self.buffer = b""
        self.timeout = timeout
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _send(self, line):
        try:
            self.process.stdin.write((line + "\n").encode())
            self.process.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as error:
            raise self.error_type("engine process is gone") from error

    def _read_line(self, deadline):
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self.timeout_type("engine timed out")
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                continue
            chunk = self.process.stdout.read(65536)
            if not chunk:
                raise self.error_type("engine closed its output")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode(errors="replace").strip()

    def _read_until(self, prefix, deadline):
        seen = []
        while True:
            line = self._read_line(deadline)
            seen.append(line)
            if line.startswith(prefix):
                return line, seen

    def close(self, *, force=False):
        if self.closed:
            return
        if force:
            self.process.kill()
            self.process.wait()
        else:
            try:
                self._send("quit")
            except self.error_type:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.closed = True

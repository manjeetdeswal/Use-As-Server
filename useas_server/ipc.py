"""Hand-off between a second launch and the running app (localhost only).

"Send to phone" from the file manager starts the app again with ``--send file1 file2``;
that copy passes the files to the window that's already running and exits. A plain second
launch just brings the running window to the front instead of starting a second server.
"""

import json
import socket
import threading

IPC_HOST = "127.0.0.1"
IPC_PORT = 47851
_MAGIC = "useas-ipc-1"


def send_to_running(message, timeout=1.5):
    """True if a running Use As Server accepted the message."""
    try:
        with socket.create_connection((IPC_HOST, IPC_PORT), timeout=timeout) as s:
            s.sendall((json.dumps({"magic": _MAGIC, **message}) + "\n").encode("utf-8"))
            s.settimeout(timeout)
            return s.recv(16).startswith(b"ok")
    except OSError:
        return False


class IpcServer(threading.Thread):
    """Listens on 127.0.0.1 only; on_message(dict) runs on this thread."""

    def __init__(self, on_message):
        super().__init__(daemon=True, name="ipc")
        self.on_message = on_message
        self.sock = None
        self.ok = False

    def bind(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.bind((IPC_HOST, IPC_PORT))
            self.sock.listen(4)
            self.ok = True
        except OSError:
            self.sock = None
            self.ok = False
        return self.ok

    def run(self):
        if self.sock is None:
            return
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                try:
                    conn.settimeout(2)
                    data = b""
                    while not data.endswith(b"\n") and len(data) < 1_000_000:
                        part = conn.recv(65536)
                        if not part:
                            break
                        data += part
                    msg = json.loads(data.decode("utf-8"))
                    if msg.get("magic") != _MAGIC:
                        continue
                    self.on_message(msg)
                    conn.sendall(b"ok\n")
                except Exception:
                    pass

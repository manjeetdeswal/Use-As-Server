"""Stream the PC's files to the phone: photos (resized) and video / music (with seeking).

A small HTTP server that only serves files the phone was given a token for (through the
normal, password-protected connection). Tokens are random and expire after a few hours.
Video and audio support HTTP Range requests, so the phone's player can seek.
"""
import io
import mimetypes
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

TOKEN_TTL = 6 * 3600
CHUNK = 256 * 1024
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"}

# Common media types first (the system's list can be incomplete or slow to read).
MEDIA_TYPES = {
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".mkv": "video/x-matroska", ".webm": "video/webm",
    ".mov": "video/quicktime", ".avi": "video/x-msvideo", ".3gp": "video/3gpp", ".ts": "video/mp2t",
    ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".aac": "audio/aac", ".flac": "audio/flac",
    ".wav": "audio/wav", ".ogg": "audio/ogg", ".opus": "audio/opus", ".wma": "audio/x-ms-wma",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif",
    ".webp": "image/webp", ".bmp": "image/bmp", ".heic": "image/heic",
}


def content_type(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in MEDIA_TYPES:
        return MEDIA_TYPES[ext]
    try:
        return mimetypes.guess_type(path)[0] or "application/octet-stream"
    except Exception:
        return "application/octet-stream"


class FileStreamServer:
    def __init__(self, host="0.0.0.0", preferred_port=0, log=None):
        self._tokens = {}               # token -> (path, expires)
        self._by_path = {}              # path -> token (reuse)
        self._lock = threading.Lock()
        self._log = log or (lambda m: None)
        handler = self._make_handler()
        try:
            self._httpd = ThreadingHTTPServer((host, preferred_port), handler)
        except OSError:
            self._httpd = ThreadingHTTPServer((host, 0), handler)      # that port is taken
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever, daemon=True, name="file-stream").start()

    def stop(self):
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except Exception:
            pass

    def register(self, path):
        """A token for one file (the same one again while it's still valid)."""
        now = time.time()
        with self._lock:
            for t, (_p, exp) in list(self._tokens.items()):
                if exp < now:
                    self._tokens.pop(t, None)
                    self._by_path.pop(_p, None)
            tok = self._by_path.get(path)
            if tok and tok in self._tokens:
                self._tokens[tok] = (path, now + TOKEN_TTL)
                return tok
            tok = secrets.token_urlsafe(18)
            self._tokens[tok] = (path, now + TOKEN_TTL)
            self._by_path[path] = tok
            return tok

    def _lookup(self, token):
        with self._lock:
            entry = self._tokens.get(token)
        if not entry or entry[1] < time.time():
            return None
        return entry[0]

    # ------------------------------------------------------------------------------------
    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):      # quiet
                pass

            def do_HEAD(self):
                self._handle(head=True)

            def do_GET(self):
                self._handle(head=False)

            def _fail(self, code, text=""):
                body = text.encode()
                self.send_response(code)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body and self.command != "HEAD":
                    self.wfile.write(body)

            def _handle(self, head):
                url = urlparse(self.path)
                parts = [p for p in url.path.split("/") if p]
                if len(parts) < 2 or parts[0] not in ("f", "img"):
                    return self._fail(404)
                path = server._lookup(parts[1])
                if not path or not os.path.isfile(path):
                    return self._fail(404, "Not found (link expired?)")
                try:
                    if parts[0] == "img":
                        q = parse_qs(url.query)
                        try:
                            max_side = max(256, min(int(q.get("max", ["2048"])[0]), 4096))
                        except ValueError:
                            max_side = 2048
                        data = _resized_jpeg(path, max_side)
                        if data is not None:
                            self.send_response(200)
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(data)))
                            self.send_header("Cache-Control", "private, max-age=3600")
                            self.end_headers()
                            if not head:
                                self.wfile.write(data)
                            return
                        # Pillow can't read it (HEIC, RAW…): send the original, the phone may manage.
                    self._send_file(path, head)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass                          # the player closed the connection (seek / stop)
                except Exception as e:
                    try:
                        self._fail(500, str(e)[:200])
                    except Exception:
                        pass

            def _send_file(self, path, head):
                size = os.path.getsize(path)
                ctype = content_type(path)
                start, end = 0, size - 1
                rng = self.headers.get("Range", "")
                partial = False
                if rng.startswith("bytes="):
                    spec = rng[6:].split(",")[0].strip()
                    a, _, b = spec.partition("-")
                    try:
                        if a == "":                       # last N bytes
                            n = int(b)
                            start, end = max(0, size - n), size - 1
                        else:
                            start = int(a)
                            end = int(b) if b else size - 1
                        end = min(end, size - 1)
                        if start > end or start >= size:
                            self.send_response(416)
                            self.send_header("Content-Range", f"bytes */{size}")
                            self.send_header("Content-Length", "0")
                            self.end_headers()
                            return
                        partial = True
                    except ValueError:
                        start, end = 0, size - 1
                length = end - start + 1
                self.send_response(206 if partial else 200)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                if partial:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if head:
                    return
                with open(path, "rb") as f:
                    f.seek(start)
                    left = length
                    while left > 0:
                        chunk = f.read(min(CHUNK, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)

        return Handler


_img_cache = {}
_img_lock = threading.Lock()


def _resized_jpeg(path, max_side):
    """The photo at most max_side pixels on its longer side, as JPEG (cached for a few)."""
    if os.path.splitext(path)[1].lower() not in IMAGE_EXTS:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, int(st.st_mtime), st.st_size, max_side)
    with _img_lock:
        if key in _img_cache:
            return _img_cache[key]
    try:
        from PIL import Image, ImageOps
        img = Image.open(path)
        img.draft("RGB", (max_side, max_side))
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        if img.mode not in ("RGB", "L"):
            bg = Image.new("RGB", img.size, (0, 0, 0))
            rgba = img.convert("RGBA")
            bg.paste(rgba, mask=rgba.getchannel("A"))
            img = bg
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=88)
        data = buf.getvalue()
    except Exception:
        return None
    with _img_lock:
        if len(_img_cache) > 24:
            _img_cache.clear()
        _img_cache[key] = data
    return data

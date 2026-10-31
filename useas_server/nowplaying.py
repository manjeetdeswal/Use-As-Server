"""What the PC is playing (title, artist, cover art) and controlling it from the phone.

* Windows: the system media controls (the same ones the volume flyout shows), so it works
  with Spotify, browsers (YouTube, Netflix…), the Media Player app, VLC 4, etc.
  Needs ``pip install winrt-Windows.Media.Control winrt-Windows.Storage.Streams
  winrt-Windows.Foundation`` (or the older all-in-one ``winsdk``).
* Linux: MPRIS through ``playerctl`` (sudo apt install playerctl).
* macOS: Music and Spotify through AppleScript.

Without a backend the phone still gets working play/pause/next/previous buttons: they fall
back to the media keys.
"""

import asyncio
import io
import queue
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlparse

POLL_SECONDS = 1.0
_UNSET = object()      # "nothing sent yet": the next poll sends everything
ART_SIZE = 320


def _shrink_art(data):
    """Any image bytes -> small JPEG (or None)."""
    if not data:
        return None
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        img.thumbnail((ART_SIZE, ART_SIZE))
        out = io.BytesIO()
        img.save(out, "JPEG", quality=82)
        return out.getvalue()
    except Exception:
        return None


def _pretty_app(app_id):
    """'Spotify.exe' -> 'Spotify', 'Microsoft.ZuneMusic_8wekyb3d8bbwe!Microsoft.ZuneMusic' -> 'Media Player'."""
    if not app_id:
        return ""
    known = {"chrome": "Chrome", "msedge": "Edge", "firefox": "Firefox", "spotify": "Spotify",
             "vlc": "VLC", "brave": "Brave", "opera": "Opera", "zunemusic": "Media Player",
             "zunevideo": "Movies & TV", "applemusic": "Apple Music", "music": "Music"}
    s = app_id.split("!")[-1].split("\\")[-1]
    s = s[:-4] if s.lower().endswith(".exe") else s
    s = s.split(".")[-1] if "." in s else s
    return known.get(s.lower(), s[:1].upper() + s[1:])


# ------------------------------------------------------------------------------------------
# Backends: each returns (info dict or None, art bytes or None) and runs commands.
# ------------------------------------------------------------------------------------------

class _WindowsBackend:
    """Global System Media Transport Controls via pywinrt (winrt-*) or winsdk."""

    def __init__(self):
        try:
            from winrt.windows.media.control import \
                GlobalSystemMediaTransportControlsSessionManager as Manager
            from winrt.windows.storage.streams import Buffer, DataReader, InputStreamOptions
        except ImportError:
            from winsdk.windows.media.control import \
                GlobalSystemMediaTransportControlsSessionManager as Manager
            from winsdk.windows.storage.streams import Buffer, DataReader, InputStreamOptions
        self.Manager, self.Buffer, self.DataReader, self.Options = Manager, Buffer, DataReader, InputStreamOptions
        self.manager = None
        self._art_key = None
        self._art = None

    async def _session(self):
        if self.manager is None:
            self.manager = await self.Manager.request_async()
        return self.manager.get_current_session()

    async def _read_thumbnail(self, ref):
        stream = await ref.open_read_async()
        size = int(stream.size)
        if size <= 0 or size > 8 * 1024 * 1024:
            return None
        buf = self.Buffer(size)
        result = await stream.read_async(buf, size, self.Options.READ_AHEAD)
        result = result if result is not None else buf
        try:
            data = bytes(result)                   # pywinrt 2: buffer protocol
            if data:
                return data[:int(result.length)] if hasattr(result, "length") else data
        except Exception:
            pass
        reader = self.DataReader.from_buffer(result)   # winsdk
        data = bytearray(int(result.length))
        reader.read_bytes(data)
        return bytes(data)

    async def get(self):
        session = await self._session()
        if session is None:
            return None, None
        props = await session.try_get_media_properties_async()
        playback = session.get_playback_info()
        timeline = session.get_timeline_properties()
        playing = int(getattr(playback.playback_status, "value", playback.playback_status)) == 4

        def seconds(td):
            try:
                return max(0.0, td.total_seconds())
            except Exception:
                try:
                    return max(0.0, int(td.duration) / 10_000_000)
                except Exception:
                    return -1.0

        duration = seconds(timeline.end_time) - max(0.0, seconds(timeline.start_time))
        position = seconds(timeline.position)
        try:   # the timeline is a snapshot; extrapolate while playing
            updated = timeline.last_updated_time
            if playing and updated is not None and position >= 0:
                from datetime import datetime, timezone
                age = (datetime.now(timezone.utc) - updated).total_seconds()
                if 0 < age < 3600:
                    position += age
        except Exception:
            pass
        try:
            controls = playback.controls
            can_seek = bool(controls.is_playback_position_enabled)
        except Exception:
            can_seek = False

        info = {
            "title": props.title or "",
            "artist": props.artist or props.album_artist or "",
            "album": props.album_title or "",
            "app": _pretty_app(session.source_app_user_model_id),
            "playing": playing,
            "position": round(position, 1) if duration > 0 else -1,
            "duration": round(duration, 1) if duration > 0 else -1,
            "can_seek": can_seek and duration > 0,
        }
        if not info["title"]:
            return None, None

        art = None
        key = (info["title"], info["artist"], info["album"])
        if key != self._art_key:
            self._art_key = key
            self._art = None
            if props.thumbnail is not None:
                try:
                    self._art = _shrink_art(await self._read_thumbnail(props.thumbnail))
                except Exception:
                    self._art = None
            art = self._art
        return info, art

    async def command(self, action, value=None):
        session = await self._session()
        if session is None:
            return False
        if action == "play_pause":
            return bool(await session.try_toggle_play_pause_async())
        if action == "next":
            return bool(await session.try_skip_next_async())
        if action == "previous":
            return bool(await session.try_skip_previous_async())
        if action == "seek" and value is not None:
            return bool(await session.try_change_playback_position_async(int(float(value) * 10_000_000)))
        return False


class _PlayerctlBackend:
    FMT = "{{status}}\t{{title}}\t{{artist}}\t{{album}}\t{{playerName}}\t{{position}}\t{{mpris:length}}\t{{mpris:artUrl}}"

    def __init__(self):
        if shutil.which("playerctl") is None:
            raise ImportError("playerctl not installed")
        self._art_url = None

    def _run(self, *args):
        try:
            r = subprocess.run(["playerctl", *args], capture_output=True, text=True, timeout=2)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    async def get(self):
        out = self._run("metadata", "--format", self.FMT)
        if not out:
            return None, None
        parts = (out.split("\t") + [""] * 8)[:8]
        status, title, artist, album, player, pos, length, art_url = parts
        if not title:
            return None, None

        def us(v):
            try:
                return int(float(v)) / 1_000_000
            except ValueError:
                return -1
        duration = us(length)
        info = {"title": title, "artist": artist, "album": album, "app": player.capitalize(),
                "playing": status == "Playing", "position": round(us(pos), 1) if duration > 0 else -1,
                "duration": round(duration, 1) if duration > 0 else -1, "can_seek": duration > 0}
        art = None
        if art_url != self._art_url:
            self._art_url = art_url
            data = None
            try:
                if art_url.startswith("file://"):
                    with open(unquote(urlparse(art_url).path), "rb") as f:
                        data = f.read()
                elif art_url.startswith("http"):
                    import urllib.request
                    with urllib.request.urlopen(art_url, timeout=4) as r:
                        data = r.read(4 * 1024 * 1024)
            except Exception:
                data = None
            art = _shrink_art(data)
        return info, art

    async def command(self, action, value=None):
        cmd = {"play_pause": ["play-pause"], "next": ["next"], "previous": ["previous"]}.get(action)
        if action == "seek" and value is not None:
            cmd = ["position", str(float(value))]
        return cmd is not None and self._run(*cmd) is not None


class _MacBackend:
    """Music and Spotify via AppleScript (other apps: media keys only)."""
    APPS = ("Spotify", "Music")

    def _osa(self, script):
        try:
            r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=3)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    def _running(self, app):
        return self._osa(f'application "{app}" is running') == "true"

    async def get(self):
        for app in self.APPS:
            if not self._running(app):
                continue
            out = self._osa(
                f'tell application "{app}" to return (player state as text) & "\t" & name of current track '
                f'& "\t" & artist of current track & "\t" & album of current track & "\t" & '
                f'(player position as text) & "\t" & (duration of current track as text)')
            if not out:
                continue
            state, title, artist, album, pos, dur = (out.split("\t") + [""] * 6)[:6]
            try:
                duration = float(dur.replace(",", "."))
                if app == "Spotify":
                    duration /= 1000.0           # Spotify reports milliseconds
            except ValueError:
                duration = -1
            try:
                position = float(pos.replace(",", "."))
            except ValueError:
                position = -1
            return ({"title": title, "artist": artist, "album": album, "app": app,
                     "playing": state == "playing", "position": round(position, 1),
                     "duration": round(duration, 1), "can_seek": duration > 0}, None)
        return None, None

    async def command(self, action, value=None):
        for app in self.APPS:
            if self._running(app):
                verb = {"play_pause": "playpause", "next": "next track",
                        "previous": "previous track"}.get(action)
                if action == "seek" and value is not None:
                    verb = f"set player position to {float(value)}"
                return verb is not None and self._osa(f'tell application "{app}" to {verb}') is not None
        return False


def make_backend():
    """The backend for this OS, or None (with a reason) if it isn't installed."""
    try:
        if sys.platform == "win32":
            return _WindowsBackend(), ""
        if sys.platform == "darwin":
            return _MacBackend(), ""
        return _PlayerctlBackend(), ""
    except Exception as e:
        if sys.platform == "win32":
            return None, ("pip install winrt-Windows.Media.Control winrt-Windows.Storage.Streams "
                          "winrt-Windows.Foundation")
        return None, f"sudo apt install playerctl ({e})"


class NowPlayingWatcher:
    """Polls the backend on its own thread (with its own asyncio loop) while phones want it,
    and runs the phone's play/pause/next/previous/seek commands on that same thread.

    on_info(info or None) is called when what's playing changes (and every 10 s while playing,
    so the phone's progress bar stays in sync); on_art(jpeg) when the cover changes.
    """

    def __init__(self, backend, on_info, on_art, log=print):
        self.backend = backend
        self.on_info, self.on_art, self._log = on_info, on_art, log
        self._wanted = threading.Event()
        self._cmds = queue.Queue()
        self._stop = False
        self._last_sig = _UNSET
        self._last_sent = 0.0
        self.current = None
        self.current_art = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="now-playing")
        self._thread.start()

    def set_active(self, active):
        if active:
            self._wanted.set()
        else:
            self._wanted.clear()

    def resend(self):
        """Next poll sends the full state again (a phone just subscribed)."""
        self._last_sig = _UNSET

    def stop(self):
        self._stop = True
        self._cmds.put(None)

    def command(self, action, value=None, timeout=4.0):
        """Run a control on the watcher thread. True if the player accepted it."""
        reply = queue.Queue()
        self._cmds.put((action, value, reply))
        try:
            return bool(reply.get(timeout=timeout))
        except queue.Empty:
            return False

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logged = False
        next_poll = 0.0
        while not self._stop:
            wait = max(0.0, next_poll - time.time()) if self._wanted.is_set() else 0.5
            try:
                item = self._cmds.get(timeout=wait)
            except queue.Empty:
                item = None
            if self._stop:
                break
            if item is not None:
                action, value, reply = item
                try:
                    ok = loop.run_until_complete(self.backend.command(action, value))
                except Exception as e:
                    self._log(f"⚠️ Media control: {e}")
                    ok = False
                reply.put(ok)
                self._last_sig = _UNSET          # report the new state right away
                next_poll = time.time() + 0.35
                continue
            if not self._wanted.is_set():
                continue
            next_poll = time.time() + POLL_SECONDS
            try:
                info, art = loop.run_until_complete(self.backend.get())
                logged = False
            except Exception as e:
                info, art = None, None
                if not logged:
                    logged = True
                    self._log(f"⚠️ Now playing: {e}")
            fresh = self._last_sig is _UNSET
            if art is not None:
                self.current_art = art
                self.on_art(art)
            elif fresh and info and self.current_art:
                self.on_art(self.current_art)         # a new subscriber needs the cover too
            sig = None if info is None else (info["title"], info["artist"], info["playing"], info["app"])
            now = time.time()
            if fresh or sig != self._last_sig or (info and info["playing"] and now - self._last_sent > 10):
                self._last_sig = sig
                self._last_sent = now
                self.current = info
                self.on_info(info)

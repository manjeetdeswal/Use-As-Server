"""Use As server core: networking, pairing, protocol, files, clipboard, streams.

Everything OS-specific (mouse, keyboard, gamepad, audio devices, virtual camera/display,
auto-start) goes through ``self.platform`` (see platforms/), so Windows, Linux and macOS
share this code and get every fix at once.
"""

import asyncio
import base64
import json
import os
import queue
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import psutil
import pyperclip
import websockets

from . import media, nowplaying, protocol
from .common import APP_VERSION, SAVE_DIR, resource_path, safe_filename, unique_path
from .platforms import get_platform
from .platforms.base import normalize_gamepad_state

try:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"❌ Error creating folder: {e}")

class DiscoveryServer(threading.Thread):
    def __init__(self, port=8080, update_queue=None):
        super().__init__()
        self.port = port
        self.running = True
        self.update_queue = update_queue
        self.current_ips = set()
        self.sock = None

    def get_active_ips(self):
        """Return every active, non-loopback IPv4 interface with its computed broadcast address."""
        results = []
        try:
            stats = psutil.net_if_stats()
            for iface, addrs in psutil.net_if_addrs().items():
                if iface in stats and not stats[iface].isup:
                    continue
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        ip = addr.address
                        if ip.startswith("127.") or ip.startswith("169.254."):
                            continue
                        if self._is_virtual_adapter(iface, ip):   # 👈 CHANGED: now passes ip too
                            continue
                        netmask = addr.netmask or "255.255.255.0"
                        results.append({
                            "ip": ip,
                            "broadcast": self._calc_broadcast(ip, netmask),
                            "iface": iface,
                            "label": self._label_interface(iface, ip),
                        })
        except Exception as e:
            if self.update_queue:
                self.update_queue.put(("log", f"⚠️ IP Scan Warning: {e}"))
            # Fallback to old method so it never totally breaks
            try:
                hostname = socket.gethostname()
                for ip in socket.gethostbyname_ex(hostname)[2]:
                    if not ip.startswith("127.") and not ip.startswith("169.254."):
                        results.append({
                            "ip": ip,
                            "broadcast": ".".join(ip.split(".")[:3]) + ".255",
                            "iface": "unknown",
                            "label": self._label_interface("unknown", ip),
                        })
            except Exception:
                pass
        return results
        
        
    def _create_socket(self):
        """Creates a fresh UDP broadcast socket with crash recovery."""
        for attempt in range(3):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                except AttributeError:
                    pass
                # Set a tight timeout so unreachable interfaces don't hang the loop
                sock.settimeout(0.1)
                return sock
            except OSError:
                time.sleep(0.2)  # Give Windows time to release the zombie port
        raise Exception("Failed to bind UDP discovery socket after 3 attempts.")
    
    
    def _get_primary_ip(self):
        """Ask the OS routing table which interface it would actually use — most reliable signal."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return None

    @staticmethod
    def _calc_broadcast(ip, netmask):
        try:
            ip_p = [int(x) for x in ip.split(".")]
            m_p = [int(x) for x in netmask.split(".")]
            return ".".join(str(ip_p[i] | (255 - m_p[i])) for i in range(4))
        except Exception:
            return ".".join(ip.split(".")[:3]) + ".255"

    @staticmethod
    def _is_virtual_adapter(iface_name, ip=None):
        """Detect adapters that aren't real physical network paths."""
        name = iface_name.lower()
        virtual_markers = [
            "virtualbox", "vmware", "hyper-v", "vethernet",
            "loopback", "docker", "wsl", "tap-", "tap0",
            "npcap", "bluetooth", "vpn", "tailscale", "zerotier",
            "virtual"
        ]
        if any(marker in name for marker in virtual_markers):
            return True

        # Fallback: block by known virtual-adapter IP ranges,
        # since Windows often renames these to generic "Ethernet N"
        if ip:
            virtual_ip_prefixes = [
                "192.168.56.",   # VirtualBox Host-Only default
                "192.168.99.",   # Docker Toolbox / legacy VirtualBox
                "172.17.",       # Docker default bridge
                "172.18.", "172.19.", "172.20.",  # Docker/other bridges
                "10.211.55.",    # Parallels
                "169.254.",      # already excluded elsewhere, but just in case
            ]
            if any(ip.startswith(p) for p in virtual_ip_prefixes):
                return True

        return False

    @staticmethod
    def _label_interface(iface, ip):
        name = iface.lower()
        if ip.startswith("192.168.42."): return "USB Tethering"
        if ip.startswith("192.168.43."): return "Mobile Hotspot"
        # Check wifi FIRST and broaden keywords, since "ethernet" can
        # appear inside some wifi adapter friendly names too
        if any(k in name for k in ["wi-fi", "wifi", "wlan", "wireless", "802.11"]):
            return "Wi-Fi"
        if any(k in name for k in ["usb", "rndis"]):
            return "USB Tethering"
        if any(k in name for k in ["ethernet", "eth", "local area connection"]):
            return "Ethernet / LAN"
        return "Network"

    def run(self):
        try:
            self.sock = self._create_socket()
        except Exception as e:
            if self.update_queue:
                self.update_queue.put(("log", f"❌ Discovery Fatal: {e}"))
            return

        message = f"UNIFIED_REMOTE_SERVER:{self.port}".encode('utf-8')

        primary_ip = self._get_primary_ip()
        interfaces = self.get_active_ips()
        for i in interfaces:
            if i["ip"] == primary_ip:
                i["label"] = f"{i['label']} ★"
        self.current_ips = {i["ip"] for i in interfaces}

        if self.update_queue:
            self.update_queue.put(("log", f"📡 Broadcasting discovery on port {self.port}..."))
            self.update_queue.put(("update_endpoints",
                [{"ip": i["ip"], "port": self.port, "label": i["label"]} for i in interfaces]))

        scan_counter = 0

        while self.running:
            scan_counter += 1

            # Only re-scan interfaces every 3 ticks (~3s) — psutil calls are
            # relatively expensive and networks rarely change every second.
            # We still broadcast every second below using the last known list.
            if scan_counter % 3 == 0:
                primary_ip = self._get_primary_ip()
                fresh_interfaces = self.get_active_ips()
                for i in fresh_interfaces:
                    if i["ip"] == primary_ip:
                        i["label"] = f"{i['label']} ★"
                new_ips = {i["ip"] for i in fresh_interfaces}

                if new_ips != self.current_ips:
                    if self.update_queue:
                        self.update_queue.put(("log", "🔄 Network change detected! Rerouting broadcast..."))
                        self.update_queue.put(("update_endpoints",
                            [{"ip": i["ip"], "port": self.port, "label": i["label"]} for i in fresh_interfaces]))
                    self.current_ips = new_ips
                    try:
                        if self.sock: self.sock.close()
                    except: pass
                    try:
                        self.sock = self._create_socket()
                    except Exception as e:
                        if self.update_queue:
                            self.update_queue.put(("log", f"❌ Socket Rebind Failed: {e}"))
                        time.sleep(1)
                        continue

                interfaces = fresh_interfaces  # keep for broadcast targets below

            # Broadcast to EACH interface's real broadcast address, not a guessed /24
            targets = {i["broadcast"] for i in interfaces}
            targets.add('255.255.255.255')

            dead = False
            for target in targets:
                try:
                    if self.sock:
                        self.sock.sendto(message, (target, 8888))
                except OSError as e:
                    if getattr(e, 'winerror', 0) in (10051, 10065) or e.errno in (101, 113):
                        dead = True
                except Exception:
                    pass

            if dead:
                self.current_ips.clear()
                scan_counter = 0  # force a re-scan on the very next tick

            time.sleep(1)

        if self.sock:
            try: self.sock.close()
            except: pass

    def stop(self):
        self.running = False



class UDPMouseServer(threading.Thread):
    """Fast relative mouse movement over UDP ("dx,dy"). Only accepts packets from IPs that
    have an authenticated WebSocket (UDP has no session of its own)."""

    def __init__(self, port=8081, is_allowed=None, move=None):
        super().__init__(daemon=True, name="udp-mouse")
        self.port = port
        self.running = True
        self.sock = None
        self.is_allowed = is_allowed or (lambda ip: True)
        self.move = move

    def run(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.settimeout(0.5)   # blocking with timeout, not a busy-poll
            self.sock.bind(("0.0.0.0", self.port))
        except Exception as e:
            print(f"❌ UDP Bind Failed on {self.port}: {e}")
            self.running = False
            return

        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                if not self.is_allowed(addr[0]):
                    continue
                parts = data.decode("utf-8").split(",")
                if len(parts) == 2 and self.move is not None:
                    dx, dy = int(parts[0]), int(parts[1])
                    if dx or dy:
                        self.move(dx, dy)
            except (socket.timeout, BlockingIOError):
                continue
            except Exception:
                pass
        try:
            self.sock.close()
        except Exception:
            pass

    def stop(self):
        self.running = False


class UnifiedRemoteServer:
    def __init__(self, host="0.0.0.0", port=8080, gaming_mode=True, password="", require_password=False,
                 update_queue: queue.Queue = None, pairing_token=""):
        self.sending_lock = threading.Lock()


        self.gaming_mode = gaming_mode

        self.host = host
        self.port = port  # Main TCP Port

        self.ack_event = threading.Event()
        # --- AUTHENTICATION ---
        self.pairing_password = password or ""
        self.require_password = bool(require_password) and bool(self.pairing_password)
        # Secret inside the QR code. Accepted like the password, so scanning the QR pairs a
        # phone even when a password is required; "New code" in the QR window replaces it.
        self.pairing_token = pairing_token or ""
        self._authenticated_clients = set()
        self._auth_attempts = {}
        # IPs with an authenticated WebSocket. The UDP mouse channel only accepts these.
        self._allowed_udp_ips = frozenset()

        self.clients = {}
        self._loop = None
        self.loop = None
        self._ws_server = None
        self._thread = None
        self.update_queue = update_queue or queue.Queue()
        self._stop_event = threading.Event()

        # OS backend (mouse, keyboard, gamepad, audio devices, virtual camera/display).
        self.platform = get_platform()
        self.platform.set_logger(lambda msg: self._put("log", msg))
        try:
            self.platform.setup()
        except Exception as e:
            self._put("log", f"⚠️ Platform setup: {e}")

        # Work that must not run on the asyncio thread (it would stall every client).
        # One worker each, so events keep their order (key down before key up, etc.).
        self._input_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="input")
        self._control_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="control")

        # Per-feature subscribers, so streams only go to the phones that asked for them
        # and stop when those phones disconnect.
        self._display_clients = set()
        self._audio_clients = set()
        self._mic_client = None
        self._video_busy = set()          # sockets with a screen frame still being sent
        self._rumble_levels = {}          # websocket -> current rumble intensity
        self._incoming_files = {}         # (websocket id, filename) -> {"part": Path, "fh": file}
        self._zoom_accum = 0.0
        self._audio_target_rate = 48000
        self._audio_restart = threading.Event()
        self._audio_started_by_phone = False
        self._audio_gen = 0
        self._display_gen = 0
        self._display_thread = None
        self._display_monitor_indices = [0]
        self._display_quality = 35

        # Phase 3 codecs. Each display phone gets the codec it asked for ("h264" or "jpeg");
        # each speaker phone gets "opus", "pcm" (binary) or "json" (old apps).
        self._display_codecs = {}         # websocket -> "h264" | "jpeg"
        self._h264_encoder = None         # created by the capture thread
        self._h264_broken = False         # no working encoder: everyone gets JPEG
        self._h264_need_key = set()       # sockets that must wait for the next keyframe
        self._h264_inflight = {}          # websocket -> H.264 packets still being written
        self._h264_key_wanted = False
        self._h264_last_forced = 0.0
        self._h264_lock = threading.Lock()
        self._h264_congested_at = 0.0
        self._h264_scale = 1.0
        self._h264_scale_changed = 0.0
        self._audio_codecs = {}           # websocket -> "opus" | "pcm" | "json"
        self._opus_encoder = None
        self._opus_broken = False
        self._mic_codec = "pcm"
        self._mic_opus = None
        self._feature_cache = None

        # Phase 4: virtual monitor for the phone (Windows: usbmmidd driver; Linux: xrandr).
        # The GUI sets this from Settings › "Virtual monitor for the phone".
        self.virtual_display_enabled = True
        self._virtual_cleanup_timer = None
        self._virtual_lock = threading.Lock()
        if self.platform.virtual_displays_added:
            # Left over from a crash: remove it unless a phone starts viewing soon.
            self._schedule_virtual_cleanup()

        # Now playing (title / artist / cover of what the PC plays) for subscribed phones.
        self._media_clients = set()
        self._np_watcher = None
        self._np_backend_checked = False
        self._np_backend = None

        # Phase 4: images on the clipboard (copied files are sent as files).
        self._last_clip_image = None      # hash of the last image sent or received
        self._last_clip_files = None

        # Camera frames from the phone are processed on their own thread; only the newest is kept.
        self._video_lock = threading.Lock()
        self._pending_video = None
        self._video_event = threading.Event()
        self._video_worker_running = True
        threading.Thread(target=self._video_worker, daemon=True, name="video").start()

        # Phone-mic playback runs on its own thread; frames are queued, oldest dropped.
        self._mic_queue = queue.Queue(maxsize=8)
        self._mic_rate = 16000
        self._mic_stream = None
        self._mic_player = None
        self._mic_reopen = False
        self._logged_mic_err = False
        threading.Thread(target=self._mic_worker, daemon=True, name="mic").start()

        # Clipboard auto-sync (PC -> phone). Off until the GUI switch turns it on.
        self.clipboard_autosync = False
        self._last_clip = None          # last text sent OR received, so a sync never echoes back
        threading.Thread(target=self._clipboard_watcher, daemon=True, name="clipboard").start()

        # 1. START DISCOVERY (Broadcasts the TCP port)
        self.discovery = DiscoveryServer(port=self.port, update_queue=self.update_queue)
        self.discovery.start()

        # 2. START UDP MOUSE (Listens on TCP Port + 1) — authenticated phones only
        self.udp_port = self.port + 1
        self.udp_mouse = UDPMouseServer(port=self.udp_port,
                                        is_allowed=lambda ip: ip in self._allowed_udp_ips,
                                        move=self.platform.mouse_move_rel)
        self.udp_mouse.start()

        self._broadcast_queue = None

        # ... (Rest of variables) ...
        self._vcam = None
        self._vcam_running = False
        self._last_frame = None
        self._frame_count = 0
        self._audio_stream = None
        self._audio_thread = None
        self._streaming_audio = False
        self._display_active = False
        self._gamepad_active = False
        self.client_gamepads = {}

        self._display_width = 1280
        self._display_height = 720
        self._display_fps = 30

        # Audio mic flag
        self._mic_active = False

        self.bg_mode = "none"
        self.bg_image_path = None
        self.bg_image_cache = None
        self.preview_active = True
        self.latest_preview_frame = None
        self.is_mirrored = False
        self.is_flipped = False
        self.brightness_boost = 0
        self.target_w = 1280
        self.target_h = 720
        self.segmentor = None


    def send_to_android(self, msg_type, payload):
        """Helper to send JSON messages to Android (Thread-Safe)."""
        # Ensure the loop exists and is running
        if self._loop and self._loop.is_running() and self._broadcast_queue:
            message = json.dumps({"type": msg_type, "payload": payload})
            # This is the thread-safe way to talk to asyncio from a normal thread
            asyncio.run_coroutine_threadsafe(self._broadcast_queue.put(message), self._loop)
        else:
            print("❌ Error: Event loop is not running, cannot send message.")

    async def _broadcast_worker(self):
        """Async worker that sends queued messages to all clients."""
        while True:
            try:
                # Safe guard if queue isn't ready
                if not self._broadcast_queue:
                    await asyncio.sleep(0.1)
                    continue

                message = await self._broadcast_queue.get()
                if message is None: break

                # Authenticated phones only. Dead sockets are cleaned up by their own
                # handler's finally block (which also unmounts the gamepad etc.).
                for client in self._auth_targets():
                    try:
                        await client.send(message)
                    except Exception:
                        pass
            except Exception as e:
                pass

    # ---------------------------------------------------------------
    # CLIPBOARD AUTO-SYNC (PC -> phone)
    # ---------------------------------------------------------------
    MAX_CLIPBOARD_CHARS = 200_000

    def set_clipboard_autosync(self, enabled):
        enabled = bool(enabled)
        if enabled and not self.clipboard_autosync:
            # Only sync what's copied from now on, not whatever was on the clipboard before.
            try:
                self._last_clip = pyperclip.paste()
            except Exception:
                pass
            self._clip_seq = self._clipboard_sequence()
        self.clipboard_autosync = enabled

    def _clipboard_sequence(self):
        return self.platform.clipboard_sequence()

    def _clipboard_is_private(self):
        return self.platform.clipboard_is_private()

    def _clipboard_watcher(self):
        self._clip_seq = self._clipboard_sequence()
        while self._video_worker_running:
            time.sleep(0.5)
            if not self.clipboard_autosync:
                continue
            seq = self._clipboard_sequence()
            if seq is not None and seq == self._clip_seq:
                continue                      # nothing copied since last check
            if not self._auth_targets():
                continue                      # no phone yet: send the latest copy once one connects
            self._clip_seq = seq
            if self._clipboard_is_private():
                continue
            try:
                text = pyperclip.paste()
            except Exception:
                continue                      # clipboard busy (another app holds it) — retry next tick
            if not text or not isinstance(text, str):
                # No text: maybe an image (screenshot) or files copied in the file manager.
                # Without a change counter (Linux) this is polled, so not every tick.
                if seq is None:
                    if time.time() - getattr(self, "_last_rich_check", 0.0) < 2.0:
                        continue
                    self._last_rich_check = time.time()
                try:
                    self.send_rich_clipboard(manual=False)
                except Exception as e:
                    self._put("log", f"⚠️ Clipboard image/files: {e}")
                continue
            if text == self._last_clip:
                continue
            self._last_clip = text
            if len(text) > self.MAX_CLIPBOARD_CHARS:
                self._put("log", "📋 Copied text too large to auto-sync")
                continue
            self.send_to_android("clipboard_text", text)
            self._last_clip_image = self._last_clip_files = None   # copying them again re-sends
            self._put("log", f"📋 PC clipboard sent to phone ({len(text)} chars)")

    MAX_CLIPBOARD_IMAGE_BYTES = 12 * 1024 * 1024
    MAX_AUTO_FILES = 10
    MAX_AUTO_FILES_BYTES = 50 * 1024 * 1024

    @staticmethod
    def _image_key(image):
        import hashlib
        rgb = image.convert("RGB")
        return hashlib.sha1(rgb.tobytes() + repr(rgb.size).encode()).hexdigest()

    def _image_capable_phones(self):
        return [ws for ws in self._auth_targets()
                if "clipboard_image" in (self.clients.get(ws) or {}).get("features", [])]

    def send_rich_clipboard(self, manual=True):
        """Send a copied image (as a clipboard image) or copied files (as files) to the phone.
        manual=True: the GUI button — always sends. manual=False: auto-sync — skips repeats and
        big file sets. Returns True if something was sent."""
        rich = self.platform.clipboard_rich()
        if not rich:
            if manual:
                self._put("log", "📋 No image or files on the clipboard")
            return False
        kind, data = rich
        if kind == "image":
            key = self._image_key(data)
            if not manual and key == self._last_clip_image:
                return False
            self._last_clip_image = key
            targets = self._image_capable_phones()
            if not targets:
                if self._auth_targets():
                    self._put("log", "📋 Update the Use As app on the phone to receive images")
                return False
            import io
            buf = io.BytesIO()
            data.save(buf, "PNG")
            png = buf.getvalue()
            if len(png) > self.MAX_CLIPBOARD_IMAGE_BYTES:
                self._put("log", f"📋 Copied image too large ({len(png) // 1024 // 1024} MB)")
                return False
            self._send_bytes_reliably(targets, bytes((protocol.BIN_CLIPBOARD_IMAGE,)) + png)
            self._put("log", f"🖼️ Copied image sent to phone ({data.width}×{data.height})")
            return True

        files = [f for f in data if os.path.isfile(f)]
        skipped = len(data) - len(files)
        key = tuple(sorted(files))
        if not files or (not manual and key == self._last_clip_files):
            return False
        self._last_clip_files = key
        if not self._auth_targets():
            return False
        total = sum(os.path.getsize(f) for f in files)
        if not manual and (len(files) > self.MAX_AUTO_FILES or total > self.MAX_AUTO_FILES_BYTES):
            self._put("log", f"📋 {len(files)} copied file(s), {total // 1024 // 1024} MB — too much to "
                             f"auto-send. Use \"Send copied image/files\" instead.")
            return False
        note = f" ({skipped} folder(s) skipped)" if skipped else ""
        self._put("log", f"📎 Sending {len(files)} copied file(s) to the phone{note}")

        def send_all():
            for f in files:
                self.send_file_to_phone_thread(f)
        threading.Thread(target=send_all, daemon=True, name="clip-files").start()
        return True

    # ---------------------------------------------------------------
    # NOW PLAYING / MEDIA CONTROLS
    # ---------------------------------------------------------------
    def _now_playing_backend(self):
        if not self._np_backend_checked:
            self._np_backend_checked = True
            self._np_backend, reason = nowplaying.make_backend()
            if self._np_backend is None:
                self._put("log", f"ℹ️ Song titles on the phone need: {reason} "
                                 f"(play/pause/next still work)")
        return self._np_backend

    def _media_subscribe(self, websocket, on):
        """A phone wants (or no longer wants) what's playing (control thread)."""
        if on:
            self._media_clients.add(websocket)
        else:
            self._media_clients.discard(websocket)
        backend = self._now_playing_backend()
        if backend is None:
            return
        if self._np_watcher is None and self._media_clients:
            self._np_watcher = nowplaying.NowPlayingWatcher(
                backend, self._on_now_playing, self._on_now_playing_art,
                log=lambda m: self._put("log", m))
        if self._np_watcher is not None:
            self._np_watcher.set_active(bool(self._media_clients))
            if on:
                self._np_watcher.resend()

    def _media_targets(self):
        return [ws for ws in self._snapshot(self._media_clients) if ws in self._authenticated_clients]

    def _on_now_playing(self, info):
        msg = protocol.encode("now_playing", info or {})
        for ws in self._media_targets():
            self._send_to_one(ws, msg)

    def _on_now_playing_art(self, jpeg):
        targets = self._media_targets()
        if targets:
            self._send_bytes_reliably(targets, bytes((protocol.BIN_NOW_PLAYING_ART,)) + jpeg)

    MEDIA_KEYS = {"play_pause": "playpause", "next": "nexttrack", "previous": "prevtrack"}

    def _media_control(self, action, position=None):
        """Play/pause, next, previous, seek from the phone (control thread). Uses the player's
        own controls when possible, else the media keys."""
        if self._np_watcher is not None and self._np_watcher.command(action, position):
            return
        key = self.MEDIA_KEYS.get(action)
        if key:
            try:
                self.platform.key_event(key, [], "press")
            except Exception as e:
                self._put("log", f"⚠️ Media key failed: {e}")

    def _send_bytes_reliably(self, targets, data):
        """Unlike screen frames, a clipboard image must not be dropped when the socket is busy."""
        loop = self.loop
        if loop is None or not loop.is_running():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._send_to(targets, data), loop)
        except RuntimeError:
            pass

    def _receive_phone_clipboard_image(self, data):
        """Phone -> PC: an image copied on the phone (control thread; decoding can take a moment)."""
        try:
            import io
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = 80_000_000
            image = Image.open(io.BytesIO(data))
            image.load()
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGBA" if "A" in image.mode or image.mode == "P" else "RGB")
        except Exception as e:
            self._put("log", f"⚠️ Couldn't read the image from the phone: {e}")
            return
        key = self._image_key(image)
        if key == self._last_clip_image:
            return                          # our own image coming back
        self._last_clip_image = key
        if self.platform.set_clipboard_image(image):
            self._clip_seq = self._clipboard_sequence()   # don't auto-sync it straight back
            self._last_clip = None
            self._put("log", f"🖼️ Image copied from phone ({image.width}×{image.height}) — paste it anywhere")
        else:
            self._put("log", "⚠️ This PC couldn't put the image on its clipboard")

    # ---------------------------------------------------------------
    # AUTH HELPERS
    # ---------------------------------------------------------------
    @staticmethod
    def _snapshot(collection):
        """Copy a set/dict that another thread may be changing (avoids
        'changed size during iteration' when called from capture/audio threads)."""
        for _ in range(5):
            try:
                return tuple(collection)
            except RuntimeError:
                continue
        return ()

    def _auth_targets(self):
        """Connected sockets that passed the password check (all of them if no password)."""
        auth = self._snapshot(self._authenticated_clients)
        return [ws for ws in self._snapshot(self.clients) if ws in auth]

    def _refresh_udp_allowlist(self):
        ips = set()
        for ws in list(self._authenticated_clients):
            try:
                ips.add(ws.remote_address[0])
            except Exception:
                pass
        self._allowed_udp_ips = frozenset(ips)

    def _credentials_ok(self, supplied):
        """Password or QR pairing token (constant-time comparison)."""
        import hmac
        supplied = (supplied or "").encode()
        for secret in (self.pairing_password, self.pairing_token):
            if secret and hmac.compare_digest(supplied, secret.encode()):
                return True
        return False

    def _mark_authenticated(self, ws):
        new = ws not in self._authenticated_clients
        self._authenticated_clients.add(ws)
        self._auth_attempts.pop(ws, None)
        self._refresh_udp_allowlist()
        if new:
            self._put("client_authenticated", None)   # GUI sends any queued "Send to phone" files

    def _send_to_clients_threadsafe(self, message: str):
        """Thread-safe method to queue messages."""
        # Only put if loop is running AND queue exists
        if self._loop and self._loop.is_running() and self._broadcast_queue:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_queue.put(message),
                self._loop
            )


    def get_local_ip(self):
        """Get local IP address with a robust offline fallback."""
        try:
            # 1. Best effort: Try connecting to an external IP (requires internet)
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            # 2. Fallback: Offline router/LAN environment
            try:
                hostname = socket.gethostname()
                local_ips = socket.gethostbyname_ex(hostname)[2]
                # Filter out loopback IPs and grab the first active LAN IP
                valid_ips = [ip for ip in local_ips if not ip.startswith("127.")]
                if valid_ips:
                    return valid_ips[0]
                return "127.0.0.1"
            except Exception:
                return "127.0.0.1"

    def _put(self, kind, payload):
        """Helper to post updates to the GUI thread via queue."""
        try:
            self.update_queue.put((kind, payload))
        except Exception:
            pass


    # ---------------------------------------------------------------
    # PROTOCOL / CONNECTION HANDLER
    # ---------------------------------------------------------------
    BASE_FEATURES = ("auth", "hello", "clipboard_autosync", "file_transfer_part", "display_subscribe",
                     "mouse_zoom", "audio_rate_switch", "binary_camera", "binary_mic_pcm",
                     "qr_pairing", "transfer_progress", "gamepad_player", "clipboard_image")

    def codec_features(self):
        """Phase 3 features, depending on whether PyAV (FFmpeg) is installed."""
        if self._feature_cache is None:
            feats = ["binary_pc_audio", "skip_static_frames", "keyframe_request", "media_control"]
            if self._now_playing_backend() is not None:
                feats.append("now_playing")
            try:
                if media.h264_supported():
                    feats.append("screen_h264")
                if media.opus_supported():
                    feats += ["audio_opus", "mic_opus"]
            except Exception:
                pass
            if not media.available():
                self._put("log", f"ℹ️ {media.unavailable_reason()} — second screen uses JPEG and "
                                 f"audio is sent uncompressed. Install it with: pip install av")
            self._feature_cache = feats
        return list(self._feature_cache)

    def server_hello(self):
        return {
            "protocol": protocol.PROTOCOL_VERSION,
            "min_protocol": protocol.MIN_CLIENT_PROTOCOL,
            "server_version": APP_VERSION,
            "os": self.platform.name,
            "features": list(self.BASE_FEATURES) + list(self.platform.features) + self.codec_features(),
            "auth_required": bool(self.require_password),
        }

    def make_handler(self):
        """One async handler per phone: pairing, the v2 handshake and message routing."""

        async def handler(websocket):
            remote_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
            remote_ip = websocket.remote_address[0]

            # 1. Client record. protocol stays 1 until the phone says hello (old apps never do).
            self.loop = asyncio.get_running_loop()
            self.clients[websocket] = {"ip": remote_ip, "name": f"Unknown ({remote_ip})",
                                       "protocol": 1, "features": [], "app_version": ""}
            self._put("client_connected", self.clients[websocket])
            self._put("log", f"🔌 Client connected: {remote_addr}")
            self._put("client_count", len(self.clients))

            # Our hello goes first, even before pairing, so an outdated app/server pair can
            # tell the user. Old apps ignore unknown message types.
            try:
                await websocket.send(protocol.encode("hello", self.server_hello()))
            except Exception:
                pass

            if not self.require_password:
                self._mark_authenticated(websocket)
            else:
                self._auth_attempts[websocket] = 0

                async def send_auth_required(ws=websocket):
                    try:
                        await ws.send(protocol.encode("auth_required", ""))
                    except Exception:
                        pass
                asyncio.create_task(send_auth_required())

                async def enforce_auth_timeout(ws=websocket):
                    await asyncio.sleep(30)
                    if ws not in self._authenticated_clients:
                        try:
                            await ws.close(code=4001, reason="Authentication required")
                        except Exception:
                            pass
                asyncio.create_task(enforce_auth_timeout())

            # v1 apps wait for "server_version" a moment after connecting.
            async def send_version_delayed():
                await asyncio.sleep(1.5)
                try:
                    await websocket.send(protocol.encode("server_version", APP_VERSION))
                except Exception:
                    pass
                info = self.clients.get(websocket)
                if info is not None and info["protocol"] < protocol.PROTOCOL_VERSION:
                    self._put("log", f"📱 {remote_ip} runs an older Use As app — update it from the Play Store "
                                     f"for the newest features.")
            asyncio.create_task(send_version_delayed())

            try:
                async for message in websocket:
                    try:
                        if isinstance(message, (bytes, bytearray)):
                            if bytes(message[:1]) != b"{":
                                if websocket in self._authenticated_clients:
                                    self._handle_binary(bytes(message), websocket)
                                continue
                            message = bytes(message).decode("utf-8", "replace")   # JSON sent as binary

                        try:
                            msg_type, payload = protocol.decode(message)
                        except protocol.ProtocolError as e:
                            self._put("log", f"⚠️ Ignored malformed message: {e}")
                            continue

                        if msg_type == "hello":
                            info = protocol.client_hello_info(payload)
                            self.clients[websocket].update(
                                protocol=info["protocol"], features=info["features"],
                                app_version=info["app_version"])
                            if info["device"]:
                                self.clients[websocket]["name"] = info["device"]
                            self._put("log", f"🤝 {info['device'] or remote_ip}: app {info['app_version'] or '?'}, "
                                             f"protocol {info['protocol']}")
                            continue

                        if msg_type == "pair":
                            supplied = str(payload).strip()
                            if not self.require_password or self._credentials_ok(supplied):
                                self._mark_authenticated(websocket)
                                self._put("log", f"✅ Client authenticated: {remote_ip}")
                                try:
                                    await websocket.send(protocol.encode("auth_result", "ok"))
                                except Exception:
                                    pass
                            else:
                                self._auth_attempts[websocket] = self._auth_attempts.get(websocket, 0) + 1
                                self._put("log", f"❌ Wrong password from: {remote_ip} "
                                                 f"(attempt {self._auth_attempts[websocket]})")
                                try:
                                    await websocket.send(protocol.encode("auth_result", "fail"))
                                except Exception:
                                    pass
                            continue

                        if websocket not in self._authenticated_clients:
                            continue   # everything else waits for pairing

                        await self._route(msg_type, payload, websocket)

                    except Exception as e:
                        self._put("log", f"⚠️ Message Error: {e}")

            except websockets.ConnectionClosedOK:
                pass
            except Exception as e:
                self._put("log", f"⚠️ Handler error: {e}")
            finally:
                self._authenticated_clients.discard(websocket)
                self._auth_attempts.pop(websocket, None)
                self._refresh_udp_allowlist()
                self._video_busy.discard(websocket)
                self._h264_need_key.discard(websocket)
                self._h264_inflight.pop(websocket, None)
                self._rumble_levels.pop(websocket, None)
                if websocket in self.clients:
                    self._put("client_disconnected", self.clients[websocket])
                    del self.clients[websocket]
                    self._put("client_count", len(self.clients))

                # Stop whatever this phone had running.
                self._abort_incoming_files(websocket)
                if websocket is self._mic_client:
                    self._stop_phone_mic()
                self._submit(self._control_executor, self._on_client_gone, websocket)
                # Unmount its controller on the input thread, after its queued input.
                self._submit(self._input_executor, self._release_gamepad, websocket, remote_ip)
                self._put("log", f"🔌 Disconnected: {remote_ip}")

        return handler

    async def _route(self, msg_type, payload, websocket):
        """Dispatch one decoded message from an authenticated phone."""
        if msg_type == "device_info":
            name = payload.get("name")
            if name:
                self.clients[websocket]["name"] = str(name)[:80]

        elif msg_type == "video_frame":
            self._queue_video_frame(payload)

        # --- input: one ordered worker thread, never the event loop ---
        elif msg_type == "mouse_move":
            self._submit(self._input_executor, self._handle_mouse_move, payload)
        elif msg_type == "mouse_click":
            self._submit(self._input_executor, self._handle_mouse_click, payload)
        elif msg_type == "mouse_scroll":
            self._submit(self._input_executor, self._handle_mouse_scroll, payload)
        elif msg_type == "mouse_zoom":
            self._submit(self._input_executor, self._handle_mouse_zoom, payload)
        elif msg_type == "key_press":
            self._submit(self._input_executor, self._handle_key_press, payload)
        elif msg_type == "gamepad_state":
            self._submit(self._input_executor, self._handle_gamepad_state, payload, websocket)

        elif msg_type == "ack":
            self.ack_event.set()

        # --- phone microphone ---
        elif msg_type == "mic_start":
            self._mic_rate = payload["rate"]
            self._mic_codec = payload.get("codec", "pcm")
            self._mic_opus = None          # fresh decoder state for the new stream
            self._mic_client = websocket
            self._mic_active = True
            self._logged_mic_err = False
            self._mic_reopen = True       # mic thread reopens at the new rate
            self._drain_mic_queue()
            self._put("log", f"🎤 Mic started at {self._mic_rate}Hz"
                             + (" (Opus)" if self._mic_codec == "opus" else ""))
        elif msg_type == "mic_stop":
            self._stop_phone_mic()
            self._put("log", "🎤 Mic Stopped")
        elif msg_type == "audio_frame":
            if websocket is self._mic_client:
                self._handle_audio_frame(payload)

        # --- virtual camera / PC audio / display ---
        elif msg_type == "system_control":
            self._submit(self._control_executor, self._handle_system_control, payload)
        elif msg_type == "audio_control":
            if payload["action"] == "start":
                self._submit(self._control_executor, self._phone_audio_start, websocket, payload["rate"],
                             payload.get("codec", ""))
            else:
                self._submit(self._control_executor, self._phone_audio_stop, websocket)
        elif msg_type == "display_request":
            self._submit(self._control_executor, self._handle_display_request, payload, websocket)
        elif msg_type == "media_subscribe":
            self._submit(self._control_executor, self._media_subscribe, websocket, bool(payload.get("on", True)))
        elif msg_type == "media_control":
            self._submit(self._control_executor, self._media_control, str(payload.get("action", "")),
                         payload.get("position"))

        # --- clipboard & files ---
        elif msg_type == "clipboard_text":
            text = payload["text"]
            if text:
                self._last_clip = text   # don't auto-sync it straight back
                try:
                    pyperclip.copy(text)
                    self._put("clipboard", text)
                    self._put("log", "📋 Text copied from phone")
                except Exception as e:
                    self._put("log", f"⚠️ Clipboard error: {e}")
        elif msg_type == "file_transfer":
            self._handle_file_transfer(payload, websocket)

        elif msg_type == "heartbeat":
            await websocket.send(protocol.encode("heartbeat", "pong"))

    def _handle_binary(self, data, websocket):
        """Binary frames from the phone (protocol v2). The first byte is the type."""
        if not data:
            return
        kind, body = data[0], data[1:]
        if kind == protocol.BIN_CAMERA_JPEG:
            self._queue_video_frame({"jpeg": body})
        elif kind == protocol.BIN_MIC_PCM:
            if websocket is self._mic_client and self._mic_active:
                self._queue_mic_pcm(body)
        elif kind == protocol.BIN_MIC_OPUS:
            if websocket is self._mic_client and self._mic_active:
                self._decode_mic_opus(body)
        elif kind == protocol.BIN_PHONE_CLIPBOARD_IMAGE:
            if len(body) <= self.MAX_CLIPBOARD_IMAGE_BYTES:
                self._submit(self._control_executor, self._receive_phone_clipboard_image, body)

    def _decode_mic_opus(self, packet):
        """Phone mic Opus packet -> 16-bit mono PCM at the mic output rate (event loop;
        decoding 10-20 ms of Opus takes well under a millisecond)."""
        try:
            if self._mic_opus is None:
                self._mic_opus = media.OpusDecoder(channels=1)
            pcm = self._mic_opus.decode(packet, out_rate=self._mic_rate, out_channels=1)
        except Exception as e:
            if not self._logged_mic_err:
                self._logged_mic_err = True
                self._put("log", f"❌ Mic Opus decode error: {e}")
            return
        if pcm:
            self._queue_mic_pcm(pcm)

    def send_file_to_phone_thread(self, file_path):
        # `with` so a failure before/while acquiring can never release another transfer's lock.
        with self.sending_lock:
            try:
                filename = safe_filename(os.path.basename(file_path))
                file_size = os.path.getsize(file_path)

                loop = self._loop
                targets = self._auth_targets()
                if not (loop and loop.is_running()) or not targets:
                    self._put("log", "⚠️ No connected phone to send the file to")
                    return

                self._put("log", f"📤 Starting transfer: {filename}")

                def send(data, timeout=30):
                    """Send to every target and WAIT until written: this is the back-pressure
                    that stops a big file from piling up in memory on slow Wi-Fi."""
                    fut = asyncio.run_coroutine_threadsafe(self._send_to(targets, data), loop)
                    fut.result(timeout=timeout)

                # 1. START SIGNAL (JSON text) on the same path as the chunks, so it can't be overtaken.
                start_payload = json.dumps({"filename": filename, "size": file_size})
                send(json.dumps({"type": "file_start", "payload": start_payload}))

                # 2. RAW DATA (binary, 0x01 header)
                total_sent = 0
                last_report = 0.0
                self._report_transfer("out", filename, 0, file_size)
                with open(file_path, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk:
                            break
                        send(b'\x01' + chunk)
                        total_sent += len(chunk)
                        now = time.time()
                        if now - last_report >= 0.2:
                            last_report = now
                            self._report_transfer("out", filename, total_sent, file_size)

                # 3. END SIGNAL (0x02)
                send(b'\x02')
                self._report_transfer("out", filename, total_sent, file_size, done=True)
                self._put("log", f"✅ Sent {filename} ({total_sent} bytes)")

            except Exception as e:
                self._report_transfer("out", os.path.basename(str(file_path)), 0, 0, done=True, failed=True)
                self._put("log", f"❌ File send error: {e}")
                import traceback
                traceback.print_exc()

    def _report_transfer(self, direction, name, done_bytes, total, done=False, failed=False):
        """Progress for the Sharing tab. total is 0 when unknown (older phone apps)."""
        self._put("transfer", {"dir": direction, "name": name, "bytes": done_bytes,
                               "total": total, "done": done, "failed": failed})

    async def _send_to(self, targets, data):
        """Send one message to the given sockets (skipping ones that have since gone away)."""
        for client in targets:
            if client not in self.clients:
                continue
            try:
                await client.send(data)
            except Exception:
                pass

    async def _broadcast_bytes(self, data):
        """Send raw bytes to all authenticated clients."""
        await self._send_to(self._auth_targets(), data)

    def _handle_file_transfer(self, data, websocket=None):
        """Phone -> PC file chunks.

        Each transfer writes to its own '<name>.part' file and is renamed to a free name
        ('photo (1).jpg' if 'photo.jpg' exists) when the end marker arrives. Previously every
        chunk was appended to SAVE_DIR/<name>, so re-sending a file doubled it and a failed
        transfer got glued onto the next one; the name was also used unsanitised.
        """
        key = None
        try:
            if isinstance(data, str):
                data = json.loads(data)

            filename = safe_filename(data.get("filename"), fallback=f"unknown_{int(time.time())}.dat")
            b64_data = data.get("data")
            is_end = data.get("is_end")
            key = (id(websocket), filename)

            entry = self._incoming_files.get(key)
            if entry is None and (b64_data or is_end):
                SAVE_DIR.mkdir(parents=True, exist_ok=True)
                part = unique_path(SAVE_DIR, filename + ".part")
                try:
                    total = int(data.get("size") or 0)   # new apps send the file size
                except (TypeError, ValueError):
                    total = 0
                entry = {"part": part, "fh": open(part, "wb"), "name": filename, "ws": websocket,
                         "bytes": 0, "total": total, "reported": 0.0}
                self._incoming_files[key] = entry

            if b64_data:
                chunk = base64.b64decode(b64_data)
                entry["fh"].write(chunk)
                entry["bytes"] += len(chunk)
                now = time.time()
                if now - entry["reported"] >= 0.2:
                    entry["reported"] = now
                    self._report_transfer("in", filename, entry["bytes"], entry["total"])

            if is_end and entry is not None:
                self._incoming_files.pop(key, None)
                entry["fh"].close()
                self._report_transfer("in", filename, entry["bytes"], entry["total"] or entry["bytes"], done=True)
                final_path = unique_path(SAVE_DIR, filename)
                os.replace(entry["part"], final_path)
                self._put("log", f"📂 Saved: {final_path.name}")
                self._put("log", f"📍 Path: {final_path.absolute()}")
                self._put("file_received", str(final_path))

        except Exception as e:
            self._put("log", f"❌ File receive error: {e}")
            if key is not None:
                self._discard_incoming(key)

    def _discard_incoming(self, key):
        entry = self._incoming_files.pop(key, None)
        if not entry:
            return
        self._report_transfer("in", entry["name"], entry.get("bytes", 0), entry.get("total", 0),
                              done=True, failed=True)
        try:
            entry["fh"].close()
        except Exception:
            pass
        try:
            os.remove(entry["part"])
        except Exception:
            pass

    def _abort_incoming_files(self, websocket):
        """Drop half-received files from a phone that disconnected mid-transfer."""
        for key in [k for k, v in self._incoming_files.items() if v.get("ws") is websocket]:
            self._put("log", f"⚠️ Incomplete transfer discarded: {self._incoming_files[key]['name']}")
            self._discard_incoming(key)


    def _place_on_canvas(self, image, w, h):
        """Centers image on black canvas. Handles resolution changes dynamically."""
        import numpy as np

        # 1. FIX: Check if canvas exists AND if it matches the new target size
        if not hasattr(self, '_canvas') or self._canvas.shape[0] != h or self._canvas.shape[1] != w:
            self._canvas = np.zeros((h, w, 3), dtype=np.uint8)
            # Force reset of dimensions logic
            self._last_dims = None

        ih, iw = image.shape[:2]
        y = (h - ih) // 2
        x = (w - iw) // 2

        # 2. Clear previous area if the image size/position changed
        if hasattr(self, '_last_dims') and self._last_dims != (x, y, iw, ih):
            self._canvas.fill(0)

        self._last_dims = (x, y, iw, ih)

        # 3. Safety Check: Ensure the image actually fits
        # If the input image is somehow larger than the target, crop it
        if ih > h or iw > w:
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
            ih, iw = h, w
            y, x = 0, 0

        self._canvas[y:y + ih, x:x + iw] = image
        return self._canvas

    def _queue_video_frame(self, payload):
        """Called on the event loop: just park the newest frame for the video thread."""
        with self._video_lock:
            self._pending_video = payload   # an older, unprocessed frame is simply dropped
        self._video_event.set()

    def _video_worker(self):
        while self._video_worker_running:
            if not self._video_event.wait(timeout=0.5):
                continue
            with self._video_lock:
                payload = self._pending_video
                self._pending_video = None
                self._video_event.clear()
            if payload is not None:
                self._handle_video_frame(payload)

    def _handle_video_frame(self, payload):
        try:
            if isinstance(payload, str):
                try:
                    frame_data = json.loads(payload)
                except:
                    return
            else:
                frame_data = payload

            # v2 phones send raw JPEG bytes (binary frame); v1 phones send base64 in JSON.
            jpeg_bytes = frame_data.get('jpeg')
            if jpeg_bytes is None:
                b64_data = frame_data.get('data')
                if not b64_data: return
                jpeg_bytes = base64.b64decode(b64_data)
            # No 'rotation'/'is_front' handling: the phone pre-rotates frames.
            nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None: return

            # --- USE DYNAMIC TARGET RESOLUTION ---
            TARGET_W, TARGET_H = self.target_w, self.target_h

            h, w = frame.shape[:2]

            # Smart resize logic
            scale = min(TARGET_W / w, TARGET_H / h)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # --- MANUAL PC ADJUSTMENTS ONLY ---
            if self.is_mirrored: 
                frame = cv2.flip(frame, 1) # Horizontal UI Toggle
            if self.is_flipped: 
                frame = cv2.flip(frame, 0) # Vertical UI Toggle

            if self.brightness_boost != 0:
                frame = cv2.convertScaleAbs(frame, alpha=1, beta=self.brightness_boost)

            # --- PROCESS BACKGROUND ---
            if self.bg_mode != "none":
                frame = self._process_background(frame)

            # --- COLOR CONVERT ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            final_frame = self._place_on_canvas(rgb_frame, TARGET_W, TARGET_H)

            self._last_frame = final_frame

            # --- PREVIEW (FULL RESOLUTION) ---
            if self.preview_active:
                self.latest_preview_frame = rgb_frame.copy()

            self._frame_count += 1

        except Exception as e:
            if not getattr(self, "_logged_frame_err", False):
                self._logged_frame_err = True
                self._put("log", f"⚠️ Camera frame error: {e}")

    def _process_background(self, img):
        try:
            import cv2
            import numpy as np
            import mediapipe as mp
            import os

            # --- SETUP (Same as before) ---
            if self.segmentor is None:
                # Bundled into the exe by UseAsServer.spec (lands in _MEIPASS); also look
                # next to the exe / script for dev runs.
                candidates = [
                    resource_path("selfie_segmenter.tflite"),
                    os.path.join(os.path.dirname(sys.executable), "selfie_segmenter.tflite"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "selfie_segmenter.tflite"),
                ]
                model_path = next((p for p in candidates if os.path.exists(p)), None)

                if model_path is None:
                    if not hasattr(self, "_logged_model_missing"):
                        # The exe has no console, so print() was invisible: use the GUI log.
                        self._put("log", "❌ Background effects unavailable: selfie_segmenter.tflite not found")
                        self._logged_model_missing = True
                    return img

                BaseOptions = mp.tasks.BaseOptions
                VisionRunningMode = mp.tasks.vision.RunningMode
                ImageSegmenter = mp.tasks.vision.ImageSegmenter
                ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions

                options = ImageSegmenterOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=VisionRunningMode.IMAGE,
                    output_category_mask=True
                )
                self.segmentor = ImageSegmenter.create_from_options(options)

            # --- PROCESSING ---

            # 1. Resize for AI Speed
            ai_w, ai_h = 320, 180
            small_frame = cv2.resize(img, (ai_w, ai_h))

            # 2. Get AI Result
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
            result = self.segmentor.segment(mp_image)

            # 3. Create Mask
            small_mask = result.category_mask.numpy_view() > 0.1
            full_mask = cv2.resize(small_mask.astype(np.float32), (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
            mask_3d = np.stack((full_mask > 0.5,) * 3, axis=-1)

            # --- APPLY EFFECTS (FLIPPED LOGIC) ---

            if self.bg_mode == "blur":
                blurred = cv2.GaussianBlur(img, (55, 55), 0)

               
                return np.where(mask_3d, blurred, img)

            elif self.bg_mode == "image" and self.bg_image_path:
                if self.bg_image_cache is None:
                    bg = cv2.imread(self.bg_image_path)
                    if bg is not None:
                        self.bg_image_cache = cv2.resize(bg, (img.shape[1], img.shape[0]))

                if self.bg_image_cache is not None:
                    if self.bg_image_cache.shape != img.shape:
                        self.bg_image_cache = cv2.resize(self.bg_image_cache, (img.shape[1], img.shape[0]))

                   
                    return np.where(mask_3d, self.bg_image_cache, img)

            return img

        except Exception as e:
            if not hasattr(self, "_logged_bg_error"):
                self._put("log", f"❌ Background effect error: {e}")
                self._logged_bg_error = True
            return img

    def _resize_with_black_bars(self, image, target_w, target_h):
        import cv2
        import numpy as np

        h, w = image.shape[:2]

        # Scale to fit HEIGHT exactly
        scale = target_h / h
        new_w = int(w * scale)
        new_h = target_h

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center in black canvas
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2

        if x_offset >= 0:
            canvas[:, x_offset:x_offset + new_w] = resized
            return canvas

        # Fallback (shouldn't happen normally)
        return resized


    def start_virtual_camera(self, device=None):
        if self._vcam_running:
            self._put("log", "⚠️ Virtual Camera is already running.")
            return

        try:
            import pyvirtualcam
        except ImportError:
            self._put("log", "❌ Critical: Run 'pip install pyvirtualcam'")
            return

        device = self.platform.resolve_vcam_device(device)
        cam = None
        try:
            try:
                cam = pyvirtualcam.Camera(width=self.target_w, height=self.target_h, fps=30,
                                          device=device, fmt=pyvirtualcam.PixelFormat.RGB)
            except Exception:
                # Fallback to auto-detect
                cam = pyvirtualcam.Camera(width=self.target_w, height=self.target_h, fps=30,
                                          device=None, fmt=pyvirtualcam.PixelFormat.RGB)
        except Exception as e:
            self._put("log", f"❌ No Virtual Camera found: {e}")
            self._put("log", f"💡 {self.platform.vcam_help()}")
            self._vcam_running = False
            self._vcam = None
            return

        self._vcam = cam
        self._vcam_running = True
        threading.Thread(target=self._vcam_loop, args=(cam,), daemon=True).start()
        self._put("log", f"📹 VCam Active: {cam.device}")

    def stop_virtual_camera(self):
        """Stop virtual webcam. The loop thread notices and closes its own camera."""
        if self._vcam:
            self._vcam_running = False
            self._vcam = None
            self._last_frame = None
            self._put("log", "📹 Virtual camera stopped")

    def _vcam_loop(self, cam):
        """Feeds one camera instance. Owns closing it, so stop/start can't double-close it
        or leave this loop calling methods on None."""
        blank = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
        try:
            while self._vcam_running and self._vcam is cam:
                frame = self._last_frame if self._last_frame is not None else blank
                if frame.shape[1] != cam.width or frame.shape[0] != cam.height:
                    frame = cv2.resize(frame, (cam.width, cam.height))
                cam.send(frame)
                time.sleep(0.03)
        except Exception as e:
            self._put("log", f"⚠️ Virtual camera error: {e}")
            if self._vcam is cam:
                self._vcam_running = False
                self._vcam = None
                self._put("camera_status", None)
        finally:
            try:
                cam.close()
            except Exception:
                pass

    def _handle_system_control(self, payload):
        """Virtual camera start/stop requested by the phone (runs on the control thread)."""
        try:
            control_data = json.loads(payload) if isinstance(payload, str) else payload
            action = control_data.get("action")
            if action == "start_vcam":
                device_name = control_data.get("device")
                self._put("log", f"📲 Mobile requested Auto-Start: {device_name}")
                self.start_virtual_camera(device_name)
                self._put("camera_status", device_name if self._vcam_running else None)
            elif action == "stop_vcam":
                self._put("log", "📲 Mobile requested Stop Virtual Cam")
                self.stop_virtual_camera()
                self._put("camera_status", None)
        except Exception as e:
            self._put("log", f"⚠️ System control error: {e}")

    # ---------------------------------------------------------------
    # PHONE MICROPHONE -> PC (played into VB-CABLE)
    # ---------------------------------------------------------------

    def _handle_audio_frame(self, payload):
        """Called on the event loop: decode and queue only. The blocking PyAudio write
        happens on the mic thread, so the mic no longer stalls clicks/keys/gamepad."""
        if not self._mic_active or not isinstance(payload, (str, bytes)):
            return
        try:
            audio_bytes = base64.b64decode(payload)
        except Exception:
            return
        self._queue_mic_pcm(audio_bytes)

    def _queue_mic_pcm(self, audio_bytes):
        try:
            self._mic_queue.put_nowait(audio_bytes)
        except queue.Full:
            # Network burst: drop the oldest chunk rather than let latency build up.
            try:
                self._mic_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._mic_queue.put_nowait(audio_bytes)
            except queue.Full:
                pass

    def _drain_mic_queue(self):
        while True:
            try:
                self._mic_queue.get_nowait()
            except queue.Empty:
                return

    def _stop_phone_mic(self):
        self._mic_active = False
        self._mic_client = None
        self._mic_reopen = True   # mic thread closes its stream, releasing VB-CABLE
        self._drain_mic_queue()

    def _open_mic_stream(self, pyaudio):
        """Open playback into the OS's virtual-mic device (VB-CABLE / UseAs_Mic sink /
        BlackHole) at self._mic_rate. Returns the stream; sets self._mic_out_channels."""
        if self._mic_player is None:
            # Created once and never terminated (terminating it broke every later session).
            self._mic_player = pyaudio.PyAudio()
        device_index, channels, label = self.platform.mic_output(self._mic_player)
        stream = self._mic_player.open(format=pyaudio.paInt16, channels=channels, rate=self._mic_rate,
                                       output=True, output_device_index=device_index)
        self._mic_out_channels = channels
        self.platform.after_mic_stream_opened()
        self._put("log", f"🎤 Routing Phone Mic to: {label} @ {self._mic_rate}Hz")
        return stream

    def _close_mic_stream(self):
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop_stream()
            except Exception:
                pass
            try:
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None

    def _mic_worker(self):
        try:
            import pyaudio
        except ImportError:
            return
        while self._video_worker_running:
            try:
                chunk = self._mic_queue.get(timeout=0.5)
            except queue.Empty:
                chunk = None

            if self._mic_reopen:
                self._mic_reopen = False
                self._close_mic_stream()
            if chunk is None or not self._mic_active:
                continue

            try:
                if self._mic_stream is None:
                    self._mic_stream = self._open_mic_stream(pyaudio)
                if getattr(self, "_mic_out_channels", 1) == 2:
                    # phone mic is mono; stereo devices (BlackHole) would play it in one ear
                    chunk = np.repeat(np.frombuffer(chunk, dtype=np.int16), 2).tobytes()
                self._mic_stream.write(chunk)
            except Exception as e:
                if not self._logged_mic_err:     # reset on every mic_start
                    self._put("log", f"❌ Mic Playback Error: {e}")
                    self._logged_mic_err = True
                self._close_mic_stream()
                time.sleep(0.2)
        self._close_mic_stream()


    # ---------------------------------------------------------------
    # GAMEPAD
    # ---------------------------------------------------------------
    def _handle_gamepad_state(self, payload, client_ws):
        """Gamepad input -> this phone's virtual controller (runs on the input thread)."""
        try:
            state = json.loads(payload) if isinstance(payload, str) else payload
            if client_ws not in self.clients:
                return  # phone already gone; don't mount a pad for it

            pad = self.client_gamepads.get(client_ws)
            if pad is None:
                # Created on first use, so phones that never open the gamepad screen
                # don't add phantom controllers that a game might pick as player 1.
                def on_rumble(intensity, ws=client_ws):
                    previous = self._rumble_levels.get(ws, 0.0)
                    self._rumble_levels[ws] = intensity
                    if previous != intensity:
                        self._send_rumble(ws, intensity)

                pad = self.platform.create_gamepad(on_rumble)
                if pad is None:
                    return
                self.client_gamepads[client_ws] = pad
                ip = self.clients.get(client_ws, {}).get("ip", "?")
                self._put("log", f"🎮 {pad.description} mounted for {ip}")
                threading.Thread(target=self._report_player_number, args=(client_ws, pad),
                                 daemon=True, name="pad-player").start()

            pad.apply(normalize_gamepad_state(state))
        except Exception as e:
            self._put("log", f"❌ Gamepad error: {e}")

    def _report_player_number(self, ws, pad):
        """Tell the phone which player its controller is (P1-P4). Windows assigns the slot a
        moment after the controller is plugged in, so check a few times."""
        sent = None
        for delay in (0.3, 0.7, 1.0, 2.0, 4.0):
            time.sleep(delay)
            if self.client_gamepads.get(ws) is not pad:
                return   # phone left / pad replaced
            try:
                number = pad.player_number()
            except Exception:
                number = None
            if number and number != sent:
                sent = number
                self._send_to_one(ws, protocol.encode("gamepad_player", str(number)))
                ip = self.clients.get(ws, {}).get("ip", "?")
                self._put("log", f"🎮 {ip} is Player {number}")

    def _send_to_one(self, ws, message):
        loop = self.loop
        if loop is None or not loop.is_running() or ws not in self.clients:
            return

        async def _send():
            try:
                await ws.send(message)
            except Exception:
                pass
        try:
            asyncio.run_coroutine_threadsafe(_send(), loop)
        except Exception:
            pass

    def _send_rumble(self, ws, intensity):
        """Rumble goes only to the phone that owns the controller (thread-safe)."""
        loop = self.loop
        if loop is None or not loop.is_running() or ws not in self.clients:
            return
        msg = protocol.encode("rumble", str(intensity))

        async def _send():
            try:
                await ws.send(msg)
            except Exception:
                pass
        try:
            asyncio.run_coroutine_threadsafe(_send(), loop)
        except Exception:
            pass

    async def _rumble_keepalive(self):
        """The game only reports rumble changes, but the phone stops vibrating 600 ms after
        the last message (its safety net for a dropped link). Re-send active rumble every
        200 ms so a long rumble isn't cut off."""
        while True:
            await asyncio.sleep(0.2)
            for ws, level in list(self._rumble_levels.items()):
                if level > 0.05 and ws in self.clients:
                    try:
                        await ws.send(protocol.encode("rumble", str(level)))
                    except Exception:
                        pass

    def _release_gamepad(self, websocket, remote_ip):
        """Unmount a disconnected phone's controller (runs on the input thread)."""
        pad = self.client_gamepads.pop(websocket, None)
        self._rumble_levels.pop(websocket, None)
        if pad is None:
            return
        try:
            pad.close()   # releases everything held, then unplugs
        except Exception as e:
            self._put("log", f"⚠️ Error clearing gamepad: {e}")
        self._put("log", f"🎮 Virtual Gamepad cleanly unmounted for {remote_ip}")

    def _handle_display_request(self, payload, websocket=None):
        """Display streaming request (runs on the control thread).

        Phones that send start_display are subscribers; frames go only to them, and the
        capture stops when the last one sends stop_display or disconnects.
        """
        try:
            request = json.loads(payload) if isinstance(payload, str) else payload
            action = request.get('action')

            if action == 'request_keyframe':
                # The phone lost a frame (or restarted its decoder): resync it.
                if websocket is not None:
                    self._h264_need_key.add(websocket)
                self._h264_key_wanted = True
                return

            if action in ['start_display', 'change_resolution']:
                self._display_width = int(request.get('width', 1280))
                self._display_height = int(request.get('height', 720))
                self._display_fps = int(request.get('fps', 30))
                self._display_quality = int(request.get('quality', 35))

                # Default to [0] (Primary) if missing
                indices = request.get('monitor_indices', [0])
                try:
                    indices = [int(i) for i in indices] or [0]
                except Exception:
                    indices = [0]
                self._display_monitor_indices = indices

                # Handle legacy single index if sent by old app version
                if 'monitor_index' in request and 'monitor_indices' not in request:
                    idx = int(request.get('monitor_index'))
                    self._display_monitor_indices = [idx] if idx >= 0 else [0, 1]  # Fallback for "All"

                self._put("log", f"🖥️ Config: {self._display_width}x{self._display_height} (Monitors: {self._display_monitor_indices})")

            if action in ('start_display', 'change_resolution') and websocket is not None:
                self._set_display_codec(websocket, str(request.get('format', 'jpeg')).lower())

            if action == 'start_display':
                if websocket is not None:
                    self._display_clients.add(websocket)
                self._ensure_requested_monitors()
                self._start_display_capture()

            elif action == 'stop_display':
                if websocket is not None:
                    self._display_clients.discard(websocket)
                if not self._display_clients:
                    self._stop_display_capture()
                    self._schedule_virtual_cleanup()

        except Exception as e:
            self._put("log", f"❌ Display request error: {e}")

    def _set_display_codec(self, websocket, wanted):
        """Pick this phone's screen codec and tell it which one it got."""
        codec = "jpeg"
        if wanted == "h264":
            if self._h264_broken:
                pass
            elif "screen_h264" in self.codec_features():
                codec = "h264"
        previous = self._display_codecs.get(websocket)
        self._display_codecs[websocket] = codec
        if codec == "h264":
            self._h264_need_key.add(websocket)      # start (or restart) at a keyframe
            self._h264_key_wanted = True
        if wanted == "h264" or (previous is not None and previous != codec):
            self._send_to_one(websocket, protocol.encode("display_config", {"codec": codec}))
        if wanted == "h264" and codec == "jpeg" and previous != "jpeg":
            why = "no working H.264 encoder" if self._h264_broken else media.unavailable_reason() or "H.264 unavailable"
            self._put("log", f"🖥️ Second screen uses JPEG ({why})")

    def _ensure_requested_monitors(self):
        """If the phone asked for a monitor that doesn't exist, add a virtual one (Windows:
        usbmmidd driver, Linux: xrandr), so the phone becomes a real extra screen."""
        self._cancel_virtual_cleanup()
        try:
            import mss
            with mss.mss() as sct:
                count = len(sct.monitors) - 1
            needed = max(self._display_monitor_indices) + 1
            if needed > count:
                if not self.virtual_display_enabled:
                    self._put("log", f"⚠️ Monitor {needed} requested but only {count} found "
                                     f"(virtual monitor is off in Settings).")
                    return
                self._put("log", f"🖥️ Monitor {needed} requested but only {count} found — adding a virtual one")
                with self._virtual_lock:
                    self.platform.ensure_display_count(needed, width=self._display_width,
                                                       height=self._display_height)
            elif self.platform.virtual_displays_added:
                # Already there (e.g. the phone reconnected): just match the resolution.
                with self._virtual_lock:
                    self.platform.ensure_display_count(needed, width=self._display_width,
                                                       height=self._display_height)
        except Exception as e:
            self._put("log", f"⚠️ Monitor check failed: {e}")

    VIRTUAL_DISPLAY_GRACE = 20.0   # seconds without a viewer before the virtual monitor goes

    def _schedule_virtual_cleanup(self):
        """Remove the virtual monitor once no phone is viewing. Not instantly: switching tabs
        or a Wi-Fi hiccup shouldn't make every window on it jump back and forth."""
        if not self.platform.virtual_displays_added:
            return
        self._cancel_virtual_cleanup()

        def run():
            if self._display_clients:
                return
            with self._virtual_lock:
                if not self._display_clients:
                    self.platform.remove_virtual_displays()

        t = threading.Timer(self.VIRTUAL_DISPLAY_GRACE, run)
        t.daemon = True
        self._virtual_cleanup_timer = t
        t.start()

    def _cancel_virtual_cleanup(self):
        t = self._virtual_cleanup_timer
        self._virtual_cleanup_timer = None
        if t is not None:
            t.cancel()

    def _start_display_capture(self):
        if getattr(self, '_display_thread', None) is not None and self._display_thread.is_alive():
            self._stop_display_capture(wait_seconds=0.8)
        # Each capture thread gets its own generation, so an old thread that didn't exit
        # in time can never keep running next to the new one.
        self._display_gen = getattr(self, '_display_gen', 0) + 1
        self._display_active = True
        self._display_thread = threading.Thread(target=self._capture_screen_loop,
                                                args=(self._display_gen,), daemon=True)
        self._display_thread.start()

    def _phone_audio_start(self, websocket, sample_rate, codec="json"):
        """PC audio requested by a phone (control thread)."""
        codec = codec or "json"
        if codec == "opus" and (self._opus_broken or "audio_opus" not in self.codec_features()):
            codec = "pcm"
        self._audio_codecs[websocket] = codec
        label = {"opus": "Opus", "pcm": "PCM"}.get(codec, "PCM/JSON")
        self._put("log", f"📲 Mobile requested Audio Start ({sample_rate}Hz, {label})")
        if codec != "json":
            # Tells a phase-3 app what it will get (e.g. PCM when PyAV isn't installed).
            self._send_to_one(websocket, protocol.encode("audio_config", {
                "codec": codec, "rate": media.OPUS_RATE if codec == "opus" else int(sample_rate)}))
        self._audio_clients.add(websocket)
        if self._streaming_audio:
            if int(sample_rate) != int(self._audio_target_rate):
                # Was ignored before ("already streaming"): restart at the new rate.
                self._audio_target_rate = int(sample_rate)
                self._audio_restart.set()
        else:
            self.start_audio_streaming(sample_rate, by_phone=True)
        self._put("audio_status", True)

    def _phone_audio_stop(self, websocket):
        self._put("log", "📲 Mobile requested Audio Stop")
        self._audio_clients.discard(websocket)
        self._audio_codecs.pop(websocket, None)
        # Only stop when no other phone is still listening.
        if not self._audio_clients and self._audio_started_by_phone:
            self.stop_audio_streaming()
            self._put("audio_status", False)

    def _on_client_gone(self, websocket):
        """A phone disconnected: stop the streams nobody is consuming any more (control thread)."""
        if websocket in self._display_clients:
            self._display_clients.discard(websocket)
            if not self._display_clients and self._display_active:
                self._put("log", "🖥️ Screen viewer disconnected — stopping capture")
                self._stop_display_capture()
            if not self._display_clients:
                self._schedule_virtual_cleanup()
        self._display_codecs.pop(websocket, None)
        self._audio_codecs.pop(websocket, None)
        if websocket in self._media_clients:
            self._media_subscribe(websocket, False)
        if websocket in self._audio_clients:
            self._audio_clients.discard(websocket)
            if not self._audio_clients and self._streaming_audio and self._audio_started_by_phone:
                self._put("log", "🔊 Audio listener disconnected — stopping audio stream")
                self.stop_audio_streaming()
                self._put("audio_status", False)

    def _probe_monitors(self, sct, target_indices):
        """One-shot diagnostic: geometry + brightness as THIS process sees it."""
        import numpy as np
        try:
            self._put("log", f"🔍 MSS sees {len(sct.monitors) - 1} monitor(s)")
            for idx in target_indices:
                mss_idx = idx + 1
                if mss_idx >= len(sct.monitors):
                    self._put("log", f"🔍 Monitor {idx + 1}: does not exist")
                    continue
                m = sct.monitors[mss_idx]
                raw = np.array(sct.grab(m))
                self._put("log",
                          f"🔍 Monitor {idx + 1}: {m['width']}x{m['height']} "
                          f"@ {m['left']},{m['top']} — brightness {raw.mean():.1f}")
        except Exception as e:
            self._put("log", f"⚠️ Probe failed: {e}")

    def _stop_display_capture(self, wait_seconds: float = 1.0):
        """Stop display capture thread (if running) and wait a short time for it to exit."""
        try:
            # Flip the active flag — capture loop checks this and should exit
            self._display_active = False

            # If thread exists, join with timeout to let it exit cleanly
            if hasattr(self, '_display_thread') and self._display_thread is not None:
                if self._display_thread.is_alive():
                    # wait a small amount for graceful stop
                    self._put("log", "🔄 Stopping previous screen capture...")
                    self._display_thread.join(timeout=wait_seconds)

                # If still alive, we just log and allow it to die (daemon threads end with process)
                if self._display_thread.is_alive():
                    self._put("log", "⚠️ Previous screen capture thread did not stop immediately (continuing).")
                else:
                    self._put("log", " Previous screen capture stopped.")
        except Exception as e:
            self._put("log", f"❌ Error stopping display capture: {e}")
            
            
            
            
    

    def _capture_screen_loop(self, gen=None):
        """Continuously capture specific monitor or combined screens (Flicker-Free using MSS)."""
        try:
            import mss
            import cv2
            import time
            import numpy as np
 
            target_indices = self._display_monitor_indices
            self._put("log", f"🖥️ Screen capture started via MSS. Targets: {target_indices}")
 
            last_mouse_pos = (0, 0)
            last_move_time = time.time()
            HIDE_TIMEOUT = 3.0
 
            # Last frame actually handed to the encoders, for skipping unchanged frames.
            last_sent_frame = None
            last_sent_at = 0.0

            # --- DIAGNOSTICS STATE ---
            # First check after 3 s: a monitor that was just added starts out black.
            last_black_check = time.time() + 2.0
            black_warned = False
            warned_missing = False
 
            with mss.mss() as sct:
                # sct.monitors[0] is all screens combined
                # sct.monitors[1] is monitor 1, sct.monitors[2] is monitor 2
                self._probe_monitors(sct, target_indices)
 
                frame_cache = [None] * len(target_indices)
 
                while self._display_active and (gen is None or gen == self._display_gen):
                    start_time = time.time()
 
                    target_w = self._display_width
                    target_h = self._display_height
                    target_fps = self._display_fps
                    frame_duration = 1.0 / max(1, target_fps)
 
                    # Rebuilt every frame so quality changes apply immediately
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),
                                    int(getattr(self, '_display_quality', 35))]
 
                    try:
                        # 1. CAPTURE & CACHE
                        frames_ready = []
 
                        for i, idx in enumerate(target_indices):
                            mss_idx = idx + 1  # app index (0-based) -> MSS index (1-based)
 
                            if mss_idx < len(sct.monitors):
                                monitor = sct.monitors[mss_idx]
                                sct_img = sct.grab(monitor)
                                # mss returns BGRA, convert to BGR for OpenCV
                                img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
 
                                frame_cache[i] = img
                                frames_ready.append(img)
                            else:
                                if not warned_missing:
                                    warned_missing = True
                                    self._put("log",
                                              f"⚠️ Monitor {idx + 1} doesn't exist — this PC has "
                                              f"{len(sct.monitors) - 1}. Pick another one in the app.")
                                frames_ready.append(frame_cache[i])
 
                        if any(f is None for f in frames_ready):
                            time.sleep(0.05)
                            continue
 
                        # 2. STITCHING (For Multi-Monitor Selection)
                        if len(frames_ready) > 1:
                            base_h = frames_ready[0].shape[0]
                            resized_frames = []
                            for f in frames_ready:
                                if f.shape[0] != base_h:
                                    aspect = f.shape[1] / f.shape[0]
                                    new_w = int(base_h * aspect)
                                    f = cv2.resize(f, (new_w, base_h))
                                resized_frames.append(f)
                            final_frame = np.hstack(resized_frames)
                        else:
                            final_frame = frames_ready[0]
 
                        # --- BLACK FRAME DETECTION (sampled, ~1 Hz) ---
                        if time.time() - last_black_check > 1.0:
                            last_black_check = time.time()
                            if final_frame[::16, ::16].max() < 8:
                                if not black_warned:
                                    black_warned = True
                                    self._put("log",
                                              "⚠️ Captured screen is black. The monitor may be asleep, "
                                              "a virtual display, driven by a second GPU, or showing "
                                              "protected content.")
                            elif black_warned:
                                black_warned = False
                                self._put("log", "🖥️ Screen capture recovered.")
 
                        # 3. DRAW MOUSE (Single Monitor Mode Only)
                        if len(target_indices) == 1:
                            try:
                                mx, my = self.platform.cursor_position()
                                if (mx, my) != last_mouse_pos:
                                    last_mouse_pos = (mx, my)
                                    last_move_time = time.time()
 
                                if time.time() - last_move_time < HIDE_TIMEOUT:
                                    monitor = sct.monitors[target_indices[0] + 1]
                                    local_mx = mx - monitor["left"]
                                    local_my = my - monitor["top"]
 
                                    if 0 <= local_mx < final_frame.shape[1] and 0 <= local_my < final_frame.shape[0]:
                                        cursor_pts = np.array([
                                            [0, 0],     # Tip
                                            [0, 16],    # Left bottom corner
                                            [4, 12],    # Inner left
                                            [7, 19],    # Tail bottom left
                                            [10, 17],   # Tail bottom right
                                            [7, 11],    # Inner right
                                            [12, 11]    # Right corner
                                        ], np.int32)
 
                                        pts = (cursor_pts + [local_mx, local_my]).reshape((-1, 1, 2))
                                        cv2.fillPoly(final_frame, [pts], (255, 255, 255))
                                        cv2.polylines(final_frame, [pts], isClosed=True,
                                                      color=(0, 0, 0), thickness=1,
                                                      lineType=cv2.LINE_AA)
                            except Exception:
                                pass
 
                        # 4. FIT TO SCREEN
                        h, w = final_frame.shape[:2]
                        scale = min(target_w / w, target_h / h)

                        if scale < 1.0:
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            # Linear (area for big reductions) keeps text readable; nearest
                            # made small fonts break up.
                            final_frame = cv2.resize(final_frame, (new_w, new_h),
                                                     interpolation=cv2.INTER_AREA if scale < 0.6
                                                     else cv2.INTER_LINEAR)

                        # 5. SKIP UNCHANGED FRAMES, ENCODE & SEND
                        # A still desktop costs nothing: identical frames aren't encoded or sent.
                        # One frame a second still goes out, so a phone that missed the last
                        # change (dropped on slow Wi-Fi) catches up.
                        now = time.time()
                        changed = (last_sent_frame is None
                                   or last_sent_frame.shape != final_frame.shape
                                   or not np.array_equal(last_sent_frame, final_frame))
                        if changed or self._h264_key_wanted or now - last_sent_at >= 1.0:
                            self._send_screen_frame(final_frame, target_fps, encode_param)
                            last_sent_frame = final_frame
                            last_sent_at = now

                        # 6. FPS LIMIT
                        elapsed = time.time() - start_time
                        remaining = frame_duration - elapsed
                        if remaining > 0:
                            time.sleep(remaining)
 
                    except Exception as e:
                        if not getattr(self, '_logged_capture_err', False):
                            self._logged_capture_err = True
                            self._put("log", f"⚠️ Capture loop error: {e}")
                        time.sleep(0.1)
 
            self._put("log", "🖥️ Screen capture stopped")
 
        except Exception as e:
            self._put("log", f"❌ Capture Fatal Error: {e}")
            self._display_active = False

    H264_MAX_INFLIGHT = 4      # H.264 packets queued per phone before frames are dropped

    def _send_screen_frame(self, frame, fps, encode_param):
        """Encode one screen frame once per codec in use and send it to the subscribers."""
        clients = self._snapshot(self._display_clients)
        h264_targets = [ws for ws in clients if self._display_codecs.get(ws) == "h264"]
        jpeg_targets = [ws for ws in clients if self._display_codecs.get(ws) != "h264"]

        if not h264_targets:
            self._h264_key_wanted = False      # only matters to H.264 phones
        if h264_targets and not self._h264_broken:
            try:
                with self._h264_lock:          # an old capture thread may still be finishing
                    self._encode_and_send_h264(frame, fps, h264_targets)
            except Exception as e:
                self._h264_broken = True
                self._h264_encoder = None
                self._put("log", f"⚠️ H.264 stopped ({e}) — the second screen switches to JPEG")
                for ws in h264_targets:
                    self._display_codecs[ws] = "jpeg"
                    self._send_to_one(ws, protocol.encode("display_config", {"codec": "jpeg"}))
                jpeg_targets += h264_targets
        elif h264_targets:
            jpeg_targets += h264_targets

        if jpeg_targets:
            ok, buffer = cv2.imencode('.jpg', frame, encode_param)
            if ok:
                # Header byte 0x03 means "This is a Video Frame"
                self._send_bytes_to_clients_threadsafe(b'\x03' + buffer.tobytes(), jpeg_targets)

    def _encode_and_send_h264(self, frame, fps, targets):
        # Every H.264 phone is still busy with earlier frames: don't encode at all. Nothing is
        # lost (no frame was produced), the next capture simply carries the newer picture.
        inflight = self._h264_inflight
        if all(inflight.get(ws, 0) >= self.H264_MAX_INFLIGHT for ws in targets):
            return
        if self._h264_encoder is None:
            self._h264_encoder = media.H264Encoder(log=lambda m: self._put("log", m))
            self._h264_scale = 1.0
            self._h264_scale_changed = time.time()
        enc = self._h264_encoder

        # Congestion control: drops lower the bitrate; a quiet spell slowly restores it.
        now = time.time()
        if self._h264_congested_at > self._h264_scale_changed and now - self._h264_scale_changed > 2.0:
            new_scale = max(0.35, self._h264_scale * 0.7)
            if new_scale < self._h264_scale:
                self._h264_scale = new_scale
                self._put("log", f"🐢 Wi-Fi can't keep up — second screen bitrate lowered to "
                                 f"{int(new_scale * 100)}%")
            self._h264_scale_changed = now
        elif self._h264_scale < 1.0 and now - max(self._h264_scale_changed, self._h264_congested_at) > 15.0:
            self._h264_scale = min(1.0, self._h264_scale + 0.15)
            self._h264_scale_changed = now

        force = False
        if self._h264_key_wanted and now - self._h264_last_forced >= 0.5:
            self._h264_key_wanted = False
            self._h264_last_forced = now
            force = True

        quality = max(5, int(self._display_quality * self._h264_scale))
        for data, is_key in enc.encode(frame, fps=fps, quality=quality, force_key=force):
            self._send_h264_threadsafe(media.screen_packet(data, is_key, enc.width, enc.height),
                                       is_key, targets)

    def _send_h264_threadsafe(self, packet, is_key, targets):
        """H.264 frames depend on the ones before them, so unlike JPEG a frame can't just be
        skipped. A phone whose queue is full misses this frame and then waits (frames are
        held back) until the next keyframe, which is requested right away."""
        loop = self.loop
        if loop is None or not loop.is_running():
            return
        inflight = self._h264_inflight

        async def send_one(ws):
            try:
                await ws.send(packet)
            except Exception:
                pass
            finally:
                n = inflight.get(ws, 1) - 1
                if n <= 0:
                    inflight.pop(ws, None)
                else:
                    inflight[ws] = n

        def schedule():
            for ws in targets:
                if ws not in self.clients or ws not in self._authenticated_clients:
                    continue
                if ws in self._h264_need_key:
                    if not is_key:
                        self._h264_key_wanted = True
                        continue
                    self._h264_need_key.discard(ws)
                if inflight.get(ws, 0) >= self.H264_MAX_INFLIGHT:
                    self._h264_need_key.add(ws)
                    self._h264_key_wanted = True
                    self._h264_congested_at = time.time()
                    continue
                inflight[ws] = inflight.get(ws, 0) + 1
                asyncio.ensure_future(send_one(ws))

        try:
            loop.call_soon_threadsafe(schedule)
        except RuntimeError:
            pass

    def _send_bytes_to_clients_threadsafe(self, data_bytes, targets=None):
        """Thread-safe binary send with back-pressure.

        A socket that is still busy with the previous frame is skipped for this one, so on
        slow Wi-Fi frames are dropped instead of piling up in memory (which made the
        display lag grow without limit). Only authenticated sockets are ever targeted.
        """
        loop = self.loop
        if loop is None or not loop.is_running():
            return
        pool = targets if targets is not None else self._authenticated_clients

        async def send_one(ws):
            try:
                await ws.send(data_bytes)
            except Exception:
                pass
            finally:
                self._video_busy.discard(ws)

        def schedule():
            for ws in list(pool):
                if ws not in self.clients or ws not in self._authenticated_clients:
                    continue
                if ws in self._video_busy:
                    continue          # previous frame still being written: drop this one
                self._video_busy.add(ws)
                asyncio.ensure_future(send_one(ws))

        try:
            loop.call_soon_threadsafe(schedule)
        except RuntimeError:
            pass  # loop closed


    # ---------------------------------------------------------------
    # PC AUDIO -> PHONE
    # ---------------------------------------------------------------
    def start_audio_streaming(self, target_rate=48000, by_phone=False):
        if self._streaming_audio:
            self._put("log", "🔊 Audio already streaming")
            return

        self._streaming_audio = True
        self._audio_started_by_phone = by_phone
        self._audio_target_rate = int(target_rate)
        self._audio_restart.clear()
        # A new generation per start, so a thread from a previous start that hasn't
        # exited yet can never keep streaming alongside the new one.
        self._audio_gen += 1
        my_gen = self._audio_gen

        def still_wanted():
            return self._streaming_audio and self._audio_gen == my_gen

        def emit(pcm, rate, channels):
            """Called from the audio thread (or a C callback): must be quick."""
            # Phones that asked for audio; if it was started from the PC's button,
            # every authenticated phone (the old behaviour).
            targets = [c for c in self._snapshot(self._audio_clients) if c in self._authenticated_clients] \
                or (self._auth_targets() if not self._audio_started_by_phone else [])
            loop = self.loop
            if not targets or loop is None or not loop.is_running():
                return
            per_client = self._encode_pc_audio(pcm, rate, channels, targets)
            if per_client:
                try:
                    loop.call_soon_threadsafe(self._schedule_audio_sends, per_client)
                except RuntimeError:
                    pass

        def manager():
            try:
                while still_wanted():
                    try:
                        self.platform.run_loopback_capture(self._audio_target_rate, still_wanted,
                                                           self._audio_restart, emit)
                        if self._audio_restart.is_set():
                            self._audio_restart.clear()
                            if still_wanted():
                                self._put("log", f"🔄 Switching audio to {self._audio_target_rate}Hz")
                    except ImportError as e:
                        self._put("log", f"❌ Audio streaming unavailable: {e}")
                        break
                    except Exception as e:
                        self._put("log", f"❌ Audio Error: {e}")
                        time.sleep(1.0)
            finally:
                # If the thread dies, don't stay stuck in "already streaming" forever.
                if self._audio_gen == my_gen:
                    self._streaming_audio = False
                self._put("log", "🔇 Audio Stopped")

        self._audio_thread = threading.Thread(target=manager, daemon=True, name="pc-audio")
        self._audio_thread.start()

    def _encode_pc_audio(self, pcm, rate, channels, targets):
        """One chunk of captured PC audio -> {websocket: [messages]} in each phone's codec.
        Runs on the capture thread; every codec is encoded once, whatever the phone count."""
        codecs = self._audio_codecs
        cache = {}
        out = {}
        for ws in targets:
            codec = codecs.get(ws, "json")
            if codec == "opus" and self._opus_broken:
                codec = "pcm"
            if codec not in cache:
                cache[codec] = self._audio_messages(codec, pcm, rate, channels)
            if cache[codec]:
                out[ws] = cache[codec]
        return out

    def _audio_messages(self, codec, pcm, rate, channels):
        if codec == "opus":
            try:
                if self._opus_encoder is None:
                    self._opus_encoder = media.OpusEncoder()
                packets = self._opus_encoder.encode(pcm, rate, channels)
                ch = bytes((self._opus_encoder.channels,))
                return [bytes((protocol.BIN_PC_AUDIO_OPUS,)) + ch + p for p in packets]
            except Exception as e:
                self._opus_broken = True
                self._put("log", f"⚠️ Opus encoder failed ({e}) — sending uncompressed audio")
                for ws, c in list(self._audio_codecs.items()):
                    if c == "opus":
                        self._audio_codecs[ws] = "pcm"
                        self._send_to_one(ws, protocol.encode("audio_config", {"codec": "pcm", "rate": rate}))
                codec = "pcm"
        if codec == "pcm":
            header = bytes((protocol.BIN_PC_AUDIO_PCM, channels & 0xFF)) + int(rate).to_bytes(4, "big")
            return [header + pcm]
        return [json.dumps({"type": "audio_frame", "rate": rate, "channels": channels,
                            "payload": base64.b64encode(pcm).decode("ascii")})]

    def _schedule_audio_sends(self, per_client):
        """Event loop: queue each phone's audio messages in order."""
        async def send_all(ws, messages):
            try:
                for m in messages:
                    await ws.send(m)
            except Exception:
                pass
        for ws, messages in per_client.items():
            if ws in self.clients:
                asyncio.ensure_future(send_all(ws, messages))

    def stop_audio_streaming(self):
        """Stop audio streaming."""
        self._streaming_audio = False
        self._audio_clients.clear()
        self._audio_restart.set()   # wake the capture loop right away
        if self._audio_thread:
            self._audio_thread.join(timeout=2)
        self._put("log", "🔊 Audio streaming stopped")

    # ---------------------------------------------------------------
    # INPUT (runs on the input thread; the OS work is in self.platform)
    # ---------------------------------------------------------------
    @staticmethod
    def _obj(payload):
        return json.loads(payload) if isinstance(payload, str) else (payload or {})

    def _handle_mouse_move(self, payload):
        try:
            data = self._obj(payload)
            dx, dy = int(data.get('deltaX', 0)), int(data.get('deltaY', 0))
            if dx or dy:
                self.platform.mouse_move_rel(dx, dy)
        except Exception:
            pass

    def _handle_mouse_click(self, payload):
        try:
            event = self._obj(payload)
            self.platform.mouse_button(int(event.get("button", 0)), event.get("action", "click"),
                                       gaming_mode=self.gaming_mode)
        except Exception as e:
            self._put("log", f"❌ Mouse click error: {e}")

    def _handle_mouse_scroll(self, payload):
        try:
            delta = float(self._obj(payload).get("scrollDelta", 0))
            if delta:
                self.platform.mouse_scroll(delta)
        except Exception as e:
            self._put("log", f"❌ Scroll error: {e}")

    def _handle_mouse_zoom(self, payload):
        """Two-finger pinch -> zoom (Ctrl+wheel; Cmd +/- on macOS). 10 units = one notch;
        the fraction is kept so slow pinches still add up."""
        try:
            self._zoom_accum += float(self._obj(payload).get("scrollDelta", 0)) / 10.0
            notches = int(self._zoom_accum)
            if notches:
                self._zoom_accum -= notches
                self.platform.mouse_zoom(notches)
        except Exception as e:
            self._put("log", f"❌ Zoom error: {e}")

    def _submit(self, executor, fn, *args):
        """Run fn on a worker thread; never raise into the event loop."""
        try:
            executor.submit(fn, *args)
        except RuntimeError:
            pass  # executor shut down (server stopping)

    def _handle_key_press(self, payload):
        try:
            event = self._obj(payload)
            key = str(event.get("key", ""))
            modifiers = list(event.get("modifiers") or [])
            action = event.get("action", "press")
            key_lower = key.lower()
            if not key:
                return

            if key_lower in ("brightnessup", "brightnessdown"):
                if action != "up":
                    threading.Thread(target=self.platform.change_brightness,
                                     args=(key_lower == "brightnessup",), daemon=True).start()
                return

            # Not on a US keyboard (é, emoji, हिन्दी): typed as Unicode.
            if len(key) == 1 and ord(key) > 127 and not modifiers:
                self.platform.type_unicode(key, action)
                return

            # Upper-case letters need Shift.
            if len(key) == 1 and key.isupper() and key.isalpha():
                if "shift" not in [m.lower() for m in modifiers]:
                    modifiers.append("shift")

            self.platform.key_event(key, modifiers, action, gaming_mode=self.gaming_mode)
        except Exception as e:
            self._put("log", f"❌ Key error: {e}")

    async def _async_starter(self):
        handler = self.make_handler()

        # Add ping_interval=None to stop auto-pings from causing jitter
        self._ws_server = await websockets.serve(
            handler,
            self.host,
            self.port,
            max_size=20 * 1024 * 1024,
            ping_interval=None
        )

        self.loop = asyncio.get_running_loop()
        try:
            self._broadcast_task = asyncio.create_task(self._broadcast_worker())
            self._rumble_task = asyncio.create_task(self._rumble_keepalive())
            self._put("log", "Broadcast worker started")
        except Exception as e:
            self._put("log", f"⚠️ Failed to start broadcast worker: {e}")

        return self._ws_server

    def _run_loop(self):
        """Thread target: create loop, start server, run forever."""
        # FIX 4: Create new loop for this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # FIX 5: Create the Queue INSIDE the loop that uses it
        self._broadcast_queue = asyncio.Queue()

        try:
            self._ws_server = self._loop.run_until_complete(self._async_starter())
            self._put("log", f"🌐 WebSocket server listening on {self.host}:{self.port}")
            self._loop.run_forever()
        except Exception as e:
            self._put("log", f"❌ Server thread error: {e}")
        finally:
            self._loop.run_until_complete(self._shutdown_server(run_from_finally=True))
            self._loop.close()

    async def _shutdown_server(self, run_from_finally=False):
        """Coroutine to gracefully close connections & the server."""
        
        
        
        for ws, dev in list(self.client_gamepads.items()):
            try:
                dev.unregister_notification()
            except:
                pass
        self.client_gamepads.clear()
        if self._ws_server is not None:
            try:
                self._ws_server.close()
                await self._ws_server.wait_closed()
            except Exception as e:
                self._put("log", f"Error closing ws server: {e}")
        

        # Close all connected clients
        clients = list(self.clients)
        if clients:
            self._put("log", f"Closing {len(clients)} client connection(s)...")
            try:
                # Create a list of tasks
                close_tasks = [ws.close(code=1001, reason="Server shutting down") for ws in clients]
                # Wait for all of them to complete, with a timeout
                await asyncio.wait_for(asyncio.gather(*close_tasks, return_exceptions=True), timeout=2.0)
            except asyncio.TimeoutError:
                self._put("log", "⚠️ Client close tasks timed out.")
            except Exception as e:
                self._put("log", f"Error closing clients: {e}")

        self.clients.clear()
        self._put("client_count", 0)

        for task in (getattr(self, "_broadcast_task", None), getattr(self, "_rumble_task", None)):
            if task is not None and not task.done():
                task.cancel()

        if not run_from_finally and self._loop.is_running():
            # If called by stop(), we need to stop the loop.
            # If called from _run_loop's finally, the loop is already stopping.
            self._loop.stop()

    def start(self):
        """Start server thread."""
        if self._thread and self._thread.is_alive():
            self._put("log", "⚠️ Server already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="WebSocketServerThread")
        self._thread.start()

    def stop(self):
        """Stop server: schedule shutdown coroutine and stop loop."""

        # 1. STOP UDP MOUSE (Critical Fix)
        if hasattr(self, 'udp_mouse'):
            self.udp_mouse.stop()

        # 2. STOP DISCOVERY
        if hasattr(self, 'discovery'):
            self.discovery.stop()

        # 3. STOP VIRTUAL CAMERA
        if self._vcam_running:
            self.stop_virtual_camera()

        # 4. STOP AUDIO
        if self._streaming_audio:
            self.stop_audio_streaming()

        # 4b. STOP SCREEN CAPTURE, PHONE MIC AND WORKER THREADS
        #     (the capture thread used to keep running after the server stopped)
        self._display_clients.clear()
        self._stop_display_capture(wait_seconds=0.5)
        # Don't leave an invisible monitor behind (windows could get stranded on it).
        self._cancel_virtual_cleanup()
        try:
            if self.platform.virtual_displays_added:
                with self._virtual_lock:
                    self.platform.remove_virtual_displays()
        except Exception as e:
            self._put("log", f"⚠️ Couldn't remove the virtual monitor: {e}")
        self._stop_phone_mic()
        if self._np_watcher is not None:
            self._np_watcher.stop()
        self._video_worker_running = False   # also ends the mic thread
        self._video_event.set()
        self._input_executor.shutdown(wait=False)
        self._control_executor.shutdown(wait=False)

        # 5. STOP WEBSOCKET SERVER
        if not (self._thread and self._thread.is_alive() and self._loop):
            self._put("log", "⚠️ Server thread not active")
            return

        try:
            # Schedule server shutdown coroutine
            asyncio.run_coroutine_threadsafe(self._shutdown_server(), self._loop)
        except Exception as e:
            self._put("log", f"❌ Error while stopping server: {e}")

        # Wait for thread to join
        self._thread.join(timeout=3)

        if self._thread.is_alive():
            self._put("log", "⚠️ Warning: server thread did not exit immediately")
        else:
            self._put("log", "Server stopped")

        self._thread = None
        self._loop = None
        self._ws_server = None
        self.clients.clear()
        self._put("client_count", 0)

    # ============================================
    # ⬆️⬆️⬆️ END OF FIX ⬆️⬆️⬆️
    # ============================================

    def cleanup(self):
        """Alias for stop."""
        self.stop()


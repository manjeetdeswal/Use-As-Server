import base64
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import customtkinter as ctk
from PIL import Image
import threading
import asyncio
import json
import platform
import sys
from pathlib import Path
import logging
import socket
import queue
import time
import websockets  # pip install websockets
import pyautogui   # pip install pyautogui
import sys
import socket
import threading
import time
import pyperclip  # For clipboard
import os         # For file saving
import webbrowser      # For opening GitHub
import urllib.request  # For checking updates
import winreg          # For Auto-Start with Windows
import cv2
import pystray                   # pip install pystray
from pystray import MenuItem as item
import numpy as np
from PIL import Image, ImageDraw, ImageTk

if sys.platform == "win32":
    import win32api
    import ctypes
    from ctypes import windll, wintypes
# --- CONSTANTS ---
APP_VERSION = "1.1"
GITHUB_REPO = "manjeetdeswal/Use-As-Server" # ⚠️ CHANGE THIS to your actual "user/repo"
GITHUB_URL = f"https://github.com/{GITHUB_REPO}"
SETTINGS_FILE = Path.home() / "Downloads" / "UseAs_Received" / "server_settings.json"
# ============================================
# SERVER CODE (async websockets running in a background thread)
# ============================================

COLORS = {
    "bg": "#1e1e1e",
    "fg": "#ffffff",
    "accent": "#00e676",      # Android Green
    "accent_hover": "#00c853",
    "secondary": "#2d2d2d",
    "highlight": "#3d3d3d",
    "error": "#cf6679",
    "text_dim": "#b0b0b0",
    "btn_disabled": "#424242"
}


SAVE_DIR = Path.home() / "Downloads" / "UseAs_Received"

# Create directory immediately
try:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📂 Saving files to: {SAVE_DIR}")
except Exception as e:
    print(f"❌ Error creating folder: {e}")




class DiscoveryServer(threading.Thread):
    def __init__(self, port=8080):
        super().__init__()
        self.port = port  # This is the TCP port the phone should connect to
        self.running = True

    def get_broadcast_addresses(self):
        """Find broadcast address for every interface."""
        addresses = set()
        try:
            addresses.add('<broadcast>')
            hostname = socket.gethostname()
            local_ips = socket.gethostbyname_ex(hostname)[2]
            for ip in local_ips:
                parts = ip.split('.')
                if len(parts) == 4:
                    broadcast = f"{parts[0]}.{parts[1]}.{parts[2]}.255"
                    addresses.add(broadcast)
        except:
            pass
        return list(addresses)

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.2)

        # BROADCAST FORMAT: "UNIFIED_REMOTE_SERVER:PORT"
        message = f"UNIFIED_REMOTE_SERVER:{self.port}".encode('utf-8')
        logging.info(f"Starting discovery broadcast for port {self.port}...")

        while self.running:
            targets = self.get_broadcast_addresses()
            for target in targets:
                try:
                    # We broadcast TO port 8888 (The phone listens on this port)
                    sock.sendto(message, (target, 8888))
                except:
                    pass
            time.sleep(1)

        sock.close()

    def stop(self):
        self.running = False


class UnifiedRemoteServer:
    def __init__(self, host="0.0.0.0", port=8080, gaming_mode=True, update_queue: queue.Queue = None):
        import threading
        self.sending_lock = threading.Lock()


        self.gaming_mode = gaming_mode

        self.host = host
        self.port = port  # Main TCP Port

        self.ack_event = threading.Event()

        self.clients = set()
        self._loop = None
        self._ws_server = None
        self._thread = None
        self.update_queue = update_queue or queue.Queue()
        self._stop_event = threading.Event()

        # 1. START DISCOVERY (Broadcasts the TCP port)
        self.discovery = DiscoveryServer(port=self.port)
        self.discovery.start()

        # 2. START UDP MOUSE (Listens on TCP Port + 1)
        self.udp_port = self.port + 1
        self.udp_mouse = UDPMouseServer(port=self.udp_port)
        self.udp_mouse.daemon = True
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
        self.mp_selfie = None
        self.segmentor = None

        try:
            user32 = ctypes.windll.user32
            self.min_x = user32.GetSystemMetrics(76)
            self.min_y = user32.GetSystemMetrics(77)
            self.max_x = self.min_x + user32.GetSystemMetrics(78)
            self.max_y = self.min_y + user32.GetSystemMetrics(79)
        except:
            self.min_x, self.min_y = 0, 0
            self.max_x, self.max_y = 1920, 1080

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

                disconnected = set()
                for client in list(self.clients):
                    try:
                        await client.send(message)
                    except Exception as e:
                        disconnected.add(client)

                for client in disconnected:
                    if client in self.clients:
                        self.clients.remove(client)
                        self._put("client_count", len(self.clients))
            except Exception as e:
                pass

    def _send_to_clients_threadsafe(self, message: str):
        """Thread-safe method to queue messages."""
        # Only put if loop is running AND queue exists
        if self._loop and self._loop.is_running() and self._broadcast_queue:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_queue.put(message),
                self._loop
            )


    def get_local_ip(self):
        """Get local IP address (best-effort)."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            try:
                # Fallback for systems without external connection
                return socket.gethostbyname(socket.gethostname())
            except Exception:
                return "127.0.0.1"

    def _put(self, kind, payload):
        """Helper to post updates to the GUI thread via queue."""
        try:
            self.update_queue.put((kind, payload))
        except Exception:
            pass

    def make_handler(self):
        """Return an async handler that captures self."""

        async def handler(websocket):
            self.clients.add(websocket)
            remote_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
            self._put("log", f"✅ Client connected: {remote_addr}")
            self._put("client_count", len(self.clients))

            try:
                async for message in websocket:
                    try:
                        if isinstance(message, bytes):
                            try:
                                message = message.decode('utf-8')
                            except UnicodeDecodeError:
                                continue

                        data = json.loads(message)
                        msg_type = data.get("type", "unknown")
                        payload = data.get("payload", "")

                        # --- ROUTING MESSAGES ---
                        if msg_type == "video_frame":
                            self._handle_video_frame(payload)
                        elif msg_type == "mouse_move":
                            self._handle_mouse_move(payload)
                        elif msg_type == "mouse_click":
                            self._handle_mouse_click(payload)
                        elif msg_type == "mouse_scroll":
                            self._handle_mouse_scroll(payload)
                        elif msg_type == "key_press":
                            self._handle_key_press(payload)
                        elif msg_type == "ack":
                            self.ack_event.set()

                        # --- MICROPHONE HANDLING (NEW) ---
                        elif msg_type == "mic_start":
                            try:
                                self._mic_active = True  # <--- ENABLE FLAG
                                new_rate = int(payload)
                                self._put("log", f"🎤 Mic started at {new_rate}Hz")

                                # Close old stream to apply new rate
                                if hasattr(self, '_mic_stream') and self._mic_stream:
                                    try:
                                        self._mic_stream.close()
                                    except:
                                        pass
                                    self._mic_stream = None
                                self._mic_rate = new_rate
                            except Exception as e:
                                self._put("log", f"⚠️ Mic start error: {e}")

                        elif msg_type == "mic_stop":
                            self._mic_active = False  # <--- DISABLE FLAG

                            if hasattr(self, '_mic_stream') and self._mic_stream:
                                try:
                                    self._mic_stream.close()
                                except:
                                    pass
                                self._mic_stream = None
                            self._put("log", "🎤 Mic Stopped")

                        elif msg_type == "audio_frame":
                            self._handle_audio_frame(payload)

                        # --- PC AUDIO STREAMING ---
                        elif msg_type == "audio_control":
                            payload_str = str(payload)
                            if payload_str == "start" or payload_str.startswith("start|"):
                                self._put("log", "📲 Mobile requested Audio Start...")
                                sample_rate = 48000
                                if "|" in payload_str:
                                    try:
                                        parts = payload_str.split("|", 1)
                                        if len(parts) > 1:
                                            config = json.loads(parts[1])
                                            sample_rate = int(config.get("rate", 48000))
                                            self._put("log", f"⚙️ Config received: {sample_rate}Hz")
                                    except Exception as e:
                                        self._put("log", f"⚠️ Config parse failed: {e}")
                                threading.Thread(target=self.start_audio_streaming, args=(sample_rate,),
                                                 daemon=True).start()
                                self._put("audio_status", True)
                            elif payload_str == "stop":
                                self._put("log", "📲 Mobile requested Audio Stop")
                                self.stop_audio_streaming()
                                self._put("audio_status", False)

                        elif msg_type == "gamepad_state":
                            self._handle_gamepad_state(payload, websocket)

                        # --- CLIPBOARD & FILES ---
                        elif msg_type == "clipboard_text":
                            try:
                                if isinstance(payload, str) and payload.startswith('{'):
                                    inner = json.loads(payload)
                                    text = inner.get("text", "")
                                else:
                                    text = str(payload)
                                if text:
                                    import pyperclip
                                    pyperclip.copy(text)
                                    self._put("clipboard", text)
                                    self._put("log", "📋 Text copied from phone")
                            except:
                                pass

                        elif msg_type == "file_transfer":
                            if isinstance(payload, str):
                                payload = json.loads(payload)
                            self._handle_file_transfer(payload)

                        elif msg_type == "display_request":
                            self._handle_display_request(payload)

                        elif msg_type == "heartbeat":
                            await websocket.send(json.dumps({"type": "heartbeat", "payload": "pong"}))

                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        pass

            except websockets.ConnectionClosedOK:
                pass
            except Exception as e:
                self._put("log", f"⚠️ Handler error: {e}")
            finally:
                if websocket in self.clients:
                    self.clients.remove(websocket)
                if hasattr(self, 'client_gamepads') and websocket in self.client_gamepads:
                    try:
                        del self.client_gamepads[websocket]
                    except:
                        pass
                self._put("log", f"❌ Client disconnected: {remote_addr}")
                self._put("client_count", len(self.clients))

        return handler

    def send_file_to_phone_thread(self, file_path):
        try:
            original_name = os.path.basename(file_path)
            # Remove protocol characters from filename
            filename = original_name.replace(":", "_")
            file_size = os.path.getsize(file_path)

            self._put("log", f"📤 Starting transfer: {filename}")
            print(f"--- BINARY TRANSFER: {filename} ---")

            self.sending_lock.acquire()

            # 1. SEND START SIGNAL (As JSON Text)
            # This tells Android to open the file and get ready.
            # payload is a JSON string containing filename and size
            start_payload = json.dumps({"filename": filename, "size": file_size})
            self.send_to_android("file_start", start_payload)

            # Wait for Android to open the file stream (Critical)
            time.sleep(0.5)

            with open(file_path, "rb") as f:
                total_sent = 0
                chunk_counter = 0

                while True:
                    # Read 64KB chunks (Optimal for TCP)
                    chunk = f.read(64 * 1024)
                    if not chunk:
                        break

                    # 2. SEND RAW DATA (As Binary Frame)
                    # Prefix with 0x01 so Android knows it's a file chunk
                    header = b'\x01'
                    packet = header + chunk

                    # Send Binary
                    if self._loop and self._loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self._broadcast_bytes(packet),
                            self._loop
                        )

                    total_sent += len(chunk)
                    chunk_counter += 1

                    # Tiny sleep to prevent router buffer overflow
                    # 0.005s is fast but safe for binary
                    time.sleep(0.005)

                    # Log progress occasionally
                    if chunk_counter % 50 == 0:
                        print(f"Sent {total_sent} bytes...")

            # 3. SEND END SIGNAL (As Binary Frame)
            # Header 0x02 means "End of File"
            end_packet = b'\x02'
            if self._loop:
                asyncio.run_coroutine_threadsafe(self._broadcast_bytes(end_packet), self._loop)

            print(f"✅ COMPLETE. Sent {total_sent} bytes.")
            self._put("log", f"✅ Sent {filename}")

        except Exception as e:
            print(f"Error: {e}")
            self._put("log", f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.sending_lock.locked():
                self.sending_lock.release()

    # Add this helper method to your class if missing
    async def _broadcast_bytes(self, data):
        """Helper to send raw bytes to all connected clients."""
        for client in list(self.clients):
            try:
                await client.send(data)
            except:
                pass

    def _handle_file_transfer(self, data):
        try:
            # 1. Debug: Check if we got data at all
            # print("DEBUG: Received file packet")

            filename = data.get("filename")
            if not filename:
                filename = f"unknown_{int(time.time())}.dat"

            b64_data = data.get("data")
            is_end = data.get("is_end")

            # 2. Confirm Path
            file_path = SAVE_DIR / filename

            # 3. Write Data
            if b64_data:
                try:
                    # Create directory if missing (sanity check)
                    SAVE_DIR.mkdir(parents=True, exist_ok=True)

                    with open(file_path, "ab") as f:
                        f.write(base64.b64decode(b64_data))

                except Exception as e:
                    print(f"❌ CRITICAL WRITE ERROR: {e}")
                    self._put("log", f"❌ Write Error: {e}")
                    return

            # 4. Handle End of File
            if is_end:
                abs_path = file_path.absolute()
                print(f"✅ COMPLETE: File saved to {abs_path}")
                self._put("log", f"📂 Saved: {filename}")
                self._put("log", f"📍 Path: {abs_path}")
                self._put("file_received", str(file_path))

        except Exception as e:
            print(f"❌ LOGIC ERROR: {e}")
            self._put("log", f"❌ File Logic Error: {e}")

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

    def _handle_video_frame(self, payload):
        try:
            # ... (Parsing and Decoding - Keep existing code) ...
            if isinstance(payload, str):
                try:
                    frame_data = json.loads(payload)
                except:
                    return
            else:
                frame_data = payload

            b64_data = frame_data.get('data')
            if not b64_data: return

            rotation = frame_data.get('rotation', 0)
            is_front = frame_data.get('is_front', False)

            jpeg_bytes = base64.b64decode(b64_data)
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

            # ... (Rotation, Mirroring, Brightness logic - Keep existing code) ...

            # 4. Rotate
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            should_mirror = False
            if is_front:
                if rotation in [90, 270]:
                    frame = cv2.flip(frame, 0)
                else:
                    should_mirror = True

            if self.is_mirrored: should_mirror = not should_mirror
            if should_mirror: frame = cv2.flip(frame, 1)
            if self.is_flipped: frame = cv2.flip(frame, 0)

            if self.brightness_boost != 0:
                frame = cv2.convertScaleAbs(frame, alpha=1, beta=self.brightness_boost)

            # --- PROCESS BACKGROUND ---
            if self.bg_mode != "none":
                frame = self._process_background(frame)

            # --- COLOR CONVERT ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            final_frame = self._place_on_canvas(rgb_frame, TARGET_W, TARGET_H)

            self._last_frame = final_frame

            if self.preview_active:
                # Calculate preview size preserving aspect ratio
                # Max preview size: 640x360
                max_w, max_h = 640, 360
                h, w = final_frame.shape[:2]
                scale = min(max_w / w, max_h / h)

                new_w, new_h = int(w * scale), int(h * scale)
                self.latest_preview_frame = cv2.resize(rgb_frame, (new_w, new_h))

            self._frame_count += 1

        except Exception as e:
            pass

    def _process_background(self, img):
        try:
            import cv2
            import numpy as np
            import mediapipe as mp
            import os

            # --- SETUP (Same as before) ---
            if self.segmentor is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_dir, "selfie_segmenter.tflite")

                if not os.path.exists(model_path):
                    if not hasattr(self, "_logged_model_missing"):
                        print(f"❌ ERROR: Model missing: {model_path}")
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

                # ✅ FIX: SWAPPED THE ORDER HERE
                # BEFORE: np.where(mask_3d, img, blurred) -> Blurred face
                # NOW:    np.where(mask_3d, blurred, img) -> Clears face, blurs background
                return np.where(mask_3d, blurred, img)

            elif self.bg_mode == "image" and self.bg_image_path:
                if self.bg_image_cache is None:
                    bg = cv2.imread(self.bg_image_path)
                    if bg is not None:
                        self.bg_image_cache = cv2.resize(bg, (img.shape[1], img.shape[0]))

                if self.bg_image_cache is not None:
                    if self.bg_image_cache.shape != img.shape:
                        self.bg_image_cache = cv2.resize(self.bg_image_cache, (img.shape[1], img.shape[0]))

                    # ✅ FIX: SWAPPED HERE TOO
                    return np.where(mask_3d, self.bg_image_cache, img)

            return img

        except Exception as e:
            if not hasattr(self, "_logged_bg_error"):
                print(f"DEBUG BG ERROR: {e}")
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

            # 1. Try specific device
            try:
                print(f"DEBUG: Attempting to start '{device}'...")
                self._vcam = pyvirtualcam.Camera(
                    width=self.target_w,
                    height=self.target_h,
                    fps=30,
                    device=device,
                    fmt=pyvirtualcam.PixelFormat.RGB  # <--- CHANGED FROM BGR TO RGB
                )
            except Exception as e:
                # 2. Fallback to Auto-Detect
                print(f"DEBUG: Specific device '{device}' failed. Auto-detecting...")
                self._vcam = pyvirtualcam.Camera(
                    width=self.target_w,
                    height=self.target_h,
                    fps=30,
                    device=None,
                    fmt=pyvirtualcam.PixelFormat.RGB  # <--- CHANGED FROM BGR TO RGB
                )

            self._vcam_running = True
            threading.Thread(target=self._vcam_loop, daemon=True).start()
            self._put("log", f"📹 VCam Active: {self._vcam.device}")

        except ImportError:
            self._put("log", "❌ Critical: Run 'pip install pyvirtualcam'")
        except Exception as e:
            self._put("log", "❌ Error: No Virtual Camera found!")
            print(f"VCAM ERROR: {e}")
            self._vcam_running = False



            # Start frame sender thread (Keep this logic the same as before)
            def send_frames():
                import time
                waiting_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(waiting_frame, "Waiting for Phone...", (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (255, 255, 255), 2)

                while self._vcam_running:
                    try:
                        if self._last_frame is not None:
                            self._vcam.send(self._last_frame)
                        else:
                            self._vcam.send(waiting_frame)
                        time.sleep(1 / 30)
                    except Exception as e:
                        break
                self._put("log", "📹 Frame sender stopped")

            self._vcam_thread = threading.Thread(target=send_frames, daemon=True)
            self._vcam_thread.start()

        except ImportError:
            self._put("log", "❌ pyvirtualcam not installed!")
        except Exception as e:
            self._put("log", f"❌ Camera Error: {e}")

            self._vcam_running = False
            self._vcam = None

    def stop_virtual_camera(self):
        """Stop virtual webcam."""
        if self._vcam:
            self._vcam_running = False
            try:
                self._vcam.close()
            except:
                pass
            self._vcam = None
            self._last_frame = None
            self._put("log", "📹 Virtual camera stopped")

    def _vcam_loop(self):
        # Update blank frame to match target resolution
        blank = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
        while self._vcam_running:
            try:
                frame = self._last_frame if self._last_frame is not None else blank

                # Safety check: If frame size mismatches (e.g. resolution changed mid-stream)
                if frame.shape[1] != self._vcam.width or frame.shape[0] != self._vcam.height:
                    # Resize to fit
                    frame = cv2.resize(frame, (self._vcam.width, self._vcam.height))

                self._vcam.send(frame)
                time.sleep(0.03)
            except:
                break
        self._vcam.close()

    def _handle_audio_frame(self, payload):
        """Handle incoming audio frame from phone microphone."""
        if not getattr(self, '_mic_active', False):
            return  # Ignore stray packets if mic is stopped

        try:
            import base64
            import pyaudio

            # 1. Decode Audio
            audio_bytes = base64.b64decode(payload)

            # 2. Initialize Audio System (One time)
            if not hasattr(self, '_mic_player'):
                self._mic_player = pyaudio.PyAudio()
                self._mic_stream = None
                self._mic_rate = 16000  # Default fallback

            # 3. Create/Re-create Stream if missing
            if not hasattr(self, '_mic_stream') or self._mic_stream is None:

                # Find Virtual Cable (Output) to play INTO
                target_device_index = None
                device_name = "Default Output"

                try:
                    info = self._mic_player.get_host_api_info_by_index(0)
                    numdevices = info.get('deviceCount')
                    for i in range(0, numdevices):
                        device = self._mic_player.get_device_info_by_host_api_device_index(0, i)
                        name = device.get('name')
                        # We look for "CABLE Input" or similar to route to mic
                        if device.get('maxOutputChannels') > 0:
                            if "CABLE Input" in name or "VB-Audio" in name:
                                target_device_index = i
                                device_name = name
                                break
                except:
                    pass

                if target_device_index:
                    self._put("log", f"🎤 Routing Phone Mic to: {device_name}")

                # Open Stream with the DYNAMIC rate
                self._mic_stream = self._mic_player.open(
                    format=pyaudio.paInt16,
                    channels=1,  # Mono (Standard for Mic)
                    rate=self._mic_rate,  # <--- USES VARIABLE FROM mic_start
                    output=True,
                    output_device_index=target_device_index
                )

            # 4. Write Data
            self._mic_stream.write(audio_bytes)

        except Exception as e:
            # self._put("log", f"❌ Mic Playback Error: {e}")
            pass

    def _handle_gamepad_state(self, payload, client_ws):
        """Handle gamepad input using a dictionary to support multiplayer."""
        try:
            import json
            state = json.loads(payload)

            # Ensure dictionary exists
            if not hasattr(self, 'client_gamepads'):
                self.client_gamepads = {}

            # Check if this specific client already has a controller
            if client_ws not in self.client_gamepads:
                try:
                    import vgamepad as vg
                    # Create a NEW gamepad instance for THIS client
                    # vgamepad automatically assigns the next available slot (Player 1, Player 2, etc.)
                    self.client_gamepads[client_ws] = vg.VX360Gamepad()
                    self._put("log", f"🎮 New Controller assigned to client")
                except Exception as e:
                    self._put("log", f"❌ Gamepad creation error: {e}")
                    return

            # Get the specific gamepad for this client
            current_gamepad = self.client_gamepads[client_ws]

            # --- Mapping Logic (Applied to 'current_gamepad') ---
            import vgamepad as vg

            button_map = {
                'a': vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
                'b': vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
                'x': vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
                'y': vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
                'lb': vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
                'rb': vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
                'view': vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
                'menu': vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
                'xbox': vg.XUSB_BUTTON.XUSB_GAMEPAD_GUIDE,
                'cross': vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
                'circle': vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
                'square': vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
                'triangle': vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
                'l1': vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
                'r1': vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
                'share': vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
                'options': vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
                'ps': vg.XUSB_BUTTON.XUSB_GAMEPAD_GUIDE,
            }

            buttons = state.get('buttons', {})

            # Reset buttons/Directions logic
            # Note: Ideally, you should only update changed buttons, but resetting/setting is safer for state sync

            for button, pressed in buttons.items():
                if button.startswith('dpad_'):
                    direction = button.replace('dpad_', '')
                    dpad_map = {
                        'up': vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
                        'down': vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
                        'left': vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
                        'right': vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT
                    }
                    if direction in dpad_map:
                        if pressed:
                            current_gamepad.press_button(dpad_map[direction])
                        else:
                            current_gamepad.release_button(dpad_map[direction])
                elif button in button_map:
                    if pressed:
                        current_gamepad.press_button(button_map[button])
                    else:
                        current_gamepad.release_button(button_map[button])

            # Analog Sticks
            left_x = float(state.get('leftStickX', 0.0))
            left_y = float(state.get('leftStickY', 0.0))
            right_x = float(state.get('rightStickX', 0.0))
            right_y = float(state.get('rightStickY', 0.0))

            current_gamepad.left_joystick_float(x_value_float=left_x, y_value_float=-left_y)
            current_gamepad.right_joystick_float(x_value_float=right_x, y_value_float=-right_y)

            # Triggers
            left_trigger = float(state.get('leftTrigger', 0.0))
            right_trigger = float(state.get('rightTrigger', 0.0))

            current_gamepad.left_trigger_float(value_float=left_trigger)
            current_gamepad.right_trigger_float(value_float=right_trigger)

            current_gamepad.update()

        except Exception as e:
            self._put("log", f"❌ Gamepad error: {e}")

    def _handle_display_request(self, payload):
        """Handle display streaming request from Android (supports forced restart)."""
        try:
            import json
            request = json.loads(payload)
            action = request.get('action')
            self._put("log", f"🖥️ Display request: {action}")

            # 1. READ PARAMS (Resolution + FPS)
            if action in ['start_display', 'change_resolution']:
                self._display_width = int(request.get('width', 1280))
                self._display_height = int(request.get('height', 720))
                self._display_fps = int(request.get('fps', 30))  # <--- READ FPS

                self._put("log", f"🖥️ Config: {self._display_width}x{self._display_height} @ {self._display_fps} FPS")

            if action == 'start_display':
                # If a capture is already active, stop it first (force restart)
                if hasattr(self, '_display_thread') and getattr(self,
                                                                '_display_thread') is not None and self._display_thread.is_alive():
                    self._put("log", "🔄 Restarting screen capture...")
                    self._stop_display_capture(wait_seconds=0.8)

                self._display_active = True

                import threading
                self._display_thread = threading.Thread(
                    target=self._capture_screen_loop,
                    daemon=True
                )
                self._display_thread.start()

            elif action == 'stop_display':
                self._stop_display_capture()
                self._display_active = False

        except Exception as e:
            self._put("log", f"❌ Display request error: {e}")

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
                    self._put("log", "✅ Previous screen capture stopped.")
        except Exception as e:
            self._put("log", f"❌ Error stopping display capture: {e}")

    def _capture_screen_loop(self):
        """Continuously capture and stream screen using DXCam (GPU Accelerated)."""
        try:
            import dxcam
            import cv2
            import base64
            import time
            import pyautogui
            import numpy as np

            self._put("log", "🖥️ Screen capture started (DXCam GPU Mode)")

            # Latency Tweak: JPEG Quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]

            # --- INITIALIZE DXCAM ---
            # output_color="BGR" ensures compatibility with OpenCV without conversion
            camera = dxcam.create(output_idx=0, output_color="BGR")

            # --- MOUSE HIDING LOGIC ---
            last_mouse_pos = (0, 0)
            last_move_time = time.time()
            HIDE_TIMEOUT = 3.0

            if camera is None:
                self._put("log", "❌ DXCam init failed! Fallback recommended.")
                return

            while self._display_active:
                start_time = time.time()

                # --- READ DYNAMIC SETTINGS INSIDE LOOP ---
                # This allows changing resolution/FPS without restarting the stream
                target_w = self._display_width
                target_h = self._display_height
                target_fps = self._display_fps

                # Calculate sleep time based on requested FPS
                frame_duration = 1.0 / max(1, target_fps)

                try:
                    # 1. Capture Frame (GPU)
                    frame = camera.grab()

                    if frame is None:
                        time.sleep(0.005)  # Tiny sleep to prevent CPU spin
                        continue

                    # 2. Draw Mouse
                    try:
                        mx, my = pyautogui.position()
                        if (mx, my) != last_mouse_pos:
                            last_mouse_pos = (mx, my)
                            last_move_time = time.time()

                        if time.time() - last_move_time < HIDE_TIMEOUT:
                            # Boundary check
                            if 0 <= mx < frame.shape[1] and 0 <= my < frame.shape[0]:
                                cv2.circle(frame, (mx, my), 8, (0, 0, 255), -1)
                                cv2.circle(frame, (mx, my), 9, (255, 255, 255), 1)
                    except:
                        pass

                    # 3. Resize (Dynamic)
                    if frame.shape[1] != target_w or frame.shape[0] != target_h:
                        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                    # 4. Encode
                    success, buffer = cv2.imencode('.jpg', frame, encode_param)

                    if success:
                        b64_data = base64.b64encode(buffer).decode('utf-8')
                        self._send_video_frame(b64_data)

                    # 5. Dynamic FPS Limiter
                    elapsed = time.time() - start_time
                    if elapsed < frame_duration:
                        time.sleep(frame_duration - elapsed)

                except Exception as e:
                    print(f"Cap error: {e}")
                    time.sleep(0.1)

            # Cleanup
            del camera
            self._put("log", "🖥️ Screen capture stopped")

        except Exception as e:
            self._put("log", f"❌ Capture Fatal Error: {e}")
            self._display_active = False

    def _send_video_frame(self, base64_data):
        """Send video frame to Android using the async broadcast queue (thread-safe)."""
        try:
            import json
            message = {
                "type": "video_frame",
                "payload": base64_data
            }
            # Put the JSON message into the async broadcast queue (thread-safe)
            payload = json.dumps(message)
            self._send_to_clients_threadsafe(payload)
        except Exception as e:
            self._put("log", f"❌ Send frame error: {e}")

    def start_audio_streaming(self, target_rate=48000):
        if self._streaming_audio:
            self._put("log", "🔊 Audio already streaming")
            return

        self._streaming_audio = True

        # --- Helper: Get Windows Default IDs ---
        def get_windows_audio_ids():
            ids = {}
            try:
                import winreg
                base_path = r"Software\Microsoft\Multimedia\Audio\DefaultRole"
                for role in ["0", "1", "2"]:
                    try:
                        key_path = f"{base_path}\\{role}"
                        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                            val, _ = winreg.QueryValueEx(key, "Device")
                            ids[role] = val
                    except:
                        ids[role] = "unknown"
            except:
                pass
            return ids

        # --- Helper: Clean Name ---
        def clean_name(n: str) -> str:
            if not n: return ""
            s = n.lower()
            for junk in ["[loopback]", "loopback", "(default)", "(default output)"]:
                s = s.replace(junk, "")
            return " ".join(s.replace("(", "").replace(")", "").replace("-", " ").replace("_", " ").split())

        # --- Helper: Find Loopback ---
        def find_best_loopback(pa_inst, default_name):
            try:
                loopbacks = [lb for lb in pa_inst.get_loopback_device_info_generator()]
            except:
                return None

            if not loopbacks: return None
            clean_def = clean_name(default_name)

            # Strategies
            for lb in loopbacks:
                if clean_name(lb["name"]) == clean_def: return lb
            for lb in loopbacks:
                if clean_def in clean_name(lb["name"]): return lb
            for lb in loopbacks:
                if clean_name(lb["name"]) in clean_def: return lb
            return loopbacks[0]

        # --- THE MANAGER THREAD ---
        def audio_manager():
            import time, base64, json, audioop
            import pyaudiowpatch as pyaudio

            # 1. THE CALLBACK (Called automatically by Audio System)
            # This runs in a separate C-thread. We must be fast here.
            def audio_callback(in_data, frame_count, time_info, status):
                try:
                    # Calculate Volume for Diagnostics
                    # (Optional: You can remove this print later)
                    # rms = audioop.rms(in_data, 2)
                    # if rms == 0: print("🔇 Silence")

                    if len(self.clients) > 0:
                        encoded = base64.b64encode(in_data).decode("utf-8")
                        msg = json.dumps({
                            "type": "audio_frame",
                            "rate": current_rate,  # Uses variable from outer scope
                            "payload": encoded
                        })
                        # Thread-safe send
                        for c in list(self.clients):
                            asyncio.run_coroutine_threadsafe(c.send(msg), self._loop)

                    return (None, pyaudio.paContinue)
                except Exception:
                    return (None, pyaudio.paAbort)

            # --- MAIN LOOP ---
            current_ids = get_windows_audio_ids()

            while self._streaming_audio:
                pa = None
                stream = None
                try:
                    # A. Initialize
                    pa = pyaudio.PyAudio()

                    try:
                        default_info = pa.get_default_output_device_info()
                        default_name = default_info.get("name", "Unknown")
                    except:
                        default_name = "Unknown"

                    self._put("log", f"🔊 Active: {default_name}")

                    # B. Find Device
                    loopback = find_best_loopback(pa, default_name)
                    if loopback is None:
                        self._put("log", "❌ Retrying device search...")
                        if pa: pa.terminate()
                        time.sleep(1.0)
                        continue

                    # C. Config
                    native_rate = int(loopback.get("defaultSampleRate", 44100))
                    current_rate = int(target_rate)  # Used by callback
                    channels = int(min(loopback.get("maxInputChannels", 2), 2))

                    # D. Start Stream (NON-BLOCKING)
                    try:
                        stream = pa.open(
                            format=pyaudio.paInt16,
                            channels=channels,
                            rate=current_rate,
                            input=True,
                            input_device_index=loopback["index"],
                            frames_per_buffer=2048,
                            stream_callback=audio_callback  # <--- KEY CHANGE
                        )
                    except OSError:
                        self._put("log", f"⚠️ Switching to {native_rate}Hz")
                        current_rate = native_rate
                        stream = pa.open(
                            format=pyaudio.paInt16,
                            channels=channels,
                            rate=current_rate,
                            input=True,
                            input_device_index=loopback["index"],
                            frames_per_buffer=2048,
                            stream_callback=audio_callback
                        )

                    self._put("log", f"✅ Stream Started: {current_rate}Hz")
                    stream.start_stream()

                    # E. Monitor Loop (Main thread is now free!)
                    # We just watch the registry here. No audio processing.
                    while self._streaming_audio and stream.is_active():
                        time.sleep(1.0)

                        # Check for Switch
                        new_ids = get_windows_audio_ids()
                        if new_ids != current_ids:
                            self._put("log", "🔄 Device Switch Detected!")
                            current_ids = new_ids
                            break  # Break loop to trigger clean restart

                    # F. Cleanup (Safe now because we aren't blocked)
                    stream.stop_stream()
                    stream.close()
                    pa.terminate()

                except Exception as e:
                    self._put("log", f"❌ Audio Error: {e}")
                    # Safety cleanup
                    try:
                        if stream: stream.close()
                        if pa: pa.terminate()
                    except:
                        pass
                    time.sleep(1.0)

            self._put("log", "🔇 Audio Stopped")

        self._audio_thread = threading.Thread(target=audio_manager, daemon=True)
        self._audio_thread.start()

    def stop_audio_streaming(self):
        """Stop audio streaming."""
        self._streaming_audio = False
        if self._audio_thread:
            self._audio_thread.join(timeout=2)
        self._put("log", "🔊 Audio streaming stopped")

    def _handle_mouse_move(self, payload):
        """
        Hyper-fast mouse handler.
        Assumes payload is a Dict (pre-parsed) or bytes.
        Skips all non-essential logic for speed.
        """
        try:
            # 1. Parsing Speedup
            # If payload is bytes/string, parse only if absolutely necessary.
            # Ideally, your handler passes the dict directly.
            if isinstance(payload, str):
                import json
                data = json.loads(payload)
                dx = int(data.get('deltaX', 0))
                dy = int(data.get('deltaY', 0))
            else:
                # Assuming payload is already a dict from the main loop
                dx = int(payload.get('deltaX', 0))
                dy = int(payload.get('deltaY', 0))

            # Filter tiny jitters (sensors are noisy)
            if dx == 0 and dy == 0: return

            # 2. Windows Low-Level Injection (No PyAutoGUI)
            # We skip bounds checking here because Windows handles it automatically
            # and GetCursorPos adds latency.
            if platform.system() == "Windows":
                # MOUSEEVENTF_MOVE = 0x0001
                # Using 'mouse_event' with relative movement is much smoother
                # than SetCursorPos for gaming/feeling
                ctypes.windll.user32.mouse_event(0x0001, dx, dy, 0, 0)
            else:
                # Linux/Mac fallback
                import pyautogui
                pyautogui.moveRel(dx, dy, _pause=False)

        except Exception:
            pass

    def _handle_mouse_click(self, payload):
        """
        Handle mouse clicks with delays (Required for Games).
        """
        try:
            import ctypes
            import json
            import time
            import platform

            event = json.loads(payload)
            button_id = event.get("button", 0)

            # "action" can be: "down", "up", or "click" (default)
            action = event.get("action", "click")

            # Windows Constants
            MOUSEEVENTF_LEFTDOWN = 0x0002
            MOUSEEVENTF_LEFTUP = 0x0004
            MOUSEEVENTF_RIGHTDOWN = 0x0008
            MOUSEEVENTF_RIGHTUP = 0x0010
            MOUSEEVENTF_MIDDLEDOWN = 0x0020
            MOUSEEVENTF_MIDDLEUP = 0x0040

            if platform.system() == "Windows":
                # --- LEFT BUTTON ---
                if button_id == 0:
                    if action == "down":
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                    elif action == "up":
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                    else:  # "click" (Full press)
                        # FIX: Add delay so games register the shot
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        if self.gaming_mode:
                            time.sleep(0.04)  # 40ms wait (Standard gaming mouse speed)
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

                # --- RIGHT BUTTON ---
                elif button_id == 1:
                    if action == "down":
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                    elif action == "up":
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                    else:
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                        if self.gaming_mode:
                            time.sleep(0.04)  # 40ms wait
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

                # --- MIDDLE BUTTON ---
                elif button_id == 2:
                    if action == "down":
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)
                    elif action == "up":
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)
                    else:
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)
                        if self.gaming_mode:
                            time.sleep(0.04)  # 40ms wait
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)

            else:
                # Fallback for Linux/Mac
                import pyautogui
                button_map = {0: "left", 1: "right", 2: "middle"}
                button = button_map.get(button_id, "left")

                if action == "down":
                    pyautogui.mouseDown(button=button)
                elif action == "up":
                    pyautogui.mouseUp(button=button)
                else:
                    pyautogui.click(button=button)

        except Exception as e:
            self._put("log", f"❌ Mouse click error: {e}")

    def _handle_mouse_scroll(self, payload):
        """Handle mouse scroll using Windows native API."""
        try:
            import ctypes
            event = json.loads(payload)
            scroll_delta = event.get("scrollDelta", 0)

            if platform.system() == "Windows":
                # MOUSEEVENTF_WHEEL = 0x0800
                # Normalize scroll
                wheel_delta = int(scroll_delta * 12)  # WHEEL_DELTA = 120
                if wheel_delta != 0:
                    ctypes.windll.user32.mouse_event(0x0800, 0, 0, wheel_delta, 0)
            else:
                # Fallback
                import pyautogui
                clicks = int(scroll_delta / 10)
                if clicks != 0:
                    pyautogui.scroll(clicks)

        except Exception as e:
            self._put("log", f"❌ Scroll error: {e}")

    def _handle_key_press(self, payload):
        """Handle keyboard input with Scan Codes (Required for Games)."""
        try:
            import platform
            import time
            import ctypes

            event = json.loads(payload)
            key = event.get("key", "")
            modifiers = event.get("modifiers", [])

            key_lower = key.lower()

            # --- BRIGHTNESS HANDLER ---
            if key_lower in ["brightnessup", "brightnessdown"]:
                try:
                    import screen_brightness_control as sbc
                    current_list = sbc.get_brightness()
                    if current_list:
                        current = current_list[0]
                        new_val = min(100, current + 10) if key_lower == "brightnessup" else max(0, current - 10)
                        threading.Thread(target=lambda: sbc.set_brightness(new_val)).start()
                except:
                    pass
                return

            # --- AUTO-SHIFT FIX ---
            if len(key) == 1 and key.isupper() and key.isalpha():
                if "shift" not in [m.lower() for m in modifiers]:
                    modifiers.append("shift")

            if platform.system() == "Windows":
                # Virtual-Key Codes
                VK_MAP = {
                    'a': 0x41, 'b': 0x42, 'c': 0x43, 'd': 0x44, 'e': 0x45,
                    'f': 0x46, 'g': 0x47, 'h': 0x48, 'i': 0x49, 'j': 0x4A,
                    'k': 0x4B, 'l': 0x4C, 'm': 0x4D, 'n': 0x4E, 'o': 0x4F,
                    'p': 0x50, 'q': 0x51, 'r': 0x52, 's': 0x53, 't': 0x54,
                    'u': 0x55, 'v': 0x56, 'w': 0x57, 'x': 0x58, 'y': 0x59, 'z': 0x5A,
                    '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34,
                    '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39,
                    'backspace': 0x08, 'tab': 0x09, 'enter': 0x0D, 'esc': 0x1B,
                    'space': 0x20, 'caps': 0x14, 'caps lock': 0x14,
                    '←': 0x25, '↑': 0x26, '→': 0x27, '↓': 0x28,
                    'shift': 0x10, 'ctrl': 0x11, 'alt': 0x12, 'meta': 0x5B, 'win': 0x5B
                }

                vk_code = VK_MAP.get(key_lower, 0)

                if vk_code:
                    # FIX 1: Get Hardware Scan Code (Essential for Games)
                    scan_code = ctypes.windll.user32.MapVirtualKeyW(vk_code, 0)

                    # Press Modifiers (Ctrl, Alt, Shift)
                    for mod in modifiers:
                        mod_vk = VK_MAP.get(mod.lower(), 0)
                        if mod_vk:
                            mod_scan = ctypes.windll.user32.MapVirtualKeyW(mod_vk, 0)
                            ctypes.windll.user32.keybd_event(mod_vk, mod_scan, 0, 0)

                    # Press Main Key
                    ctypes.windll.user32.keybd_event(vk_code, scan_code, 0, 0)  # Down

                    # FIX 2: Hold for 30ms (Games need time to detect input)
                    if self.gaming_mode:
                        time.sleep(0.03)

                    ctypes.windll.user32.keybd_event(vk_code, scan_code, 2, 0)  # Up

                    # Release Modifiers
                    for mod in reversed(modifiers):
                        mod_vk = VK_MAP.get(mod.lower(), 0)
                        if mod_vk:
                            mod_scan = ctypes.windll.user32.MapVirtualKeyW(mod_vk, 0)
                            ctypes.windll.user32.keybd_event(mod_vk, mod_scan, 2, 0)
                else:
                    # Fallback for keys not in map (Volume, Media, etc.)
                    import pyautogui
                    if key_lower == 'caps': key_lower = 'capslock'
                    if modifiers:
                        pyautogui.hotkey(*[m.lower() for m in modifiers] + [key_lower])
                    else:
                        pyautogui.press(key_lower)

            else:
                # Linux/Mac fallback (PyAutoGUI usually works fine on Linux games)
                import pyautogui
                key_map = {
                    "Backspace": "backspace", "Enter": "enter", "Space": "space",
                    "Tab": "tab", "Esc": "esc", "Caps": "capslock",
                    "←": "left", "→": "right", "↑": "up", "↓": "down"
                }
                mapped_key = key_map.get(key, key.lower())

                if modifiers:
                    pyautogui.hotkey(*[m.lower() for m in modifiers] + [mapped_key])
                else:
                    pyautogui.press(mapped_key)

        except Exception as e:
            self._put("log", f"❌ Key error: {e}")

    def _handle_gamepad(self, payload):
        """Handle gamepad input (map to keyboard for now)."""
        try:
            state = json.loads(payload)
            buttons = state.get("buttons", {})
            left_stick_x = state.get("leftStickX", 0)
            left_stick_y = state.get("leftStickY", 0)

            # Map left stick to WASD
            if abs(left_stick_x) > 0.3 or abs(left_stick_y) > 0.3:
                if left_stick_y < -0.3:  # Up
                    pyautogui.press('w')
                elif left_stick_y > 0.3:  # Down
                    pyautogui.press('s')
                if left_stick_x < -0.3:  # Left
                    pyautogui.press('a')
                elif left_stick_x > 0.3:  # Right
                    pyautogui.press('d')

            # Map buttons
            if buttons.get("cross") or buttons.get("a"):
                pyautogui.press('space')
            if buttons.get("circle") or buttons.get("b"):
                pyautogui.press('escape')

            self._put("log", f"🎮 Gamepad input processed")
        except Exception as e:
            self._put("log", f"❌ Gamepad error: {e}")

    # ============================================
    # ⬇️⬇️⬇️ CODE FROM PREVIOUS FIX ⬇️⬇️⬇️
    # ============================================

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

        try:
            self._broadcast_task = asyncio.create_task(self._broadcast_worker())
            self._put("log", "✅ Broadcast worker started")
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
            self._put("log", "✅ Server stopped")

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

# ============================================
# GUI CODE
# ============================================


class UDPMouseServer(threading.Thread):
    def __init__(self, port=8081):
        super().__init__()
        self.port = port
        self.running = True
        self.sock = None

    def run(self):
        import ctypes
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.setblocking(False)
            self.sock.bind(("0.0.0.0", self.port))
            print(f"🚀 UDP Mouse Server listening on port {self.port}")
        except Exception as e:
            print(f"❌ UDP Bind Failed on {self.port}: {e}")
            self.running = False
            return

        while self.running:
            try:
                data, _ = self.sock.recvfrom(1024)
                decoded = data.decode('utf-8')
                parts = decoded.split(',')
                if len(parts) == 2:
                    dx = int(parts[0])
                    dy = int(parts[1])
                    # Windows fast movement
                    ctypes.windll.user32.mouse_event(0x0001, dx, dy, 0, 0)
            except BlockingIOError:
                time.sleep(0.001)
            except Exception:
                pass

        if self.sock:
            self.sock.close()

    def stop(self):
        self.running = False


class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, parent, prefs, callback_save):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("400x550")
        self.prefs = prefs
        self.callback_save = callback_save

        # Modal behavior
        self.transient(parent)
        self.grab_set()

        # Layout container
        self.grid_columnconfigure(0, weight=1)

        # Title
        ctk.CTkLabel(self, text="Configuration", font=("Segoe UI", 20, "bold")).pack(pady=(20, 10))

        # --- GENERAL SETTINGS ---
        frame_gen = ctk.CTkFrame(self)
        frame_gen.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(frame_gen, text="General", font=("Segoe UI", 14, "bold"), text_color="#00e676").pack(anchor="w",
                                                                                                          padx=15,
                                                                                                          pady=(10, 5))

        self.var_autostart_pc = ctk.BooleanVar(value=prefs.get("autostart_pc", False))
        ctk.CTkSwitch(frame_gen, text="Auto-start with Windows", variable=self.var_autostart_pc).pack(anchor="w",
                                                                                                      padx=15, pady=5)

        self.var_autostart_server = ctk.BooleanVar(value=prefs.get("autostart_server", False))
        ctk.CTkSwitch(frame_gen, text="Auto-start Server on launch", variable=self.var_autostart_server).pack(
            anchor="w", padx=15, pady=5)

        self.var_admin = ctk.BooleanVar(value=prefs.get("run_as_admin", False))
        ctk.CTkSwitch(frame_gen, text="Request Admin Rights", variable=self.var_admin).pack(anchor="w", padx=15,
                                                                                            pady=(5, 15))

        # --- GAMING SETTINGS ---
        frame_input = ctk.CTkFrame(self)
        frame_input.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(frame_input, text="Input & Gaming", font=("Segoe UI", 14, "bold"), text_color="#00e676").pack(
            anchor="w", padx=15, pady=(10, 5))

        self.var_gaming_mode = ctk.BooleanVar(value=prefs.get("gaming_mode", True))
        ctk.CTkSwitch(frame_input, text="Gaming Mode (Low Latency)", variable=self.var_gaming_mode).pack(anchor="w",
                                                                                                         padx=15,
                                                                                                         pady=(5, 0))

        ctk.CTkLabel(frame_input,
                     text="Adds micro-delays so games detect input.\nDisable for faster desktop typing.",
                     font=("Segoe UI", 11), text_color="gray").pack(anchor="w", padx=55, pady=(0, 15))

        # --- PORT ---
        frame_net = ctk.CTkFrame(self)
        frame_net.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(frame_net, text="Server Port:", font=("Segoe UI", 12, "bold")).pack(side="left", padx=15)
        self.ent_port = ctk.CTkEntry(frame_net, width=80, placeholder_text="8080")
        self.ent_port.insert(0, str(prefs.get("port", 8080)))
        self.ent_port.pack(side="right", padx=15, pady=10)

        # --- SAVE BUTTON ---
        ctk.CTkButton(self, text="Save & Close", fg_color="#00e676", text_color="black", hover_color="#00c853",
                      height=40, font=("Segoe UI", 14, "bold"), command=self.save_and_close).pack(fill="x", padx=20,
                                                                                                  pady=20,
                                                                                                  side="bottom")

    def save_and_close(self):
        try:
            port = int(self.ent_port.get())
            if not (1024 <= port <= 65535): raise ValueError
        except:
            messagebox.showerror("Invalid Port", "Port must be a number between 1024 and 65535")
            return

        new_prefs = {
            "autostart_pc": self.var_autostart_pc.get(),
            "autostart_server": self.var_autostart_server.get(),
            "run_as_admin": self.var_admin.get(),
            "gaming_mode": self.var_gaming_mode.get(),
            "port": port
        }
        self.callback_save(new_prefs)
        self.destroy()




class ServerGUI:
    def __init__(self):
        # 1. Main Window Setup
        self.root = ctk.CTk()
        self.root.title(f"Use As Server v{APP_VERSION}")
        self.root.geometry("900x700")

        # 2. State & Prefs
        self.tray_icon = None
        self.prefs = {
            "autostart_pc": False,
            "autostart_server": False,
            "run_as_admin": False,
            "gaming_mode": True,
            "port": 8080
        }
        self.load_preferences()

        try:
            if Path("icon.ico").exists(): self.root.iconbitmap("icon.ico")
        except:
            pass

        self.update_queue = queue.Queue()
        self.server = None
        self.is_running = False

        # 3. Build UI
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 4. Auto-Start Logic
        if self.prefs["autostart_server"]:
            self.root.after(1000, self.toggle_server)

        self.root.after(100, self.process_queue)
        self.root.after(50, self.update_preview)

    # --- MISSING FUNCTION ADDED HERE ---
    def open_settings(self):
        SettingsDialog(self.root, self.prefs, self.save_preferences)

    # --- HELPERS ---
    def load_preferences(self):
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, 'r') as f: self.prefs.update(json.load(f))
        except:
            pass

    def save_preferences(self, new_prefs=None):
        if new_prefs: self.prefs = new_prefs
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(self.prefs, f, indent=4)
        except:
            pass
        if sys.platform == "win32":
            try:
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                                     winreg.KEY_SET_VALUE)
                app_path = sys.executable if getattr(sys, 'frozen', False) else f'"{sys.executable}" "{__file__}"'
                if self.prefs["autostart_pc"]:
                    winreg.SetValueEx(key, "UseAsServer", 0, winreg.REG_SZ, app_path)
                else:
                    try:
                        winreg.DeleteValue(key, "UseAsServer")
                    except:
                        pass
                winreg.CloseKey(key)
            except:
                pass

    def create_default_icon(self):
        width = 64;
        height = 64;
        color1 = "#00e676";
        color2 = "#1e1e1e"
        image = Image.new('RGB', (width, height), color1)
        dc = ImageDraw.Draw(image)
        dc.rectangle((width // 4, height // 4, 3 * width // 4, 3 * height // 4), fill=color2)
        return image

    def minimize_to_tray(self):
        self.root.withdraw()
        image = Image.open("icon.ico") if Path("icon.ico").exists() else self.create_default_icon()
        menu = (item('Restore', self.restore_from_tray, default=True), item('Stop Server & Quit', self.quit_app))
        self.tray_icon = pystray.Icon("UseAsServer", image, "Use As Server", menu)
        self.tray_icon.run_detached()

    def restore_from_tray(self, icon=None, item=None):
        if self.tray_icon: self.tray_icon.stop(); self.tray_icon = None
        self.root.after(0, self.root.deiconify)

    def quit_app(self, icon=None, item=None):
        if self.tray_icon: self.tray_icon.stop()
        self.root.after(0, self.on_closing)

    # --- UI SETUP ---
    def setup_ui(self):
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # 1. SIDEBAR
        self.sidebar = ctk.CTkFrame(self.root, width=140, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="Use As\nServer", font=("Segoe UI", 24, "bold"), text_color="#00e676").pack(
            pady=30)

        self.btn_nav_dash = ctk.CTkButton(self.sidebar, text="Dashboard", fg_color="transparent", border_width=2,
                                          text_color=("gray10", "#DCE4EE"),
                                          command=lambda: self.tabview.set("Dashboard"))
        self.btn_nav_dash.pack(pady=10, padx=20, fill="x")

        self.btn_nav_cam = ctk.CTkButton(self.sidebar, text="Camera Studio", fg_color="transparent", border_width=2,
                                         text_color=("gray10", "#DCE4EE"),
                                         command=lambda: self.tabview.set("Camera"))
        self.btn_nav_cam.pack(pady=10, padx=20, fill="x")

        self.btn_nav_share = ctk.CTkButton(self.sidebar, text="Sharing", fg_color="transparent", border_width=2,
                                           text_color=("gray10", "#DCE4EE"),
                                           command=lambda: self.tabview.set("Sharing"))
        self.btn_nav_share.pack(pady=10, padx=20, fill="x")

        ctk.CTkLabel(self.sidebar, text=f"v{APP_VERSION}", text_color="gray").pack(side="bottom", pady=20)

        # 2. MAIN TABS
        self.tabview = ctk.CTkTabview(self.root, fg_color="transparent")
        self.tabview.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")

        self.tabview.add("Dashboard")
        self.tabview.add("Camera")
        self.tabview.add("Sharing")

        self.build_dashboard(self.tabview.tab("Dashboard"))
        self.build_camera(self.tabview.tab("Camera"))
        self.build_sharing(self.tabview.tab("Sharing"))

    def build_dashboard(self, parent):
        # Server Status Card
        card = ctk.CTkFrame(parent)
        card.pack(fill="x", pady=10)

        self.lbl_status = ctk.CTkLabel(card, text="🔴 Offline", font=("Segoe UI", 22, "bold"), text_color="#cf6679")
        self.lbl_status.pack(pady=(20, 5))

        self.ent_ip = ctk.CTkEntry(card, justify="center", placeholder_text="Waiting for start...", width=300,
                                   font=("Consolas", 14), state="readonly")
        self.ent_ip.pack(pady=5)

        self.btn_start = ctk.CTkButton(card, text="START SERVER", font=("Segoe UI", 14, "bold"), height=45,
                                       fg_color="#00e676", text_color="black", hover_color="#00c853",
                                       command=self.toggle_server)
        self.btn_start.pack(pady=20)

        # Quick Actions
        card_act = ctk.CTkFrame(parent, fg_color="transparent")
        card_act.pack(fill="x", pady=5)

        self.btn_tray = ctk.CTkButton(card_act, text="Minimize to Tray", command=self.minimize_to_tray,
                                      fg_color="#2d2d2d")
        self.btn_tray.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.btn_settings = ctk.CTkButton(card_act, text="Settings", command=self.open_settings, fg_color="#2d2d2d")
        self.btn_settings.pack(side="right", fill="x", expand=True, padx=(5, 0))

        # Features Card
        feat_frame = ctk.CTkFrame(parent)
        feat_frame.pack(fill="x", pady=15)
        ctk.CTkLabel(feat_frame, text="Active Features", font=("Segoe UI", 12, "bold"), text_color="gray").pack(pady=5)

        row = ctk.CTkFrame(feat_frame, fg_color="transparent")
        row.pack(pady=10)

        self.btn_obs = ctk.CTkButton(row, text="OBS Camera", state="disabled",
                                     command=lambda: self.toggle_camera("OBS Virtual Camera"))
        self.btn_obs.pack(side="left", padx=5)

        self.btn_unity = ctk.CTkButton(row, text="Unity Camera", state="disabled",
                                       command=lambda: self.toggle_camera("Unity Video Capture"))
        self.btn_unity.pack(side="left", padx=5)

        self.btn_audio = ctk.CTkButton(row, text="Audio Stream", state="disabled", command=self.toggle_audio)
        self.btn_audio.pack(side="left", padx=5)

        # Logs
        ctk.CTkLabel(parent, text="Activity Log", anchor="w").pack(fill="x")
        self.log_box = ctk.CTkTextbox(parent, height=120, font=("Consolas", 12))
        self.log_box.pack(fill="x", pady=5)

    def build_camera(self, parent):
        # Split: Left Controls | Right Preview
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(parent, width=220)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        right = ctk.CTkFrame(parent, fg_color="black")
        right.grid(row=0, column=1, sticky="nsew")

        # --- Controls ---
        ctk.CTkLabel(left, text="Background", font=("Segoe UI", 14, "bold")).pack(pady=(10, 5))
        self.var_bg = ctk.StringVar(value="none")
        ctk.CTkRadioButton(left, text="None", variable=self.var_bg, value="none",
                           command=self.update_cam_settings).pack(anchor="w", padx=20, pady=2)
        ctk.CTkRadioButton(left, text="Blur", variable=self.var_bg, value="blur",
                           command=self.update_cam_settings).pack(anchor="w", padx=20, pady=2)
        ctk.CTkRadioButton(left, text="Image", variable=self.var_bg, value="image",
                           command=self.update_cam_settings).pack(anchor="w", padx=20, pady=2)
        ctk.CTkButton(left, text="Select Image...", command=self.select_bg_image, height=24, fg_color="#333").pack(
            pady=5)

        ctk.CTkLabel(left, text="Adjustments", font=("Segoe UI", 14, "bold")).pack(pady=(15, 5))
        self.var_mirror = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(left, text="Mirror Video", variable=self.var_mirror, command=self.update_cam_settings).pack(
            anchor="w", padx=20, pady=5)
        self.var_flip = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(left, text="Flip Vertical", variable=self.var_flip, command=self.update_cam_settings).pack(
            anchor="w", padx=20, pady=5)

        ctk.CTkLabel(left, text="Brightness", font=("Segoe UI", 12)).pack(pady=(5, 0))
        self.scale_bright = ctk.CTkSlider(left, from_=-100, to=100, command=lambda x: self.update_cam_settings())
        self.scale_bright.set(0)
        self.scale_bright.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(left, text="Resolution", font=("Segoe UI", 12)).pack(pady=(10, 5))
        self.combo_res = ctk.CTkOptionMenu(left, values=["1280x720 (16:9)", "1920x1080 (16:9)", "800x600 (4:3)",
                                                         "720x720 (1:1)"],
                                           command=self.change_aspect_ratio)
        self.combo_res.pack(padx=20)

        # --- 👇 RESTORED OUTPUT BUTTONS HERE 👇 ---
        ctk.CTkLabel(left, text="Output Control", font=("Segoe UI", 14, "bold")).pack(pady=(20, 5))

        self.btn_cs_obs = ctk.CTkButton(left, text="Start OBS Camera", fg_color=["#3B8ED0", "#1F6AA5"],
                                        command=lambda: self.toggle_camera("OBS Virtual Camera"))
        self.btn_cs_obs.pack(padx=20, pady=5, fill="x")

        self.btn_cs_unity = ctk.CTkButton(left, text="Start Unity Camera", fg_color=["#3B8ED0", "#1F6AA5"],
                                          command=lambda: self.toggle_camera("Unity Video Capture"))
        self.btn_cs_unity.pack(padx=20, pady=5, fill="x")
        # ------------------------------------------

        # --- Preview ---
        self.lbl_preview = ctk.CTkLabel(right, text="Waiting for connection...", text_color="gray")
        self.lbl_preview.place(relx=0.5, rely=0.5, anchor="center")

    def build_sharing(self, parent):
        # Clipboard
        card_clip = ctk.CTkFrame(parent)
        card_clip.pack(fill="x", pady=10)
        ctk.CTkLabel(card_clip, text="Clipboard Sync", font=("Segoe UI", 14, "bold"), text_color="#00e676").pack(
            anchor="w", padx=15, pady=10)

        self.txt_clip = ctk.CTkTextbox(card_clip, height=80)
        self.txt_clip.pack(fill="x", padx=15, pady=5)

        row = ctk.CTkFrame(card_clip, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=10)
        ctk.CTkButton(row, text="Copy to PC", command=self.copy_to_pc).pack(side="left", expand=True, padx=5)
        ctk.CTkButton(row, text="Send to Phone", command=self.send_text_to_phone, fg_color="#2d2d2d").pack(side="right",
                                                                                                           expand=True,
                                                                                                           padx=5)

        # Files
        card_file = ctk.CTkFrame(parent)
        card_file.pack(fill="both", expand=True, pady=10)
        ctk.CTkLabel(card_file, text="File Transfer", font=("Segoe UI", 14, "bold"), text_color="#00e676").pack(
            anchor="w", padx=15, pady=10)

        self.list_files = ctk.CTkTextbox(card_file)
        self.list_files.pack(fill="both", expand=True, padx=15, pady=5)

        row2 = ctk.CTkFrame(card_file, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=10)
        ctk.CTkButton(row2, text="Open Folder", command=self.open_folder).pack(side="left", expand=True, padx=5)
        ctk.CTkButton(row2, text="Send File...", command=self.send_file_pick, fg_color="#2d2d2d").pack(side="right",
                                                                                                       expand=True,
                                                                                                       padx=5)

    # --- LOGIC HANDLERS ---
    def toggle_server(self):
        if not self.is_running:
            self.btn_start.configure(text="STOP SERVER", fg_color="#cf6679", hover_color="#b00020")
            self.lbl_status.configure(text="Starting...", text_color="orange")
            threading.Thread(target=self.start_sequence, daemon=True).start()
        else:
            self.stop_server()

    def stop_server(self):
        if self.server: self.server.stop()
        self.is_running = False
        self.btn_start.configure(text="START SERVER", fg_color="#00e676", hover_color="#00c853")
        self.lbl_status.configure(text="🔴 Offline", text_color="#cf6679")
        self.ent_ip.configure(state="normal");
        self.ent_ip.delete(0, "end");
        self.ent_ip.insert(0, "Not Running");
        self.ent_ip.configure(state="readonly")

        for btn in [self.btn_obs, self.btn_unity, self.btn_audio]:
            btn.configure(state="disabled", fg_color=["#3B8ED0", "#1F6AA5"])

    def start_sequence(self):
        try:
            if self.server: self.server.stop()
            port = self.prefs.get("port", 8080);
            gaming_mode = self.prefs.get("gaming_mode", True)
            self.server = UnifiedRemoteServer(port=port, gaming_mode=gaming_mode, update_queue=self.update_queue)
            self.server.start()
            for _ in range(20):
                if self.server._loop and self.server._loop.is_running():
                    ip = self.server.get_local_ip();
                    self.update_queue.put(("server_ready", f"{ip}:{port}"));
                    return
                time.sleep(0.1)
            raise Exception("Timeout")
        except Exception as e:
            self.update_queue.put(("error", str(e))); self.stop_server()

    def toggle_camera(self, device_name):
        if not self.server: return
        if self.server._vcam_running:
            self.server.stop_virtual_camera()
            self._update_cam_buttons(None)
        else:
            self.server.start_virtual_camera(device_name)
            self.root.after(100, lambda: self._update_cam_buttons(device_name))

    def _update_cam_buttons(self, active_device):
        is_running = self.server and self.server._vcam_running
        if is_running:
            if active_device == "OBS Virtual Camera":
                self.btn_obs.configure(text="⏹ Stop OBS", fg_color="#cf6679", state="normal")
                self.btn_unity.configure(state="disabled")
            else:
                self.btn_unity.configure(text="⏹ Stop Unity", fg_color="#cf6679", state="normal")
                self.btn_obs.configure(state="disabled")
        else:
            self.btn_obs.configure(text="OBS Camera", fg_color=["#3B8ED0", "#1F6AA5"], state="normal")
            self.btn_unity.configure(text="Unity Camera", fg_color=["#3B8ED0", "#1F6AA5"], state="normal")

    def toggle_audio(self):
        if not self.server._streaming_audio:
            self.server.start_audio_streaming()
            self.btn_audio.configure(text="⏹ Stop Audio", fg_color="#cf6679")
        else:
            self.server.stop_audio_streaming()
            self.btn_audio.configure(text="Audio Stream", fg_color=["#3B8ED0", "#1F6AA5"])

    def send_text_to_phone(self):
        text = self.txt_clip.get("1.0", "end").strip()
        if text and self.server: self.server.send_to_android("clipboard_text", text)

    def copy_to_pc(self):
        pyperclip.copy(self.txt_clip.get("1.0", "end").strip())

    def send_file_pick(self):
        paths = filedialog.askopenfilenames()
        if paths and self.server: threading.Thread(target=self.process_multiple_files, args=(paths,),
                                                   daemon=True).start()

    def process_multiple_files(self, paths):
        for p in paths: self.server.send_file_to_phone_thread(p); time.sleep(0.5)

    def open_folder(self):
        os.startfile(SAVE_DIR)

    def update_cam_settings(self):
        if self.server:
            self.server.bg_mode = self.var_bg.get();
            self.server.is_mirrored = self.var_mirror.get()
            self.server.is_flipped = self.var_flip.get();
            self.server.brightness_boost = int(self.scale_bright.get())

    def select_bg_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path and self.server:
            self.server.bg_image_path = path;
            self.server.bg_image_cache = None;
            self.var_bg.set("image");
            self.update_cam_settings()

    def change_aspect_ratio(self, selection):
        if not self.server: return
        try:
            w, h = map(int, selection.split(" ")[0].split("x"))
            self.server.target_w = w;
            self.server.target_h = h
            if self.server._vcam_running:
                dev = self.server._vcam.device;
                self.server.stop_virtual_camera()
                self.root.after(500, lambda: self.server.start_virtual_camera(dev))
        except:
            pass

    def update_preview(self):
        if self.server and self.server.latest_preview_frame is not None:
            try:
                frame = self.server.latest_preview_frame
                img = Image.fromarray(frame)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(frame.shape[1], frame.shape[0]))
                self.lbl_preview.configure(image=ctk_img, text="")
            except:
                pass
        self.root.after(33, self.update_preview)

    def process_queue(self):
        try:
            while True:
                kind, data = self.update_queue.get_nowait()

                if kind == "log":
                    self.log(data)

                elif kind == "server_ready":
                    self.is_running = True
                    self.lbl_status.configure(text="🟢 Online", text_color="#00e676")
                    self.ent_ip.configure(state="normal");
                    self.ent_ip.delete(0, "end");
                    self.ent_ip.insert(0, f"ws://{data}");
                    self.ent_ip.configure(state="readonly")
                    self.btn_obs.configure(state="normal")
                    self.btn_unity.configure(state="normal")
                    self.btn_audio.configure(state="normal")

                elif kind == "error":
                    messagebox.showerror("Error", data)
                    self.stop_server()

                elif kind == "clipboard":
                    self.txt_clip.delete("1.0", "end")
                    self.txt_clip.insert("end", data)

                elif kind == "file_received":
                    self.list_files.insert("end", f"📄 {os.path.basename(data)}\n")

                # 👇👇👇 NEW: HANDLE AUDIO STATUS UPDATES 👇👇👇
                elif kind == "audio_status":
                    is_on = data  # True/False
                    if is_on:
                        self.btn_audio.configure(text="⏹ Stop Audio", fg_color="#cf6679")
                    else:
                        self.btn_audio.configure(text="Start Audio Stream", fg_color=["#3B8ED0", "#1F6AA5"])

                # 👇👇👇 NEW: HANDLE CAMERA (OBS/UNITY) UPDATES 👇👇👇
                elif kind == "camera_status":
                    # data will be the device name (e.g. "OBS Virtual Camera") or None if stopped
                    self._update_cam_buttons(data)

        except queue.Empty:
            pass
        except Exception as e:
            print(f"Queue Error: {e}")

        self.root.after(200, self.process_queue)

    def log(self, msg):
        self.log_box.insert("end", f"> {msg}\n");
        self.log_box.see("end")

    def on_closing(self):
        if self.is_running: self.stop_server()
        self.root.destroy()


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    # Admin check (Keep existing logic)
    must_be_admin = False
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, 'r') as f: must_be_admin = json.load(f).get("run_as_admin", False)
    except:
        pass

    is_admin = False
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
    except:
        pass

    if must_be_admin and not is_admin:
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)
        sys.exit()

    if sys.platform == "win32": asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    app = ServerGUI()
    app.root.mainloop()

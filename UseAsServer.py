import base64
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
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
APP_VERSION = "1.0.0"
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
    def __init__(self, host="0.0.0.0", port=8080, update_queue: queue.Queue = None):
        import threading
        self.sending_lock = threading.Lock()
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

        # --- FIX: DO NOT CREATE ASYNCIO OBJECTS HERE ---
        # asyncio.Queue() requires an active event loop.
        # Since __init__ runs in a standard thread, this would crash.
        # We initialize it as None here, and create the real Queue inside _run_loop later.
        self._broadcast_queue = None
        # -----------------------------------------------

        # ... (Rest of variables: vcam, audio, display, codec, etc.) ...
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
        self._display_width = 1920
        self._display_height = 1080

        # ... (Rest of your initialization variables remain the same) ...
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
                        # 1. Handle Binary Data (Video frames often come as bytes)
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

                        # --- AUDIO & GAMEPAD ---
                        elif msg_type == "audio_frame":
                            self._handle_audio_frame(payload)

                        elif msg_type == "audio_control":
                            action = payload
                            if action == "start":
                                self._put("log", "📲 Auto-starting Audio Stream...")
                                # Run in thread to avoid blocking the websocket loop
                                threading.Thread(target=self.start_audio_streaming, daemon=True).start()

                                # OPTIONAL: Update GUI Button automatically
                                # This requires a callback or queue event handled by process_queue
                                self._put("audio_status", True)

                            elif action == "stop":
                                self._put("log", "📲 Client requested Audio Stop")
                                self.stop_audio_streaming()
                                self._put("audio_status", False)
                        elif msg_type == "gamepad_state":
                            # Pass websocket so we know WHICH controller to update
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
                        pass  # Ignore tiny decode errors

            except websockets.ConnectionClosedOK:
                pass
            except Exception as e:
                self._put("log", f"⚠️ Handler error: {e}")

            # --- CLEANUP ON DISCONNECT ---
            finally:
                if websocket in self.clients:
                    self.clients.remove(websocket)

                # 🔌 UNPLUG GAMEPAD (Fixes "4 Controllers" issue)
                if hasattr(self, 'client_gamepads') and websocket in self.client_gamepads:
                    try:
                        del self.client_gamepads[websocket]
                        self._put("log", f"🎮 Gamepad unplugged for {remote_addr}")
                    except Exception as e:
                        print(f"Error removing gamepad: {e}")

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
        """Handle incoming audio frame from microphone."""
        try:
            import base64
            import pyaudio

            # Decode base64 audio
            audio_bytes = base64.b64decode(payload)

            # Initialize PyAudio if needed
            if not hasattr(self, '_audio_player'):
                self._audio_player = pyaudio.PyAudio()

                # --- FIND VIRTUAL CABLE DEVICE ---
                target_device_index = None
                device_name = "Default Speakers"

                try:
                    # List all devices to find the Virtual Cable
                    info = self._audio_player.get_host_api_info_by_index(0)
                    numdevices = info.get('deviceCount')

                    for i in range(0, numdevices):
                        device = self._audio_player.get_device_info_by_host_api_device_index(0, i)
                        name = device.get('name')
                        max_input = device.get('maxInputChannels')
                        max_output = device.get('maxOutputChannels')

                        # Look for VB-Cable (Windows) or PulseAudio Null Sink (Linux)
                        # We need an OUTPUT device (Python plays TO it)
                        if max_output > 0:
                            # "CABLE Input" is what we play TO. "CABLE Output" is the mic.
                            if "CABLE Input" in name or "VB-Audio" in name or "Virtual Audio Driver" in name:
                                target_device_index = i
                                device_name = name
                                break
                except Exception:
                    pass

                # Log where we are sending audio
                if target_device_index is not None:
                    self._put("log", f"🎤 Found Virtual Mic Driver: {device_name}")
                    self._put("log", "✅ Audio is now routing to your Input Devices!")
                else:
                    self._put("log", "⚠️ Virtual Cable NOT Found - Playing to Speakers")
                    self._put("log", "💡 Install VB-CABLE (Win) or setup PulseAudio (Linux) to use as Mic")

                # Open Stream
                self._audio_stream = self._audio_player.open(
                    format=pyaudio.paInt16,
                    channels=1,  # Mono (Matches Android config)
                    rate=16000,  # Standard Sample Rate
                    output=True,
                    output_device_index=target_device_index  # Use the virtual cable if found
                )
                self._put("log", "🔊 Audio playback initialized")

            # Write audio to the device
            self._audio_stream.write(audio_bytes)

        except Exception as e:
            # self._put("log", f"❌ Audio Error: {e}")
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

            if action == 'start_display':
                width = request.get('width', 1920)
                height = request.get('height', 1080)

                self._display_width = width
                self._display_height = height

                # If a capture is already active, stop it first (force restart)
                if hasattr(self, '_display_thread') and getattr(self,
                                                                '_display_thread') is not None and self._display_thread.is_alive():
                    self._put("log", "🔄 Restarting screen capture (stop existing first)...")
                    self._stop_display_capture(wait_seconds=0.8)

                self._display_active = True
                self._put("log", f"🖥️ Starting screen capture: {width}x{height}")

                # Start screen capture thread
                import threading
                self._display_thread = threading.Thread(
                    target=self._capture_screen_loop,
                    daemon=True
                )
                self._display_thread.start()

                self._put("log", "✅ Screen capture thread started")
                import psutil, os
                p = psutil.Process(os.getpid())
                p.nice(psutil.HIGH_PRIORITY_CLASS)

            elif action == 'stop_display':
                # Stop capture gracefully
                self._put("log", "🛑 Stop display requested")
                self._stop_display_capture()
                self._display_active = False

            elif action == 'change_resolution':
                width = request.get('width', 1920)
                height = request.get('height', 1080)
                self._display_width = width
                self._display_height = height
                self._put("log", f"🖥️ Resolution changed: {width}x{height}")

        except Exception as e:
            self._put("log", f"❌ Display request error: {e}")
            import traceback
            self._put("log", f"Stack trace: {traceback.format_exc()}")

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
        """Continuously capture and stream screen using OpenCV (Much Faster)."""
        try:
            import mss
            import numpy as np
            import cv2
            import base64
            import time
            import pyautogui

            self._put("log", "🖥️ Screen capture started (Fast Mode)")

            # Latency Tweak: Lower resolution = Lower Latency
            # 1280x720 is the sweet spot for WiFi streaming
            target_w, target_h = 1280, 720

            # Latency Tweak: JPEG Quality
            # 50-60 is good. Below 40 looks bad, above 70 adds lag.
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]

            # --- MOUSE HIDING LOGIC ---
            last_mouse_pos = (0, 0)
            last_move_time = time.time()
            HIDE_TIMEOUT = 3.0  # Hide pointer after 3 seconds of inactivity

            with mss.mss() as sct:
                # Select Monitor
                monitor_idx = 1
                if len(sct.monitors) > 2: monitor_idx = 2
                monitor = sct.monitors[monitor_idx]

                mon_left = monitor["left"]
                mon_top = monitor["top"]

                while self._display_active:
                    start_time = time.time()

                    try:
                        # 1. Capture (Raw BGRA)
                        sct_img = sct.grab(monitor)

                        # 2. Convert to Numpy Array (Fast)
                        # MSS returns BGRA, OpenCV uses BGR. Remove Alpha channel for speed.
                        frame = np.array(sct_img)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                        # 3. Draw Mouse (Only if moved recently)
                        try:
                            mx, my = pyautogui.position()
                            current_pos = (mx, my)

                            # Check if mouse moved
                            if current_pos != last_mouse_pos:
                                last_mouse_pos = current_pos
                                last_move_time = time.time()

                            # Only draw if the timeout hasn't passed
                            if time.time() - last_move_time < HIDE_TIMEOUT:
                                rel_x = mx - mon_left
                                rel_y = my - mon_top

                                # Scaling check
                                if 0 <= rel_x < sct_img.width and 0 <= rel_y < sct_img.height:
                                    # Draw red circle
                                    cv2.circle(frame, (rel_x, rel_y), 8, (0, 0, 255), -1)
                                    cv2.circle(frame, (rel_x, rel_y), 9, (255, 255, 255), 1)
                        except:
                            pass

                        # 4. Resize (Inter_Linear is fast enough)
                        if frame.shape[1] != target_w or frame.shape[0] != target_h:
                            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                        # 5. Encode to JPEG
                        success, buffer = cv2.imencode('.jpg', frame, encode_param)

                        if success:
                            # 6. Convert to Base64
                            b64_data = base64.b64encode(buffer).decode('utf-8')
                            self._send_video_frame(b64_data)

                        # 7. FPS Limiter (Cap at 30 to save CPU for Network)
                        elapsed = time.time() - start_time
                        if elapsed < 0.033:
                            time.sleep(0.033 - elapsed)

                    except Exception as e:
                        # Only log fatal errors, ignore glitches
                        print(f"Cap error: {e}")
                        time.sleep(0.1)

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

    def start_audio_streaming(self):

        if self._streaming_audio:
            self._put("log", "🔊 Audio already streaming")
            return

        self._streaming_audio = True

        def clean_name(n: str) -> str:
            # Normalize names for fuzzy matching
            if not n:
                return ""
            s = n.lower()
            for junk in ["[loopback]", "loopback", "(default)", "(default output)"]:
                s = s.replace(junk, "")
            s = s.replace("(", "").replace(")", "")
            s = s.replace("-", " ").replace("_", " ").strip()
            return " ".join(s.split())

        def find_best_loopback(pa, default_name):
            """Try multiple strategies to find a matching loopback device."""
            # Build lists
            loopbacks = []
            try:
                for lb in pa.get_loopback_device_info_generator():
                    loopbacks.append(lb)
            except Exception:
                loopbacks = []

            if not loopbacks:
                return None

            defscore = []
            clean_default = clean_name(default_name)

            # 1) exact match (cleaned)
            for lb in loopbacks:
                if clean_name(lb["name"]) == clean_default:
                    return lb

            # 2) case-insensitive substring (default in loopback)
            for lb in loopbacks:
                if clean_default in clean_name(lb["name"]) or clean_name(lb["name"]) in clean_default:
                    return lb

            # 3) partial token overlap (score)
            tokens_def = set(clean_default.split())
            best = None
            best_score = 0
            for lb in loopbacks:
                tokens_lb = set(clean_name(lb["name"]).split())
                score = len(tokens_def.intersection(tokens_lb))
                if score > best_score:
                    best_score = score
                    best = lb
            if best_score > 0:
                return best

            # 4) fallback to first loopback
            return loopbacks[0]

        def audio_worker():
            import time
            import base64
            import json
            import pyaudiowpatch as pyaudio  # safe import here

            pa = None
            stream = None

            # Keep the worker alive; if no suitable device, sleep and retry
            while self._streaming_audio:
                try:
                    pa = pyaudio.PyAudio()

                    # Try to get default output name (may raise)
                    try:
                        default_info = pa.get_default_output_device_info()
                        default_name = default_info.get("name", "")
                    except Exception:
                        default_name = ""

                    self._put("log", f"🔊 Windows default output: {default_name}")

                    # Find best loopback (fuzzy)
                    loopback = find_best_loopback(pa, default_name)

                    if loopback is None:
                        self._put("log", "❌ No loopback devices found — retrying in 1s")
                        try:
                            pa.terminate()
                        except:
                            pass
                        time.sleep(1.0)
                        continue

                    # Use loopback native settings
                    RATE = int(loopback.get("defaultSampleRate", 48000))
                    CHANNELS = int(min(loopback.get("maxInputChannels", 2), 2))
                    CHUNK = 2048

                    self._put("log", f"🎧 Capturing from: {loopback['name']}")
                    self._put("log", f"🎵 Format: {RATE} Hz, {CHANNELS} ch")

                    # Open stream (fail safe)
                    try:
                        stream = pa.open(
                            format=pyaudio.paInt16,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            input_device_index=loopback["index"],
                            frames_per_buffer=CHUNK
                        )
                    except Exception as e:
                        self._put("log", f"❌ Failed to open loopback ({loopback['name']}): {e}")
                        try:
                            stream and stream.close()
                        except:
                            pass
                        try:
                            pa.terminate()
                        except:
                            pass
                        time.sleep(1.0)
                        continue

                    self._audio_stream = stream
                    self._put("log", "🎧 Audio streaming started")

                    frames = 0
                    last_device_check = time.time()

                    # Keep streaming even if 0 clients — do not exit immediately to avoid thrash
                    while self._streaming_audio:
                        try:
                            # Periodically check if the loopback device still exists
                            if time.time() - last_device_check >= 1.0:
                                last_device_check = time.time()
                                # Re-scan loopback names to ensure this device still present
                                still_there = False
                                try:
                                    for lb in pa.get_loopback_device_info_generator():
                                        if lb["index"] == loopback["index"]:
                                            still_there = True
                                            break
                                except Exception:
                                    still_there = True  # be permissive if API flaky

                                if not still_there:
                                    # device disappeared (e.g., user switched output)
                                    self._put("log", f"🔄 Loopback disappeared: {loopback['name']}")
                                    break  # restart outer worker to find new device

                            # Read a chunk
                            try:
                                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                            except Exception as e:
                                self._put("log", f"⚠️ Read error: {e}")
                                break  # restart (or try re-open)

                            # If no clients, don't spam sending — sleep shortly instead
                            if len(self.clients) == 0:
                                # Still keep reading to keep stream warmed up, but don't send
                                time.sleep(0.02)  # tiny sleep to avoid CPU spin
                                continue

                            encoded_audio = base64.b64encode(audio_data).decode("utf-8")

                            message = json.dumps({
                                "type": "audio_frame",
                                "rate": RATE,
                                "channels": CHANNELS,
                                "payload": encoded_audio
                            })

                            # Broadcast (best-effort)
                            for c in list(self.clients):
                                try:
                                    asyncio.run_coroutine_threadsafe(c.send(message), self._loop)
                                except Exception:
                                    # ignore per-client send failures
                                    pass

                            frames += 1

                        except Exception as e:
                            self._put("log", f"⚠️ Stream loop error: {e}")
                            break

                    # cleanup after inner loop
                    try:
                        if stream:
                            stream.stop_stream()
                            stream.close()
                    except:
                        pass

                    try:
                        if pa:
                            pa.terminate()
                    except:
                        pass

                    self._put("log", "🔊 Audio streaming stopped")

                    # gentle pause before trying to re-open to avoid tight restart loops
                    time.sleep(0.5)
                    continue

                except Exception as e:
                    # fatal-ish, but keep worker alive and retry
                    self._put("log", f"❌ Fatal worker error: {e}")
                    try:
                        if stream:
                            stream.close()
                    except:
                        pass
                    try:
                        if pa:
                            pa.terminate()
                    except:
                        pass
                    time.sleep(1.0)
                    continue

            # worker exit
            self._put("log", "🔇 Audio worker exited")

        # launch thread
        self._audio_thread = threading.Thread(target=audio_worker, daemon=True)
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
        Handle mouse click, down, and up events using Windows native API.
        Supports Dragging.
        """
        try:
            import ctypes
            import json

            event = json.loads(payload)
            button_id = event.get("button", 0)

            # Check for explicit action (default to 'click' for backward compatibility)
            # The Android app now sends "down" or "up" for dragging
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
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

                # --- RIGHT BUTTON ---
                elif button_id == 1:
                    if action == "down":
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                    elif action == "up":
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                    else:
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

                # --- MIDDLE BUTTON ---
                elif button_id == 2:
                    if action == "down":
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)
                    elif action == "up":
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)
                    else:
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)

            else:
                # Fallback for Linux/Mac (PyAutoGUI)
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
        """Handle keyboard input using Windows native API (instant response)."""
        try:
            import platform
            event = json.loads(payload)
            key = event.get("key", "")
            modifiers = event.get("modifiers", [])

            key_lower = key.lower()

            # --- BRIGHTNESS HANDLER (ADD THIS BLOCK) ---
            if key_lower in ["brightnessup", "brightnessdown"]:
                try:
                    import screen_brightness_control as sbc

                    # Get current brightness (returns list, get first display)
                    current_list = sbc.get_brightness()
                    if current_list:
                        current = current_list[0]

                        if key_lower == "brightnessup":
                            new_val = min(100, current + 10)  # Increase by 10%
                        else:
                            new_val = max(0, current - 10)  # Decrease by 10%

                        sbc.set_brightness(new_val)
                        self._put("log", f"💡 Brightness set to {new_val}%")
                    return  # Exit function, we are done
                except ImportError:
                    self._put("log", "❌ Install 'screen-brightness-control' for brightness support")
                    return
                except Exception as e:
                    self._put("log", f"❌ Brightness Error: {e}")
                    return


            # --- FIX: Auto-Shift for Uppercase Letters ---
            # If Android sends "A", we treat it as "Shift + a"
            # This ensures Caps works regardless of the PC's actual Caps Lock state
            if len(key) == 1 and key.isupper() and key.isalpha():
                # Add shift to modifiers if not already there
                if "shift" not in [m.lower() for m in modifiers]:
                    modifiers.append("shift")
            # ---------------------------------------------

            if platform.system() == "Windows":
                import ctypes

                # Virtual-Key Codes
                VK_MAP = {
                    # Letters
                    'a': 0x41, 'b': 0x42, 'c': 0x43, 'd': 0x44, 'e': 0x45,
                    'f': 0x46, 'g': 0x47, 'h': 0x48, 'i': 0x49, 'j': 0x4A,
                    'k': 0x4B, 'l': 0x4C, 'm': 0x4D, 'n': 0x4E, 'o': 0x4F,
                    'p': 0x50, 'q': 0x51, 'r': 0x52, 's': 0x53, 't': 0x54,
                    'u': 0x55, 'v': 0x56, 'w': 0x57, 'x': 0x58, 'y': 0x59, 'z': 0x5A,
                    # Numbers
                    '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34,
                    '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39,
                    # Special keys
                    'backspace': 0x08, 'tab': 0x09, 'enter': 0x0D, 'esc': 0x1B,
                    'space': 0x20, 'caps': 0x14, 'caps lock': 0x14,
                    '←': 0x25, '↑': 0x26, '→': 0x27, '↓': 0x28,
                    # Modifiers
                    'shift': 0x10, 'ctrl': 0x11, 'alt': 0x12, 'meta': 0x5B,
                    'win': 0x5B, 'menu': 0x5D,'brightnessup': None,
                    # Windows doesn't always have a direct VK for this via keybd_event easily
                    'brightnessdown': None,
                    # but some laptops use special driver keys.
                }

                # Normalize key input for lookup
                key_lower = key.lower()
                vk_code = VK_MAP.get(key_lower, 0)

                if vk_code:
                    # 1. Press modifiers
                    for mod in modifiers:
                        mod_vk = VK_MAP.get(mod.lower(), 0)
                        if mod_vk:
                            ctypes.windll.user32.keybd_event(mod_vk, 0, 0, 0)

                    # 2. Press main key
                    ctypes.windll.user32.keybd_event(vk_code, 0, 0, 0)  # Down
                    ctypes.windll.user32.keybd_event(vk_code, 0, 2, 0)  # Up

                    # 3. Release modifiers
                    for mod in reversed(modifiers):
                        mod_vk = VK_MAP.get(mod.lower(), 0)
                        if mod_vk:
                            ctypes.windll.user32.keybd_event(mod_vk, 0, 2, 0)
                else:
                    # Fallback
                    import pyautogui
                    if key_lower == 'caps': key_lower = 'capslock'

                    if modifiers:
                        pyautogui.hotkey(*[m.lower() for m in modifiers] + [key_lower])
                    else:
                        pyautogui.press(key_lower)

            else:
                # Non-Windows fallback
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


class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, prefs, callback_save):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("400x550")
        self.configure(bg="#2d2d2d")
        self.prefs = prefs
        self.callback_save = callback_save
        self.transient(parent)  # Keep on top of main window

        # Style
        lbl_style = {"bg": "#2d2d2d", "fg": "white", "font": ("Segoe UI", 10)}

        # --- 1. GENERAL SETTINGS ---
        frame_gen = tk.LabelFrame(self, text="General", bg="#2d2d2d", fg="#00e676", font=("Segoe UI", 10, "bold"))
        frame_gen.pack(fill="x", padx=10, pady=10)

        # Auto-start PC
        self.var_autostart_pc = tk.BooleanVar(value=prefs.get("autostart_pc", False))
        tk.Checkbutton(frame_gen, text="Start app when computer starts", variable=self.var_autostart_pc,
                       bg="#2d2d2d", fg="white", selectcolor="#2d2d2d", activebackground="#2d2d2d").pack(anchor="w",
                                                                                                         padx=5)

        # Auto-start Server
        self.var_autostart_server = tk.BooleanVar(value=prefs.get("autostart_server", False))
        tk.Checkbutton(frame_gen, text="Auto-start server when app opens", variable=self.var_autostart_server,
                       bg="#2d2d2d", fg="white", selectcolor="#2d2d2d", activebackground="#2d2d2d").pack(anchor="w",
                                                                                                         padx=5)

        # Admin Rights
        self.var_admin = tk.BooleanVar(value=prefs.get("run_as_admin", False))
        tk.Checkbutton(frame_gen, text="Request Admin Rights (Task Manager fix)", variable=self.var_admin,
                       bg="#2d2d2d", fg="white", selectcolor="#2d2d2d", activebackground="#2d2d2d").pack(anchor="w",
                                                                                                         padx=5)

        # Port
        tk.Frame(frame_gen, height=1, bg="#444").pack(fill="x", pady=5)
        f_port = tk.Frame(frame_gen, bg="#2d2d2d")
        f_port.pack(fill="x", padx=5)
        tk.Label(f_port, text="Server Port:", **lbl_style).pack(side="left")
        self.ent_port = tk.Entry(f_port, width=8, bg="#3d3d3d", fg="white", insertbackground="white")
        self.ent_port.insert(0, str(prefs.get("port", 8080)))
        self.ent_port.pack(side="right")

        # --- 2. ABOUT & UPDATES ---
        frame_about = tk.LabelFrame(self, text="About", bg="#2d2d2d", fg="#00e676", font=("Segoe UI", 10, "bold"))
        frame_about.pack(fill="x", padx=10, pady=10)

        tk.Label(frame_about, text=f"Current Version: {APP_VERSION}", **lbl_style).pack(anchor="w", padx=5, pady=2)

        btn_frame = tk.Frame(frame_about, bg="#2d2d2d")
        btn_frame.pack(fill="x", pady=5)

        tk.Button(btn_frame, text="GitHub Page", bg="#3d3d3d", fg="white", relief="flat",
                  command=lambda: webbrowser.open(GITHUB_URL)).pack(side="left", padx=5)

        tk.Button(btn_frame, text="Check for Updates", bg="#2196f3", fg="white", relief="flat",
                  command=self.check_update).pack(side="right", padx=5)

        # --- SAVE BUTTONS ---
        frame_b = tk.Frame(self, bg="#2d2d2d")
        frame_b.pack(side="bottom", fill="x", pady=15)
        tk.Button(frame_b, text="Save & Close", bg="#00e676", fg="black", font=("Segoe UI", 10, "bold"),
                  command=self.save_and_close).pack(pady=5, ipadx=20)

    def check_update(self):
        """Simple check against GitHub releases API."""
        try:
            api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
            req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                data = json.load(response)
                latest_tag = data.get("tag_name", "").replace("v", "")

                # Basic string compare (ideally use packaging.version)
                if latest_tag != APP_VERSION:
                    ans = messagebox.askyesno("Update Available",
                                              f"New version {latest_tag} is available!\n\nOpen download page?")
                    if ans:
                        webbrowser.open(data.get("html_url", GITHUB_URL))
                else:
                    messagebox.showinfo("Up to Date", "You are using the latest version.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not check for updates:\n{e}")

    def save_and_close(self):
        # Validate Port
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
            "port": port
        }
        self.callback_save(new_prefs)
        self.destroy()


class ServerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"Use As Server v{APP_VERSION}")
        self.root.geometry("600x850")
        self.root.configure(bg="#1e1e1e")

        # Tray Icon variable
        self.tray_icon = None

        # --- LOAD PREFERENCES ---
        self.prefs = {
            "autostart_pc": False,
            "autostart_server": False,
            "run_as_admin": False,
            "port": 8080
        }
        self.load_preferences()


        try:
            if Path("icon.ico").exists():
                self.root.iconbitmap("icon.ico")
        except:
            pass

        self.update_queue = queue.Queue()
        self.server = None
        self.is_running = False

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- AUTO START SERVER LOGIC ---
        if self.prefs["autostart_server"]:
            self.root.after(1000, self.toggle_server)

        self.root.after(100, self.process_queue)
        self.root.after(50, self.update_preview)

    def create_default_icon(self):
        """Generates a simple green box icon if icon.ico is missing."""
        width = 64
        height = 64
        color1 = "#00e676"
        color2 = "#1e1e1e"
        image = Image.new('RGB', (width, height), color1)
        dc = ImageDraw.Draw(image)
        dc.rectangle((width // 4, height // 4, 3 * width // 4, 3 * height // 4), fill=color2)
        return image

    def minimize_to_tray(self):
        """Hide window and show system tray icon."""
        self.root.withdraw()  # Hide the main window

        image = None
        try:
            if Path("icon.ico").exists():
                image = Image.open("icon.ico")
            else:
                image = self.create_default_icon()
        except:
            image = self.create_default_icon()

        # Define Tray Menu
        menu = (
            item('Restore', self.restore_from_tray, default=True),
            item('Stop Server & Quit', self.quit_app)
        )

        self.tray_icon = pystray.Icon("UseAsServer", image, "Use As Server", menu)
        # Run detached so it doesn't block the Tkinter loop
        self.tray_icon.run_detached()

    def restore_from_tray(self, icon=None, item=None):
        """Stop tray icon and show window."""
        if self.tray_icon:
            self.tray_icon.stop()
            self.tray_icon = None

        self.root.after(0, self.root.deiconify)  # Show window safely

    def quit_app(self, icon=None, item=None):
        """Clean shutdown from tray."""
        if self.tray_icon:
            self.tray_icon.stop()
        self.root.after(0, self.on_closing)

    def load_preferences(self):
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, 'r') as f:
                    data = json.load(f)
                    self.prefs.update(data)
        except Exception as e:
            print(f"Error loading prefs: {e}")

    def save_preferences(self, new_prefs=None):
        """Save JSON and Apply System Changes (Registry)."""
        if new_prefs:
            self.prefs = new_prefs

        # 1. Save JSON
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(self.prefs, f, indent=4)
        except Exception as e:
            print(f"Error saving json: {e}")

        # 2. Apply Windows Registry Changes (Auto-start PC)
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
                    except FileNotFoundError:
                        pass  # Key didn't exist, which is fine
                winreg.CloseKey(key)
            except Exception as e:
                print(f"Registry Error: {e}")

    def open_settings(self):
        SettingsDialog(self.root, self.prefs, self.save_preferences)

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background="#1e1e1e", borderwidth=0)
        style.configure("TNotebook.Tab", background="#2d2d2d", foreground="white", padding=[15, 5])
        style.map("TNotebook.Tab", background=[("selected", "#00e676")], foreground=[("selected", "#000")])
        style.configure("TFrame", background="#1e1e1e")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        self.tab_dashboard = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_dashboard, text="  Dashboard  ")
        self.build_dashboard(self.tab_dashboard)

        self.tab_sharing = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_sharing, text="  Sharing  ")
        self.build_sharing(self.tab_sharing)

        self.tab_cam = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_cam, text=" 📷 Camera Studio ")
        self.build_camera_studio()

    def build_camera_studio(self):
        p = self.tab_cam

        # --- LEFT: CONTROLS ---
        left_panel = tk.Frame(p, bg="#1e1e1e", width=250)
        left_panel.pack(side="left", fill="y", padx=10, pady=10)

        # 1. Output Resolution
        lf_res = tk.LabelFrame(left_panel, text="Resolution / Aspect Ratio", bg="#1e1e1e", fg="white")
        lf_res.pack(fill="x", pady=5)

        self.combo_res = ttk.Combobox(lf_res, values=[
            "1280x720 (16:9) - Standard",
            "1920x1080 (16:9) - Full HD",
            "800x600 (4:3) - Old School",
            "1024x768 (4:3) - Standard 4:3",
            "720x720 (1:1) - Square"
        ], state="readonly")
        self.combo_res.current(0)
        self.combo_res.pack(fill="x", padx=5, pady=5)
        self.combo_res.bind("<<ComboboxSelected>>", self.change_aspect_ratio)

        # 2. Background Settings
        lf_bg = tk.LabelFrame(left_panel, text="Background Effects", bg="#1e1e1e", fg="white")
        lf_bg.pack(fill="x", pady=5)

        self.var_bg = tk.StringVar(value="none")
        tk.Radiobutton(lf_bg, text="None", variable=self.var_bg, value="none",
                       bg="#1e1e1e", fg="white", selectcolor="#2d2d2d", command=self.update_cam_settings).pack(
            anchor="w")
        tk.Radiobutton(lf_bg, text="Blur", variable=self.var_bg, value="blur",
                       bg="#1e1e1e", fg="white", selectcolor="#2d2d2d", command=self.update_cam_settings).pack(
            anchor="w")
        tk.Radiobutton(lf_bg, text="Replace Image", variable=self.var_bg, value="image",
                       bg="#1e1e1e", fg="white", selectcolor="#2d2d2d", command=self.update_cam_settings).pack(
            anchor="w")

        tk.Button(lf_bg, text="Select Image...", bg="#3d3d3d", fg="white",
                  command=self.select_bg_image).pack(fill="x", padx=5, pady=5)

        # 3. Picture Settings
        lf_pic = tk.LabelFrame(left_panel, text="Picture Adjust", bg="#1e1e1e", fg="white")
        lf_pic.pack(fill="x", pady=5)

        self.var_mirror = tk.BooleanVar(value=False)
        self.var_flip = tk.BooleanVar(value=False)

        tk.Checkbutton(lf_pic, text="Mirror Video", variable=self.var_mirror,
                       bg="#1e1e1e", fg="white", selectcolor="#2d2d2d", command=self.update_cam_settings).pack(
            anchor="w")
        tk.Checkbutton(lf_pic, text="Flip Vertically", variable=self.var_flip,
                       bg="#1e1e1e", fg="white", selectcolor="#2d2d2d", command=self.update_cam_settings).pack(
            anchor="w")

        tk.Label(lf_pic, text="Brightness Boost", bg="#1e1e1e", fg="gray").pack(anchor="w", pady=(5, 0))
        self.scale_bright = tk.Scale(lf_pic, from_=-100, to=100, orient="horizontal",
                                     bg="#1e1e1e", fg="white", highlightthickness=0,
                                     command=lambda x: self.update_cam_settings())
        self.scale_bright.set(0)
        self.scale_bright.pack(fill="x")

        # 4. Output Buttons (UPDATED PART)
        lf_out = tk.LabelFrame(left_panel, text="Output", bg="#1e1e1e", fg="white")
        lf_out.pack(fill="x", pady=5)

        # Save these to 'self' variables so we can change text later
        self.btn_cs_obs = tk.Button(lf_out, text="Start OBS Camera", bg="#9c27b0", fg="white",
                                    command=lambda: self.toggle_camera("OBS Virtual Camera"))
        self.btn_cs_obs.pack(fill="x", pady=2)

        self.btn_cs_unity = tk.Button(lf_out, text="Start Unity Camera", bg="#673ab7", fg="white",
                                      command=lambda: self.toggle_camera("Unity Video Capture"))
        self.btn_cs_unity.pack(fill="x", pady=2)

        # --- RIGHT: PREVIEW ---
        right_panel = tk.Frame(p, bg="#000000")
        right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.lbl_preview = tk.Label(right_panel, text="Waiting for connection...", bg="black", fg="gray")
        self.lbl_preview.place(relx=0.5, rely=0.5, anchor="center")

    def change_aspect_ratio(self, event=None):
        if not self.server: return

        # Parse the selection string like "1280x720 (16:9)..."
        selection = self.combo_res.get()
        try:
            res_part = selection.split(" ")[0]  # "1280x720"
            w_str, h_str = res_part.split("x")
            new_w = int(w_str)
            new_h = int(h_str)

            # Update Server Variables
            self.server.target_w = new_w
            self.server.target_h = new_h

            print(f"Resolution changed to {new_w}x{new_h}")

            # Restart VCam if running to apply new resolution
            if self.server._vcam_running:
                # Find which device was active (OBS or Unity)
                # We can just restart with the current VCam device if stored,
                # or just stop it and let user restart.
                # Auto-restart logic:
                current_device = self.server._vcam.device
                self.server.stop_virtual_camera()
                # Tiny delay to allow cleanup
                self.root.after(500, lambda: self.server.start_virtual_camera(current_device))

        except Exception as e:
            print(f"Error changing resolution: {e}")



    def build_dashboard(self, parent):
        # Header
        header = tk.Frame(parent, bg="#1e1e1e")
        header.pack(fill='x', pady=20)

        tk.Label(header, text="Use As Server", font=("Segoe UI", 24, "bold"), bg="#1e1e1e", fg="white").pack(side="left", padx=20)

        # BUTTONS FRAME (Settings + Tray)
        btn_box = tk.Frame(header, bg="#1e1e1e")
        btn_box.pack(side="right", padx=20)

        # 1. Tray Button (NEW)
        tk.Button(btn_box, text="⬇️Minimize to Tray", font=("Segoe UI", 9), bg="#3d3d3d", fg="white",
                  relief='flat', command=self.minimize_to_tray).pack(side="left", padx=5)

        # 2. Settings Button
        tk.Button(btn_box, text="⚙️ Settings", font=("Segoe UI", 9), bg="#3d3d3d", fg="white",
                  relief='flat', command=self.open_settings).pack(side="left", padx=5)

        # Status
        self.status_label = tk.Label(parent, text="🔴 Offline", font=("Segoe UI", 10), bg="#2d2d2d",
                                     fg="#cf6679", padx=10, pady=5)
        self.status_label.pack(pady=10)

        self.ip_var = tk.StringVar(value="Not Running")
        tk.Entry(parent, textvariable=self.ip_var, font=("Consolas", 12), justify='center', bg="#2d2d2d",
                 fg="#00e676", relief='flat', state='readonly').pack(pady=5, ipadx=10, ipady=5)

        self.btn_start = tk.Button(parent, text="START SERVER", font=("Segoe UI", 12, "bold"), bg="#00e676",
                                   fg="black", activebackground="#00c853", relief='flat',
                                   command=self.toggle_server)
        self.btn_start.pack(pady=15, ipadx=30, ipady=10)

        # --- FEATURES FRAME ---
        features_frame = tk.LabelFrame(parent, text="Features", bg="#1e1e1e", fg="white", font=("Segoe UI", 10, "bold"))
        features_frame.pack(fill='x', padx=20, pady=10)

        # Camera Row
        cam_frame = tk.Frame(features_frame, bg="#1e1e1e")
        cam_frame.pack(fill='x', pady=2, padx=5)

        self.btn_obs = tk.Button(cam_frame, text="Start OBS Camera", font=("Segoe UI", 9), bg="#9c27b0", fg="white",
                                 relief='flat', command=lambda: self.toggle_camera("OBS Virtual Camera"), state='disabled')
        self.btn_obs.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_unity = tk.Button(cam_frame, text="Start Unity Camera", font=("Segoe UI", 9), bg="#673ab7", fg="white",
                                   relief='flat', command=lambda: self.toggle_camera("Unity Video Capture"), state='disabled')
        self.btn_unity.pack(side="right", fill="x", expand=True, padx=(2, 0))

        # Audio Button
        self.btn_audio = tk.Button(features_frame, text="🔊 Audio Streaming", font=("Segoe UI", 10), bg="#2196f3",
                                   fg="white", relief='flat', command=self.toggle_audio, state='disabled')
        self.btn_audio.pack(fill='x', pady=2, padx=5)

        # Log
        log_frame = tk.LabelFrame(parent, text="Activity Log", bg="#1e1e1e", fg="#b0b0b0", font=("Consolas", 9))
        log_frame.pack(fill='both', expand=True, padx=20, pady=10)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=5, bg="#2d2d2d", fg="white", font=("Consolas", 9), relief='flat')
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)

    def build_sharing(self, parent):
        # Clipboard
        frame_clip = tk.LabelFrame(parent, text="Clipboard", bg="#1e1e1e", fg="white", font=("Segoe UI", 10, "bold"))
        frame_clip.pack(fill='x', padx=20, pady=10)
        self.txt_clip = tk.Text(frame_clip, height=3, bg="#2d2d2d", fg="white", relief='flat', font=("Consolas", 10))
        self.txt_clip.pack(fill='x', padx=10, pady=5)
        btn_frame = tk.Frame(frame_clip, bg="#1e1e1e")
        btn_frame.pack(fill='x', padx=10, pady=5)
        tk.Button(btn_frame, text="Copy to PC Clipboard", bg="#2d2d2d", fg="white", relief='flat', command=self.copy_to_pc).pack(side='left')
        tk.Button(btn_frame, text="Send Text to Phone", bg="#00e676", fg="black", relief='flat', command=self.send_text_to_phone).pack(side='right')

        # File Transfer
        frame_file = tk.LabelFrame(parent, text="File Transfer", bg="#1e1e1e", fg="white", font=("Segoe UI", 10, "bold"))
        frame_file.pack(fill='both', expand=True, padx=20, pady=10)
        self.list_files = tk.Listbox(frame_file, bg="#2d2d2d", fg="white", relief='flat', font=("Segoe UI", 9))
        self.list_files.pack(fill='both', expand=True, padx=10, pady=5)
        self.list_files.bind("<Double-Button-1>", self.open_file)
        btn_frame_f = tk.Frame(frame_file, bg="#1e1e1e")
        btn_frame_f.pack(fill='x', padx=10, pady=10)
        tk.Button(btn_frame_f, text="📂 Open Received Folder", bg="#2d2d2d", fg="white", relief='flat', command=self.open_folder).pack(side='left')
        tk.Button(btn_frame_f, text="📤 Send File to Phone", bg="#00e676", fg="black", relief='flat', command=self.send_file_pick).pack(side='right')

    def toggle_camera(self, device_name):
        """Handles logic for both camera buttons."""
        if not self.server: return

        if self.server._vcam_running:
            # STOP CAMERA
            self.server.stop_virtual_camera()

            # Reset BOTH buttons to default state
            self.btn_obs.config(text="Start OBS Camera", bg="#9c27b0", state="normal")
            self.btn_unity.config(text="Start Unity Camera", bg="#673ab7", state="normal")

        else:
            # START CAMERA (Specific Driver)
            self.server.start_virtual_camera(device_name)

            # Update UI based on success check (allow small delay for thread to set flag)
            self.root.after(100, lambda: self._update_cam_buttons(device_name))

    def toggle_server(self):
        if not self.is_running:
            self.btn_start.config(text="STOP SERVER", bg="#cf6679", fg="white")
            self.status_label.config(text="🟡 Starting...", fg="orange")
            threading.Thread(target=self.start_sequence, daemon=True).start()
        else:
            self.stop_server()

    def update_cam_settings(self):
        if self.server:
            self.server.bg_mode = self.var_bg.get()
            self.server.is_mirrored = self.var_mirror.get()
            self.server.is_flipped = self.var_flip.get()
            self.server.brightness_boost = int(self.scale_bright.get())

    def select_bg_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path and self.server:
            self.server.bg_image_path = path
            self.server.bg_image_cache = None
            self.var_bg.set("image")
            self.update_cam_settings()

    def update_preview(self):
        if self.server and self.server.latest_preview_frame is not None:
            try:
                frame = self.server.latest_preview_frame
                if frame.shape[0] > 0:
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.lbl_preview.config(image=imgtk, text="")
                    self.lbl_preview.image = imgtk
            except Exception:
                pass
        self.root.after(33, self.update_preview)

    def _update_cam_buttons(self, active_device):
        """Helper to update button states (Color & Text) after starting/stopping."""

        # Check if server and vcam are running
        is_running = self.server and self.server._vcam_running

        if is_running:
            # --- CAMERA IS ON ---
            if active_device == "OBS Virtual Camera":
                # Update Dashboard Button
                self.btn_obs.config(text="⏹ Stop OBS", bg="#cf6679")
                self.btn_unity.config(state="disabled", bg="#424242")

                # Update Camera Studio Button (NEW)
                if hasattr(self, 'btn_cs_obs'):
                    self.btn_cs_obs.config(text="⏹ Stop OBS Camera", bg="#cf6679")
                if hasattr(self, 'btn_cs_unity'):
                    self.btn_cs_unity.config(state="disabled", bg="#424242")

            else:
                # Update Dashboard Button
                self.btn_unity.config(text="⏹ Stop Unity", bg="#cf6679")
                self.btn_obs.config(state="disabled", bg="#424242")

                # Update Camera Studio Button (NEW)
                if hasattr(self, 'btn_cs_unity'):
                    self.btn_cs_unity.config(text="⏹ Stop Unity Camera", bg="#cf6679")
                if hasattr(self, 'btn_cs_obs'):
                    self.btn_cs_obs.config(state="disabled", bg="#424242")
        else:
            # --- CAMERA IS OFF (Reset All) ---
            # Dashboard
            self.btn_obs.config(state="normal", bg="#9c27b0", text="Start OBS Camera")
            self.btn_unity.config(state="normal", bg="#673ab7", text="Start Unity Camera")

            # Camera Studio
            if hasattr(self, 'btn_cs_obs'):
                self.btn_cs_obs.config(state="normal", bg="#9c27b0", text="Start OBS Camera")
            if hasattr(self, 'btn_cs_unity'):
                self.btn_cs_unity.config(state="normal", bg="#673ab7", text="Start Unity Camera")

    def start_sequence(self):
        try:
            if self.server: self.server.stop()

            # USE CUSTOM PORT FROM SETTINGS
            port = self.prefs.get("port", 8080)
            self.server = UnifiedRemoteServer(port=port, update_queue=self.update_queue)
            self.server.start()

            for _ in range(20):
                if self.server._loop and self.server._loop.is_running():
                    ip = self.server.get_local_ip()
                    self.update_queue.put(("server_ready", f"{ip}:{port}"))
                    return
                time.sleep(0.1)
            raise Exception(f"Server timed out on port {port}.")
        except Exception as e:
            self.update_queue.put(("error", str(e)))
            if self.server: self.server.stop()

    # ... (Add existing helper methods: stop_server, toggle_vcam, etc. here) ...

    # Copied helpers for completeness of the GUI block (abbreviated)
    def stop_server(self):
        if self.server:
            self.server.stop()

        self.is_running = False
        self.btn_start.config(text="START SERVER", bg="#00e676", fg="black")
        self.status_label.config(text="🔴 Offline", fg="#cf6679")
        self.ip_var.set("Not Running")

        # --- DISABLE DASHBOARD BUTTONS ---
        if hasattr(self, 'btn_obs'):
            self.btn_obs.config(state='disabled', bg="#424242", text="Start OBS Camera")
        if hasattr(self, 'btn_unity'):
            self.btn_unity.config(state='disabled', bg="#424242", text="Start Unity Camera")
        if hasattr(self, 'btn_audio'):
            self.btn_audio.config(state='disabled', bg="#424242", text="🔊 Audio Streaming")

        # --- DISABLE CAMERA STUDIO BUTTONS (NEW) ---
        if hasattr(self, 'btn_cs_obs'):
            self.btn_cs_obs.config(state='disabled', bg="#424242", text="Start OBS Camera")
        if hasattr(self, 'btn_cs_unity'):
            self.btn_cs_unity.config(state='disabled', bg="#424242", text="Start Unity Camera")

    def toggle_vcam(self):
        if not self.server._vcam_running:
            self.server.start_virtual_camera()
            self.btn_vcam.config(text="⏹ Stop Camera", bg="#cf6679")
        else:
            self.server.stop_virtual_camera()
            self.btn_vcam.config(text="📹 Virtual Camera", bg="#9c27b0")

    def toggle_audio(self):
        if not self.server._streaming_audio:
            self.server.start_audio_streaming()
            self.btn_audio.config(text="🔇 Stop Audio", bg="#cf6679")
        else:
            self.server.stop_audio_streaming()
            self.btn_audio.config(text="🔊 Audio Streaming", bg="#2196f3")

    def send_text_to_phone(self):
        text = self.txt_clip.get("1.0", tk.END).strip()
        if text and self.server: self.server.send_to_android("clipboard_text", text)

    def send_file_pick(self):
        if not self.server or not self.is_running: return
        paths = filedialog.askopenfilenames()
        if paths: threading.Thread(target=self.process_multiple_files, args=(paths,), daemon=True).start()

    def process_multiple_files(self, paths):
        for p in paths:
            self.server.send_file_to_phone_thread(p)
            time.sleep(0.5)

    def copy_to_pc(self):
        pyperclip.copy(self.txt_clip.get("1.0", tk.END).strip())

    def open_folder(self):
        try:
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            os.startfile(SAVE_DIR)
        except:
            pass

    def open_file(self, event):
        sel = self.list_files.curselection()
        if sel:
            try:
                os.startfile(SAVE_DIR / self.list_files.get(sel[0]))
            except:
                pass

    def process_queue(self):
        try:
            while True:
                kind, data = self.update_queue.get_nowait()

                if kind == "log":
                    self.log(data)

                elif kind == "server_ready":
                    self.is_running = True
                    self.status_label.config(text="🟢 Online", fg="#00e676")
                    self.ip_var.set(f"ws://{data}")
                    self.log(f"Server ready on {data}")

                    # Enable buttons
                    if hasattr(self, 'btn_obs'): self.btn_obs.config(state='normal', bg="#9c27b0")
                    if hasattr(self, 'btn_unity'): self.btn_unity.config(state='normal', bg="#673ab7")
                    if hasattr(self, 'btn_audio'): self.btn_audio.config(state='normal', bg="#2196f3")

                elif kind == "error":
                    self.log(f"Error: {data}")
                    messagebox.showerror("Error", data)
                    self.stop_server()

                # --- FIX: ADD MISSING HANDLERS HERE ---

                elif kind == "clipboard":
                    # Update the clipboard text box
                    self.txt_clip.delete("1.0", tk.END)
                    self.txt_clip.insert(tk.END, data)
                elif kind == "audio_status":
                    is_on = data  # True or False
                    if is_on:
                        self.btn_audio.config(text="🔇 Stop Audio", bg="#cf6679")
                    else:
                        self.btn_audio.config(text="🔊 Audio Streaming", bg="#2196f3")

                elif kind == "file_received":
                    # Add filename to the listbox
                    # data is the full path, we just want the filename for the list
                    import os
                    filename = os.path.basename(data)
                    self.list_files.insert(tk.END, filename)
                    self.list_files.see(tk.END)  # Scroll to bottom

        except queue.Empty:
            pass
        except Exception as e:
            print(f"Queue Error: {e}")

        # Keep checking the queue
        self.root.after(200, self.process_queue)

    def log(self, msg):
        self.log_text.insert(tk.END, f"> {msg}\n")
        self.log_text.see(tk.END)

    def on_closing(self):
        if self.is_running: self.stop_server()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    # 1. Check if we need Admin Rights based on previous settings
    must_be_admin = False
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, 'r') as f:
                data = json.load(f)
                must_be_admin = data.get("run_as_admin", False)
    except:
        pass

    # 2. Check if we ARE Admin
    is_admin = False
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
    except:
        pass

    # 3. If we need admin but don't have it -> Restart as Admin
    if must_be_admin and not is_admin:
        print("⚠️ Restarting as Administrator...")
        # Re-run the script with Admin privileges
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, __file__, None, 1
        )
        sys.exit()

    # 4. Normal Startup
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    app = ServerGUI()
    app.run()

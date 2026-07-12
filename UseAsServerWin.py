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
APP_VERSION = "1.6"
GITHUB_REPO = "manjeetdeswal/Use-As-Server" 
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

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


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
        """Find broadcast address for every interface, plus universal."""
        # Enforce universal broadcast IPs automatically
        addresses = {'<broadcast>', '255.255.255.255'}
        try:
            hostname = socket.gethostname()
            local_ips = socket.gethostbyname_ex(hostname)[2]
            for ip in local_ips:
                # Ignore loopback adapters
                if not ip.startswith("127."):
                    parts = ip.split('.')
                    if len(parts) == 4:
                        # Append the standard /24 subnet broadcast 
                        broadcast = f"{parts[0]}.{parts[1]}.{parts[2]}.255"
                        addresses.add(broadcast)
        except Exception:
            pass
        return list(addresses)

    def run(self):
        # Explicitly set IPPROTO_UDP for broader OS compatibility
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        # Prevent "Address already in use" zombie locks
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except AttributeError:
            pass
            
        sock.settimeout(0.2)

        message = f"UNIFIED_REMOTE_SERVER:{self.port}".encode('utf-8')
        logging.info(f"Starting discovery broadcast for port {self.port}...")

        while self.running:
            targets = self.get_broadcast_addresses()
            for target in targets:
                try:
                    # We broadcast TO port 8888 (The phone listens on this port)
                    sock.sendto(message, (target, 8888))
                except OSError:
                    # Silently ignore "Network is unreachable" for disconnected adapters
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

        self.clients = {}
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
                        del self.clients[client]
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

    def make_handler(self):
        """Return a unified async handler that manages client state, routing, 
        and virtual gamepads with full rumble support and anti-ghost clean-up.
        """
        import asyncio
        import json
        import threading

        async def handler(websocket):
            remote_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
            remote_ip = websocket.remote_address[0]
            
            # 1. Initialize Client tracking metadata
            self.loop = asyncio.get_running_loop()
            self.clients[websocket] = {"ip": remote_ip, "name": f"Unknown ({remote_ip})"}
            self._put("client_connected", self.clients[websocket])
            self._put("log", f"🔌 Client connected: {remote_addr}")
            self._put("client_count", len(self.clients))
            
            # --- SEND SERVER VERSION ---
            async def send_version_delayed():
                await asyncio.sleep(1.5)  # Wait 1.5s for Android to load the listener
                try:
                    version_msg = json.dumps({"type": "server_version", "payload": APP_VERSION})
                    await websocket.send(version_msg)
                except Exception:
                    pass
            
            asyncio.create_task(send_version_delayed())

            # 2. Initialize Virtual Gamepad with Rumble Support
            try:
                import vgamepad as vg
                dev = vg.VX360Gamepad()
                self.client_gamepads[websocket] = dev
                websocket.last_rumble = -1.0 
                
                def rumble_callback(client, target, large_motor, small_motor, led_number, user_data):
                    # Calculate intensity and round it to avoid micro-fluctuations
                    intensity = max(large_motor, small_motor) / 255.0
                    intensity = round(intensity, 2)
                    
                    # ONLY send a message if the intensity actually changed!
                    if getattr(websocket, "last_rumble", -1.0) != intensity:
                        websocket.last_rumble = intensity
                        
                        payload = json.dumps({"type": "rumble", "payload": str(intensity)})
                        
                        # ✅ TARGET ONLY THE SPECIFIC PHONE THAT OWNS THIS CONTROLLER
                        if self.loop and self.loop.is_running():
                            asyncio.run_coroutine_threadsafe(websocket.send(payload), self.loop)
                            
                # Attach the vibration callback to the virtual controller
                dev.register_notification(callback_function=rumble_callback)
                self._put("log", f"🎮 Virtual Xbox 360 Controller mounted for {remote_ip}")
            except Exception as e:
                self._put("log", f"❌ ViGEmBus Error (Is it installed?): {e}")
                # Optional: Decide whether to return or continue if app functions without gamepad

            # 3. Message Processing & Routing Loop
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
                        raw_payload = data.get("payload", "")

                        # Smart Unwrap Payload
                        payload = raw_payload
                        if isinstance(raw_payload, str) and raw_payload.strip().startswith("{"):
                            try:
                                payload = json.loads(raw_payload)
                            except Exception as e:
                                self._put("log", f"⚠️ JSON Parse Error: {e}")

                        # --- ROUTING MESSAGES ---
                        if msg_type == "device_info":
                            name = payload.get("name", self.clients[websocket]["name"])
                            self.clients[websocket]["name"] = name
                            self._put("client_updated", self.clients[websocket])
                            self._put("log", f"📱 Device Identified: {name}")

                        elif msg_type == "theme_update":
                            theme_name = payload.get("name", "Dark Carbon")
                            self._put("theme_changed", theme_name)

                        elif msg_type == "video_frame":
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

                        # --- MICROPHONE HANDLING ---
                        elif msg_type == "mic_start":
                            try:
                                self._mic_active = True
                                new_rate = int(payload)
                                self._put("log", f"🎤 Mic started at {new_rate}Hz")

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
                            self._mic_active = False
                            if hasattr(self, '_mic_stream') and self._mic_stream:
                                try:
                                    self._mic_stream.close()
                                except:
                                    pass
                                self._mic_stream = None
                            self._put("log", "🎤 Mic Stopped")

                        elif msg_type == "audio_frame":
                            self._handle_audio_frame(payload)

                        # --- SYSTEM CONTROL (Virtual Camera) ---
                        elif msg_type == "system_control":
                            try:
                                control_data = json.loads(payload) if isinstance(payload, str) else payload
                                action = control_data.get("action")
                                
                                if action == "start_vcam":
                                    device_name = control_data.get("device")
                                    self._put("log", f"📲 Mobile requested Auto-Start: {device_name}")
                                    self.start_virtual_camera(device_name)
                                    self._put("camera_status", device_name)
                                    
                                elif action == "stop_vcam":
                                    self._put("log", "📲 Mobile requested Stop Virtual Cam")
                                    self.stop_virtual_camera()
                                    self._put("camera_status", None)
                            except Exception as e:
                                self._put("log", f"⚠️ System control error: {e}")

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
                            self._handle_gamepad_state(raw_payload, websocket)

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
                            self._handle_file_transfer(payload)

                        elif msg_type == "display_request":
                            self._handle_display_request(payload)

                        elif msg_type == "heartbeat":
                            await websocket.send(json.dumps({"type": "heartbeat", "payload": "pong"}))

                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        self._put("log", f"⚠️ Message Error: {e}")

            except websockets.ConnectionClosedOK:
                pass
            except Exception as e:
                self._put("log", f"⚠️ Handler error: {e}")
            finally:
                # 4. Clean up Client Records
                if websocket in self.clients:
                    self._put("client_disconnected", self.clients[websocket])
                    del self.clients[websocket]
                    self._put("client_count", len(self.clients))
                
                # 5. --- ANTI-GHOST CONTROLLER FIX WITH VIBRATION CLEANUP ---
                if websocket in self.client_gamepads:
                    try:
                        orphan_dev = self.client_gamepads[websocket]
                        
                        # Break the callback reference to clear memory allocations
                        orphan_dev.unregister_notification()
                        
                        del self.client_gamepads[websocket]
                        
                        # Unmount controller cleanly from ViGEmBus
                        del orphan_dev 
                        self._put("log", f"🎮 Virtual Gamepad cleanly unmounted for {remote_ip}")
                    except Exception as e:
                        self._put("log", f"⚠️ Error clearing gamepad: {e}")
                        
                self._put("log", f"🔌 Disconnected: {remote_ip}")

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

            print(f" COMPLETE. Sent {total_sent} bytes.")
            self._put("log", f" Sent {filename}")

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
                print(f" COMPLETE: File saved to {abs_path}")
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
            if isinstance(payload, str):
                try:
                    frame_data = json.loads(payload)
                except:
                    return
            else:
                frame_data = payload

            b64_data = frame_data.get('data')
            if not b64_data: return

            # We don't need 'rotation' or 'is_front' anymore because the phone pre-rotates it!
            
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
            print(f"❌ Frame Drop Error: {e}")

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
        """Handle gamepad input and haptic feedback (rumble)."""
        try:
            import json
            import asyncio
            
            # FIX: Only parse if it's a string
            if isinstance(payload, str):
                state = json.loads(payload)
            else:
                state = payload

            # Ensure dictionary exists
            if not hasattr(self, 'client_gamepads'):
                self.client_gamepads = {}

            if client_ws not in self.client_gamepads:
                try:
                    import vgamepad as vg
                    new_gamepad = vg.VX360Gamepad()
                    self.client_gamepads[client_ws] = new_gamepad
                    client_ws.last_rumble = -1.0 
                    
                    def rumble_callback(client, target, large_motor, small_motor, led_number, user_data):
                        try:
                            intensity = max(large_motor, small_motor) / 255.0
                            intensity = round(intensity, 2)
                            
                            if getattr(client_ws, "last_rumble", -1.0) != intensity:
                                client_ws.last_rumble = intensity
                                payload_str = json.dumps({"type": "rumble", "payload": str(intensity)})
                                
                                # 2. FIX: NEVER call client_ws.send() directly here! 
                                # Use the thread-safe broadcast queue to prevent ConcurrencyError disconnects.
                                self._send_to_clients_threadsafe(payload_str)
                        except Exception as e:
                            pass 

                    new_gamepad.register_notification(callback_function=rumble_callback)
                    self._put("log", f"🎮 Virtual Xbox Controller mapped with Rumble Support!")
                except Exception as e:
                    self._put("log", f"❌ Gamepad creation error: {e}")
                    return

            current_gamepad = self.client_gamepads[client_ws]
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
                'ls_btn': vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,  
                'rs_btn': vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB, 
                'thumbl': vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,  
                'thumbr': vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB  
            }

            buttons = state.get('buttons', {})
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
                        if pressed: current_gamepad.press_button(dpad_map[direction])
                        else: current_gamepad.release_button(dpad_map[direction])
                elif button in button_map:
                    if pressed: current_gamepad.press_button(button_map[button])
                    else: current_gamepad.release_button(button_map[button])

            left_x = float(state.get('leftStickX', 0.0))
            left_y = float(state.get('leftStickY', 0.0))
            right_x = float(state.get('rightStickX', 0.0))
            right_y = float(state.get('rightStickY', 0.0))

            current_gamepad.left_joystick_float(x_value_float=left_x, y_value_float=-left_y)
            current_gamepad.right_joystick_float(x_value_float=right_x, y_value_float=-right_y)

            left_trigger = float(state.get('leftTrigger', 0.0))
            right_trigger = float(state.get('rightTrigger', 0.0))

            current_gamepad.left_trigger_float(value_float=left_trigger)
            current_gamepad.right_trigger_float(value_float=right_trigger)
            current_gamepad.update()

        except Exception as e:
            self._put("log", f"❌ Gamepad error: {e}")

    def _handle_display_request(self, payload):
        """Handle display streaming request."""
        try:
            import json
            request = json.loads(payload)
            action = request.get('action')
            self._put("log", f"🖥️ Display request: {action}")

            if action in ['start_display', 'change_resolution']:
                self._display_width = int(request.get('width', 1280))
                self._display_height = int(request.get('height', 720))
                self._display_fps = int(request.get('fps', 30))
                self._display_quality = int(request.get('quality', 35))
                
             
                # Default to [0] (Primary) if missing
                self._display_monitor_indices = request.get('monitor_indices', [0])
                
                # Handle legacy single index if sent by old app version
                if 'monitor_index' in request and 'monitor_indices' not in request:
                     idx = int(request.get('monitor_index'))
                     self._display_monitor_indices = [idx] if idx >= 0 else [0, 1] # Fallback for "All"

                self._put("log", f"🖥️ Config: {self._display_width}x{self._display_height} (Monitors: {self._display_monitor_indices})")

            if action == 'start_display':
                if hasattr(self, '_display_thread') and getattr(self, '_display_thread') is not None and self._display_thread.is_alive():
                    self._stop_display_capture(wait_seconds=0.8)

                self._display_active = True
                import threading
                self._display_thread = threading.Thread(target=self._capture_screen_loop, daemon=True)
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
                    self._put("log", " Previous screen capture stopped.")
        except Exception as e:
            self._put("log", f"❌ Error stopping display capture: {e}")

    def _capture_screen_loop(self):
        """Continuously capture specific monitor or combined screens (Flicker-Free using MSS)."""
        try:
            import mss
            import cv2
            import base64
            import time
            import pyautogui
            import numpy as np

            target_indices = self._display_monitor_indices
            self._put("log", f"🖥️ Screen capture started via MSS. Targets: {target_indices}")

            current_quality = getattr(self, '_display_quality', 35)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), current_quality]
            
            last_mouse_pos = (0, 0)
            last_move_time = time.time()
            HIDE_TIMEOUT = 3.0

            with mss.mss() as sct:
                # sct.monitors[0] is all screens combined
                # sct.monitors[1] is monitor 1, sct.monitors[2] is monitor 2
                
                frame_cache = [None] * len(target_indices)

                while self._display_active:
                    start_time = time.time()
                    
                    target_w = self._display_width
                    target_h = self._display_height
                    target_fps = self._display_fps
                    frame_duration = 1.0 / max(1, target_fps)

                    try:
                        # 1. CAPTURE & CACHE
                        frames_ready = []
                        
                        for i, idx in enumerate(target_indices):
                            mss_idx = idx + 1 # Convert app index (0-based) to MSS index (1-based)
                            
                            if mss_idx < len(sct.monitors):
                                monitor = sct.monitors[mss_idx]
                                sct_img = sct.grab(monitor)
                                # mss returns BGRA, convert to BGR for OpenCV
                                img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                                
                                frame_cache[i] = img
                                frames_ready.append(img)
                            else:
                                # Fallback if monitor doesn't exist yet
                                if frame_cache[i] is not None:
                                    frames_ready.append(frame_cache[i])
                                else:
                                    frames_ready.append(None)

                        if any(f is None for f in frames_ready):
                            time.sleep(0.01)
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

                        # 3. DRAW MOUSE (Single Monitor Mode Only)
                        if len(target_indices) == 1:
                            try:
                                mx, my = pyautogui.position()
                                if (mx, my) != last_mouse_pos:
                                    last_mouse_pos = (mx, my)
                                    last_move_time = time.time()
                                
                                if time.time() - last_move_time < HIDE_TIMEOUT:
                                    monitor = sct.monitors[target_indices[0] + 1]
                                    local_mx = mx - monitor["left"]
                                    local_my = my - monitor["top"]
                                    
                                    # Ensure the base point is somewhat within the screen bounds
                                    if 0 <= local_mx < final_frame.shape[1] and 0 <= local_my < final_frame.shape[0]:
                                        
                                        # Define the polygon points for a classic mouse cursor arrow
                                        cursor_pts = np.array([
                                            [0, 0],     # Tip
                                            [0, 16],    # Left bottom corner
                                            [4, 12],    # Inner left
                                            [7, 19],    # Tail bottom left
                                            [10, 17],   # Tail bottom right
                                            [7, 11],    # Inner right
                                            [12, 11]    # Right corner
                                        ], np.int32)
                                        
                                        # Shift the points to the current mouse position
                                        pts = cursor_pts + [local_mx, local_my]
                                        pts = pts.reshape((-1, 1, 2))
                                        
                                        # Draw White Fill
                                        cv2.fillPoly(final_frame, [pts], (255, 255, 255))
                                        # Draw Black Outline (LINE_AA makes the edges smooth)
                                        cv2.polylines(final_frame, [pts], isClosed=True, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                                        
                            except: pass

                        # 4. FIT TO SCREEN
                        h, w = final_frame.shape[:2]
                        scale = min(target_w / w, target_h / h)
                        
                        if scale < 1.0:
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            final_frame = cv2.resize(final_frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                        # 5. ENCODE & SEND
                        success, buffer = cv2.imencode('.jpg', final_frame, encode_param)
                        if success:
                            # Header byte 0x03 means "This is a Video Frame"
                            header = b'\x03'
                            video_bytes = header + buffer.tobytes()
                            
                            if hasattr(self, '_send_bytes_to_clients_threadsafe'):
                                self._send_bytes_to_clients_threadsafe(video_bytes)
                            else:
                                import asyncio
                                asyncio.run_coroutine_threadsafe(
                                    self._broadcast_bytes(video_bytes), self.loop
                                )

                        # 6. FPS LIMIT
                        while (time.time() - start_time) < frame_duration:
                            pass

                    except Exception as e:
                        time.sleep(0.1)

            self._put("log", "🖥️ Screen capture stopped")

        except Exception as e:
            self._put("log", f"❌ Capture Fatal Error: {e}")
            self._display_active = False

    def _send_bytes_to_clients_threadsafe(self, data_bytes):
        """Thread-safe way to broadcast raw binary data to all connected clients."""
        if not hasattr(self, 'clients') or not self.clients:
            return

        async def broadcast():
            dead_clients = set()
            for ws in list(self.clients):
                try:
                    # Send raw bytes. websockets library automatically handles binary frames.
                    await ws.send(data_bytes)
                except websockets.exceptions.ConnectionClosed:
                    dead_clients.add(ws)
                except Exception as e:
                    self._put("log", f"⚠️ Binary send error: {e}")
                    dead_clients.add(ws)
            
            # Clean up disconnected clients
            for ws in dead_clients:
                if ws in self.clients:
                    del self.clients[ws]

        # Safely schedule the broadcast on the main event loop
        if hasattr(self, 'loop') and self.loop.is_running():
            import asyncio
            asyncio.run_coroutine_threadsafe(broadcast(), self.loop)
        else:
            self._put("log", "❌ Cannot send bytes: Async loop is not running.")

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

                    self._put("log", f" Stream Started: {current_rate}Hz")
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
            import time
            import platform

            # THE FIX: Check if payload is already a dict to prevent json.loads() crash!
            if isinstance(payload, str):
                import json
                event = json.loads(payload)
            else:
                event = payload

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
            
            # THE FIX: Check payload type
            if isinstance(payload, str):
                import json
                event = json.loads(payload)
            else:
                event = payload
                
            scroll_delta = event.get("scrollDelta", 0)

            if platform.system() == "Windows":
                # MOUSEEVENTF_WHEEL = 0x0800
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
        """Handle keyboard input (Respects Down/Up actions and Left/Right Modifiers)."""
        try:
            import platform
            import time
            import ctypes
            import json

            if isinstance(payload, str):
                event = json.loads(payload)
            else:
                event = payload

            key = event.get("key", "")
            modifiers = event.get("modifiers", [])
            action = event.get("action", "press")

            key_lower = key.lower()

            # --- BRIGHTNESS HANDLER ---
            if key_lower in ["brightnessup", "brightnessdown"]:
                try:
                    import screen_brightness_control as sbc
                    current_list = sbc.get_brightness()
                    if current_list:
                        current = current_list[0]
                        new_val = min(100, current + 10) if key_lower == "brightnessup" else max(0, current - 10)
                        import threading
                        threading.Thread(target=lambda: sbc.set_brightness(new_val)).start()
                except:
                    pass
                return

            # --- AUTO-SHIFT FIX ---
            if len(key) == 1 and key.isupper() and key.isalpha():
                if "shift" not in [m.lower() for m in modifiers]:
                    modifiers.append("shift")

            if platform.system() == "Windows":
                # Virtual-Key Codes including Left/Right variants
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
                    'shift': 0x10, 'shift_l': 0xA0, 'shift_r': 0xA1,
                    'ctrl': 0x11, 'ctrl_l': 0xA2, 'ctrl_r': 0xA3,
                    'alt': 0x12, 'alt_l': 0xA4, 'alt_r': 0xA5,
                    'meta': 0x5B, 'win': 0x5B, 'win_l': 0x5B, 'win_r': 0x5C
                }

                vk_code = VK_MAP.get(key_lower, 0)

                if vk_code:
                    scan_code = ctypes.windll.user32.MapVirtualKeyW(vk_code, 0)

                    def trigger_modifiers(is_down):
                        flags = 0 if is_down else 2  # 0=Down, 2=Up
                        for mod in modifiers:
                            mod_vk = VK_MAP.get(mod.lower(), 0)
                            if mod_vk:
                                mod_scan = ctypes.windll.user32.MapVirtualKeyW(mod_vk, 0)
                                ctypes.windll.user32.keybd_event(mod_vk, mod_scan, flags, 0)

                    if action == "down":
                        trigger_modifiers(True)
                        ctypes.windll.user32.keybd_event(vk_code, scan_code, 0, 0)

                    elif action == "up":
                        ctypes.windll.user32.keybd_event(vk_code, scan_code, 2, 0)
                        trigger_modifiers(False)

                    else:
                        trigger_modifiers(True)
                        ctypes.windll.user32.keybd_event(vk_code, scan_code, 0, 0)
                        if getattr(self, 'gaming_mode', False):
                            time.sleep(0.03)
                        ctypes.windll.user32.keybd_event(vk_code, scan_code, 2, 0)
                        trigger_modifiers(False)

                else:
                    # Fallback (PyAutoGUI)
                    import pyautogui
                    if key_lower == 'caps': key_lower = 'capslock'
                    
                    # Map Left/Right to PyAutoGUI equivalents
                    pyautogui_mod_map = {
                        'shift_l': 'shiftleft', 'shift_r': 'shiftright',
                        'ctrl_l': 'ctrlleft', 'ctrl_r': 'ctrlright',
                        'alt_l': 'altleft', 'alt_r': 'altright'
                    }
                    
                    mapped_mods = [pyautogui_mod_map.get(m.lower(), m.lower()) for m in modifiers]
                    keys_to_press = mapped_mods + [key_lower]

                    if action == "down":
                        for k in keys_to_press: pyautogui.keyDown(k)
                    elif action == "up":
                        for k in reversed(keys_to_press): pyautogui.keyUp(k)
                    else:
                        pyautogui.hotkey(*keys_to_press)

            else:
                # Linux/Mac logic
                import pyautogui
                key_map = {"Backspace": "backspace", "Enter": "enter", "Space": "space", "Tab": "tab", "Esc": "esc"}
                mapped_key = key_map.get(key, key.lower())
                keys_to_press = [m.lower() for m in modifiers] + [mapped_key]

                if action == "down":
                    for k in keys_to_press: pyautogui.keyDown(k)
                elif action == "up":
                    for k in reversed(keys_to_press): pyautogui.keyUp(k)
                else:
                    pyautogui.hotkey(*keys_to_press)

        except Exception as e:
            if hasattr(self, '_put'):
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
        self.geometry("460x580")
        self.resizable(False, False)
        self.prefs = prefs
        self.callback_save = callback_save
        self.configure(fg_color="#141414")
        self.transient(parent)
        self.grab_set()
 
        # Header
        header = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=0, height=64)
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(header, text="SETTINGS", font=("Courier New", 18, "bold"),
                      text_color="#00e676").pack(side="left", padx=24, pady=20)
        ctk.CTkLabel(header, text=f"v{APP_VERSION}", font=("Courier New", 11),
                      text_color="#555").pack(side="right", padx=24)
 
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent", scrollbar_button_color="#2a2a2a",
                                         scrollbar_button_hover_color="#00e676")
        scroll.pack(fill="both", expand=True, padx=16, pady=(12, 0))
 
        def section(parent, title):
            f = ctk.CTkFrame(parent, fg_color="#1c1c1c", corner_radius=10,
                              border_width=1, border_color="#2a2a2a")
            f.pack(fill="x", pady=(0, 12))
            ctk.CTkLabel(f, text=title, font=("Courier New", 11, "bold"),
                          text_color="#00e676").pack(anchor="w", padx=16, pady=(12, 6))
            sep = ctk.CTkFrame(f, height=1, fg_color="#2a2a2a")
            sep.pack(fill="x", padx=16, pady=(0, 10))
            return f
 
        def toggle_row(parent, text, hint, var):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=4)
            left = ctk.CTkFrame(row, fg_color="transparent")
            left.pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(left, text=text, font=("Segoe UI", 13), text_color="#e0e0e0",
                          anchor="w").pack(anchor="w")
            if hint:
                ctk.CTkLabel(left, text=hint, font=("Segoe UI", 10), text_color="#555",
                              anchor="w").pack(anchor="w")
            ctk.CTkSwitch(row, text="", variable=var, width=48,
                           button_color="#00e676", button_hover_color="#00c853",
                           progress_color="#00e676").pack(side="right")
 
        # General
        sec1 = section(scroll, "  GENERAL")
        self.var_autostart_pc = ctk.BooleanVar(value=prefs.get("autostart_pc", False))
        toggle_row(sec1, "Auto-start with Windows", "Launch on login", self.var_autostart_pc)
        self.var_autostart_server = ctk.BooleanVar(value=prefs.get("autostart_server", False))
        toggle_row(sec1, "Auto-start server", "Start server immediately on launch", self.var_autostart_server)
        self.var_admin = ctk.BooleanVar(value=prefs.get("run_as_admin", False))
        toggle_row(sec1, "Request admin rights", "Required for some input features", self.var_admin)
        ctk.CTkFrame(sec1, height=8, fg_color="transparent").pack()
 
        # Gaming
        sec2 = section(scroll, "  INPUT & GAMING")
        self.var_gaming_mode = ctk.BooleanVar(value=prefs.get("gaming_mode", True))
        toggle_row(sec2, "Gaming mode", "Micro-delays so games register input", self.var_gaming_mode)
        ctk.CTkFrame(sec2, height=8, fg_color="transparent").pack()
 
        # Network
        sec3 = section(scroll, "  NETWORK")
        net_row = ctk.CTkFrame(sec3, fg_color="transparent")
        net_row.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkLabel(net_row, text="WebSocket port", font=("Segoe UI", 13), text_color="#e0e0e0").pack(side="left")
        self.ent_port = ctk.CTkEntry(net_row, width=90, font=("Courier New", 13),
                                      fg_color="#111", border_color="#2a2a2a",
                                      text_color="#00e676", justify="center")
        self.ent_port.insert(0, str(prefs.get("port", 8080)))
        self.ent_port.pack(side="right")
 
        # Save
        ctk.CTkButton(self, text="SAVE CHANGES", font=("Courier New", 13, "bold"),
                        fg_color="#00e676", text_color="#000", hover_color="#00c853",
                        height=48, corner_radius=0, command=self.save_and_close).pack(
            fill="x", padx=0, pady=0, side="bottom")
 
    def save_and_close(self):
        try:
            port = int(self.ent_port.get())
            if not (1024 <= port <= 65535): raise ValueError
        except:
            messagebox.showerror("Invalid Port", "Port must be between 1024 and 65535")
            return
        self.callback_save({
            "autostart_pc": self.var_autostart_pc.get(),
            "autostart_server": self.var_autostart_server.get(),
            "run_as_admin": self.var_admin.get(),
            "gaming_mode": self.var_gaming_mode.get(),
            "port": port
        })
        self.destroy()
 
 
# ============================================================
# MAIN GUI — Fully redesigned
# ============================================================
 
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")
 
 
class ServerGUI:
    # ── colour palette ──────────────────────────────────────
    C_BG       = "#0d0d0d"
    C_SURFACE  = "#141414"
    C_CARD     = "#1a1a1a"
    C_BORDER   = "#242424"
    C_ACCENT   = "#00e676"
    C_ACCENT2  = "#00c853"
    C_ERR      = "#ff5252"
    C_WARN     = "#ffd740"
    C_DIM      = "#FFFFFF"
    C_TEXT     = "#e8e8e8"
    C_TEXT2    = "#999999"
 
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title(f"Use As Server  ·  v{APP_VERSION}")
        self.root.geometry("980x700")
        self.root.minsize(860, 620)
        self.root.configure(fg_color=self.C_BG)
 
        self.tray_icon = None
        self.prefs = {"autostart_pc": False, "autostart_server": False,
                      "run_as_admin": False, "gaming_mode": True, "port": 8080}
        self.load_preferences()
 
        try:
            icon_path = resource_path("icon.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except:
            pass
 
        self.update_queue = queue.Queue()
        self.server = None
        self.is_running = False
        self._client_count = 0
 
        self._build_root_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._set_title_bar_color()
        self.check_for_updates(manual=False)
 
        if self.prefs["autostart_server"]:
            self.root.after(1000, self.toggle_server)
 
        self.root.after(100, self.process_queue)
        self.root.after(50, self.update_preview)





    def _set_title_bar_color(self):
        """Forces the Windows title bar to match the theme (Windows 11+)."""
        if sys.platform != "win32":
            return
            
        try:
            import ctypes
            # Force the window to draw first so we can grab its ID
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            
            # Windows DWM Constants
            DWMWA_CAPTION_COLOR = 35
            DWMWA_TEXT_COLOR = 36
            
            # Colors must be in COLORREF format: 0x00bbggrr (Blue Green Red)
            # Background: #0d0d0d -> 0x000d0d0d
            # Text: #00e676 -> 0x0076e600
            bg_color = ctypes.c_int(0x000d0d0d)
            fg_color = ctypes.c_int(0x0076e600)
            
            # Apply the colors
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_CAPTION_COLOR, ctypes.byref(bg_color), ctypes.sizeof(bg_color)
            )
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_TEXT_COLOR, ctypes.byref(fg_color), ctypes.sizeof(fg_color)
            )
        except Exception as e:
            print(f"Could not set title bar color (might be an older Windows version): {e}")

    def check_for_updates(self, manual=False):
        """Spawns a background thread to check GitHub for updates."""
        import threading
        threading.Thread(target=self._fetch_latest_version, args=(manual,), daemon=True).start()

    def _fetch_latest_version(self, manual):
        import urllib.request
        import json
        import webbrowser
        import re
        from tkinter import messagebox

        url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
        
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                
                remote_tag = data.get("tag_name", "0.0").strip().lower().lstrip('v')
                current_tag = APP_VERSION.strip().lower().lstrip('v')

                # Helper to convert "1.10.1" -> (1, 10, 1)
                def parse_version(v_str):
                    parts = re.findall(r'\d+', v_str)
                    return tuple(map(int, parts)) if parts else (0,)

                try:
                    remote_tuple = parse_version(remote_tag)
                    current_tuple = parse_version(current_tag)
                    is_newer = remote_tuple > current_tuple
                except ValueError:
                    # Fallback if tags have weird characters
                    is_newer = remote_tag != current_tag

                if is_newer:
                    html_url = data.get("html_url", f"https://github.com/{GITHUB_REPO}/releases")
                    self.root.after(0, lambda: self._show_update_dialog(remote_tag, html_url))
                elif manual:
                    self.root.after(0, lambda: messagebox.showinfo("Up to Date", f"You are running the latest version (v{APP_VERSION})."))
        except Exception as e:
            if manual:
                self.root.after(0, lambda: messagebox.showerror("Update Error", f"Could not check for updates:\n{e}"))

    def _show_update_dialog(self, new_version, download_url):
        import webbrowser
        from tkinter import messagebox
        
        ans = messagebox.askyesno(
            "Update Available", 
            f"A new version (v{new_version}) is available!\n\nWould you like to open GitHub to download it?"
        )
        if ans:
            webbrowser.open(download_url)
 
    # ── preferences ────────────────────────────────────────
    def load_preferences(self):
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, 'r') as f: self.prefs.update(json.load(f))
        except: pass
 
    def save_preferences(self, new_prefs=None):
        if new_prefs: self.prefs = new_prefs
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SETTINGS_FILE, 'w') as f: json.dump(self.prefs, f, indent=4)
        except: pass
        if sys.platform == "win32":
            try:
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                     r"Software\Microsoft\Windows\CurrentVersion\Run",
                                     0, winreg.KEY_SET_VALUE)
                app_path = (sys.executable if getattr(sys, 'frozen', False)
                            else f'"{sys.executable}" "{__file__}"')
                if self.prefs["autostart_pc"]:
                    winreg.SetValueEx(key, "UseAsServer", 0, winreg.REG_SZ, app_path)
                else:
                    try: winreg.DeleteValue(key, "UseAsServer")
                    except: pass
                winreg.CloseKey(key)
            except: pass
 
    def open_settings(self):
        SettingsDialog(self.root, self.prefs, self.save_preferences)
 
    # ── tray ───────────────────────────────────────────────
    def create_default_icon(self):
        img = Image.new('RGB', (64, 64), "#00e676")
        dc = ImageDraw.Draw(img)
        dc.rectangle((16, 16, 48, 48), fill="#141414")
        return img
 
    def minimize_to_tray(self):
        self.root.withdraw()
        
        # --- NEW CODE ---
        icon_path = resource_path("icon.ico")
        if os.path.exists(icon_path):
            image = Image.open(icon_path)
        else:
            image = self.create_default_icon()
            
        menu = (item('Restore', self.restore_from_tray, default=True),
                item('Stop & Quit', self.quit_app))
        self.tray_icon = pystray.Icon("UseAsServer", image, "Use As Server", menu)
        self.tray_icon.run_detached()
 
    def restore_from_tray(self, icon=None, item=None):
        if self.tray_icon: self.tray_icon.stop(); self.tray_icon = None
        self.root.after(0, self.root.deiconify)
 
    def quit_app(self, icon=None, item=None):
        if self.tray_icon: self.tray_icon.stop()
        self.root.after(0, self.on_closing)
 
    # ── root layout ────────────────────────────────────────
    def _build_root_layout(self):
        self.root.grid_columnconfigure(0, weight=0)   # sidebar (fixed)
        self.root.grid_columnconfigure(1, weight=1)   # content
        self.root.grid_rowconfigure(0, weight=1)
 
        self._build_sidebar()
        self._build_content_area()
        self._nav_go("dashboard")
 
    # ── sidebar ────────────────────────────────────────────
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self.root, width=200, fg_color=self.C_SURFACE,
                           corner_radius=0, border_width=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        sb.grid_rowconfigure(10, weight=1)
 
        # Logo block
        logo_f = ctk.CTkFrame(sb, fg_color="#0d0d0d", corner_radius=0, height=80)
        logo_f.pack(fill="x")
        logo_f.pack_propagate(False)
        ctk.CTkLabel(logo_f, text="USE AS", font=("Courier New", 15, "bold"),
                      text_color=self.C_DIM, anchor="w").place(x=20, y=12)
                      
        
        ctk.CTkLabel(logo_f, text="SERVER", font=("Courier New", 22, "bold"),
                      text_color=self.C_ACCENT, anchor="w").place(x=16, y=38)
 
        # Thin accent line
        ctk.CTkFrame(sb, height=2, fg_color=self.C_ACCENT, corner_radius=0).pack(fill="x")
 
        # Nav buttons
        self._active_nav = "dashboard"
        self._nav_buttons = {}
 
        nav_items = [
            ("dashboard", "⬛  Dashboard"),
            ("camera",    "◉  Camera Studio"),
            ("sharing",   "⇆  Sharing"),
        ]
 
        nav_container = ctk.CTkFrame(sb, fg_color="transparent")
        nav_container.pack(expand=True, fill="x")

        # 👇 2. UPDATE THE LOOP TO PACK INTO 'nav_container' 👇
        for key, label in nav_items:
            btn = ctk.CTkButton(
                nav_container, text=label, anchor="w", 
                font=("Segoe UI", 15),
                fg_color="transparent",
                text_color=self.C_TEXT2,
                hover_color="#1e1e1e",
                corner_radius=6, height=40,
                command=lambda k=key: self._nav_go(k)
            )
            btn.pack(fill="x", padx=10, pady=2)
            self._nav_buttons[key] = btn
 
          # set initial active
 
        # Bottom area
        bottom = ctk.CTkFrame(sb, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=10, pady=16)
 
        # Connection badge
        self.lbl_clients = ctk.CTkLabel(
            bottom, text="● 0 connected",
            font=("Courier New", 15), text_color=self.C_DIM)
        self.lbl_clients.pack(anchor="w", pady=(0, 8))
 
        # Settings Button
        ctk.CTkButton(bottom, text="⚙  Settings", height=42,
                       font=("Segoe UI", 14, "bold"), 
                       fg_color="#000000", hover_color="#0a2914", text_color=self.C_ACCENT,
                       border_width=1.5, border_color=self.C_ACCENT,
                       corner_radius=8, command=self.open_settings).pack(fill="x", pady=(0, 8))
        
        ctk.CTkButton(bottom, text="🔄  Check for Update", height=42,
                       font=("Segoe UI", 14, "bold"), 
                       fg_color="#000000", hover_color="#0a2914", text_color=self.C_ACCENT,
                       border_width=1.5, border_color=self.C_ACCENT,
                       corner_radius=8, command=lambda: self.check_for_updates(manual=True)).pack(fill="x", pady=(0, 8))

        # Tray Button
        ctk.CTkButton(bottom, text="⊡  Tray", height=42,
                       font=("Segoe UI", 14, "bold"), 
                       fg_color="#000000", hover_color="#0a2914", text_color=self.C_ACCENT,
                       border_width=1.5, border_color=self.C_ACCENT,
                       corner_radius=8, command=self.minimize_to_tray).pack(fill="x")
 
    def _nav_go(self, key):
        self._active_nav = key
        for k, btn in self._nav_buttons.items():
            if k == key:
                btn.configure(fg_color="#1f2e24", text_color=self.C_ACCENT)
            else:
                btn.configure(fg_color="transparent", text_color=self.C_TEXT2)
        # Show matching frame
        for k, frame in self._pages.items():
            if k == key:
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                frame.grid_forget()
 
    # ── content area ───────────────────────────────────────
    def _build_content_area(self):
        self._content = ctk.CTkFrame(self.root, fg_color=self.C_BG, corner_radius=0)
        self._content.grid(row=0, column=1, sticky="nsew")
        self._content.grid_columnconfigure(0, weight=1)
        self._content.grid_rowconfigure(0, weight=1)
 
        self._pages = {}
        for key, builder in [
            ("dashboard", self._build_dashboard),
            ("camera",    self._build_camera),
            ("sharing",   self._build_sharing),
        ]:
            page = ctk.CTkFrame(self._content, fg_color=self.C_BG, corner_radius=0)
            page.grid_columnconfigure(0, weight=1)
            page.grid_rowconfigure(0, weight=1)
            builder(page)
            self._pages[key] = page
 
    # ── helpers ────────────────────────────────────────────
    def _card(self, parent, **kw):
        return ctk.CTkFrame(parent, fg_color=self.C_CARD, corner_radius=10,
                             border_width=1, border_color=self.C_BORDER, **kw)
 
    def _section_label(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=("Courier New", 10, "bold"),
                      text_color=self.C_DIM).pack(anchor="w", padx=4, pady=(0, 4))
 
    def _pill_btn(self, parent, text, command, accent=False, danger=False, **kw):
        # Default style matches the new Feature Chips (Black with Green Border)
        fg = "#000000"
        hov = "#0a2914"
        tc = self.C_ACCENT
        bw = 1.5
        bc = self.C_ACCENT

        if danger:
            fg, hov, tc = self.C_ERR, "#cc3333", "#ffffff"
            bw = 0
        elif accent:
            # Accent buttons become solid green with black text
            fg, hov, tc = self.C_ACCENT, self.C_ACCENT2, "#000000"
            bw = 0

        return ctk.CTkButton(parent, text=text, command=command,
                              fg_color=fg, hover_color=hov, text_color=tc,
                              font=("Segoe UI", 14, "bold"), corner_radius=8, height=42,
                              border_width=bw, border_color=bc, **kw)
 
    # ── Dashboard ──────────────────────────────────────────
    def _build_dashboard(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color=self.C_BG,
                                          scrollbar_button_color="#1e1e1e",
                                          scrollbar_button_hover_color=self.C_ACCENT)
        scroll.grid(row=0, column=0, sticky="nsew", padx=24, pady=20)
        scroll.grid_columnconfigure(0, weight=1)
 
        # Page title
        ctk.CTkLabel(scroll, text="DASHBOARD",
                      font=("Courier New", 13, "bold"), text_color=self.C_DIM).pack(anchor="w")
        ctk.CTkFrame(scroll, height=1, fg_color=self.C_BORDER).pack(fill="x", pady=(4, 16))
 
        # ─ Server control card ─────────────────────────────
        ctrl = self._card(scroll)
        ctrl.pack(fill="x", pady=(0, 16))
 
        top_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        top_row.pack(fill="x", padx=20, pady=(20, 12))
 
        # Status indicator
        ind_f = ctk.CTkFrame(top_row, fg_color="transparent")
        ind_f.pack(side="left")
        self._status_dot = ctk.CTkLabel(ind_f, text="●",
                                          font=("Segoe UI", 28), text_color=self.C_ERR)
        self._status_dot.pack(side="left")
        status_labels = ctk.CTkFrame(ind_f, fg_color="transparent")
        status_labels.pack(side="left", padx=12)
        self.lbl_status = ctk.CTkLabel(status_labels, text="OFFLINE",
                                         font=("Courier New", 18, "bold"), text_color=self.C_ERR)
        self.lbl_status.pack(anchor="w")
        self.lbl_status_sub = ctk.CTkLabel(status_labels, text="Server not running",
                                             font=("Segoe UI", 14), text_color=self.C_DIM)
        self.lbl_status_sub.pack(anchor="w")
 
        self.btn_start = ctk.CTkButton(
            top_row, text="START", width=110, height=44,
            font=("Courier New", 16, "bold"),
            fg_color=self.C_ACCENT, text_color="#000", hover_color=self.C_ACCENT2,
            corner_radius=6, command=self.toggle_server)
        self.btn_start.pack(side="right")
 
        # IP display
        ip_f = ctk.CTkFrame(ctrl, fg_color="#111111", corner_radius=6)
        ip_f.pack(fill="x", padx=20, pady=(0, 20))
        ctk.CTkLabel(ip_f, text="ENDPOINT", font=("Courier New", 9, "bold"),
                      text_color=self.C_DIM).pack(anchor="w", padx=12, pady=(8, 2))
        self.ent_ip = ctk.CTkEntry(ip_f, font=("Courier New", 14),
                                    fg_color="transparent", border_width=0,
                                    text_color=self.C_ACCENT, state="readonly",
                                    placeholder_text="ws://—.—.—.—:——", height=34)
        self.ent_ip.pack(fill="x", padx=8, pady=(0, 10))
 
        # ─ Quick features row ──────────────────────────────
        self._section_label(scroll, "FEATURES")
        feat = self._card(scroll)
        feat.pack(fill="x", pady=(0, 16))
 
        feat_inner = ctk.CTkFrame(feat, fg_color="transparent")
        feat_inner.pack(fill="x", padx=16, pady=14)
 
        self.btn_obs = self._feature_chip(feat_inner, "OBS Camera", "📹",
                                           lambda: self.toggle_camera("OBS Virtual Camera"))
        self.btn_obs.pack(side="left", padx=(0, 8))
        self.btn_obs.configure(state="disabled")
 
        self.btn_unity = self._feature_chip(feat_inner, "Unity Camera", "🎥",
                                             lambda: self.toggle_camera("Unity Video Capture"))
        self.btn_unity.pack(side="left", padx=(0, 8))
        self.btn_unity.configure(state="disabled")
 
        self.btn_audio = self._feature_chip(feat_inner, "Audio Stream", "🔊",
                                             self.toggle_audio)
        self.btn_audio.pack(side="left")
        self.btn_audio.configure(state="disabled")
 
        # ─ Activity log ────────────────────────────────────
        log_header = ctk.CTkFrame(scroll, fg_color="transparent")
        log_header.pack(fill="x", pady=(10, 4))
        
        # 2. Pack the section label to the left
        ctk.CTkLabel(log_header, text="ACTIVITY LOG", 
                     font=("Courier New", 10, "bold"), 
                     text_color=self.C_DIM).pack(side="left", padx=4)

        # 3. Add a slider to dynamically adjust the log box height
        self.log_height_slider = ctk.CTkSlider(
            log_header, from_=100, to=800, width=120, height=14,
            button_color=self.C_ACCENT, button_hover_color=self.C_ACCENT2,
            progress_color=self.C_ACCENT,
            command=lambda val: self.log_box.configure(height=int(val))
        )
        self.log_height_slider.set(220) # Set default height
        self.log_height_slider.pack(side="right", padx=10)
        
        ctk.CTkLabel(log_header, text="Window Height:", 
                     font=("Segoe UI", 10), text_color=self.C_DIM).pack(side="right")

        # 4. Create the card and textbox (Now with green text!)
        log_card = self._card(scroll)
        log_card.pack(fill="both", expand=True, pady=(0, 4))

        self.log_box = ctk.CTkTextbox(
            log_card, font=("Courier New", 14),
            fg_color="#0d0d0d", 
            text_color=self.C_ACCENT, # 🟢 THIS MAKES THE TEXT GREEN
            scrollbar_button_color="#1e1e1e",
            scrollbar_button_hover_color=self.C_ACCENT,
            height=220, corner_radius=8)
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)
 
    def _feature_chip(self, parent, label, icon, command):
        """Updated toggle chip for feature shortcuts."""
        return ctk.CTkButton(
            parent, text=f"{icon}  {label}", height=42,
            font=("Segoe UI", 14, "bold"), corner_radius=8,
            fg_color="#000000", hover_color="#0a2914",
            text_color=self.C_ACCENT, border_width=1.5,
            border_color=self.C_ACCENT, command=command)
 
    # ── Camera Studio ──────────────────────────────────────
    def _build_camera(self, parent):
        parent.grid_columnconfigure(0, weight=0)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)
 
        # Left panel
        left = ctk.CTkScrollableFrame(parent, width=220, fg_color=self.C_SURFACE,
                                       corner_radius=0, border_width=0,
                                       scrollbar_button_color="#1e1e1e",
                                       scrollbar_button_hover_color=self.C_ACCENT)
        left.grid(row=0, column=0, sticky="nsew")
 
        # Page title in panel
        ctk.CTkFrame(left, height=2, fg_color=self.C_BORDER).pack(fill="x")
        ctk.CTkLabel(left, text="CAMERA STUDIO",
                      font=("Courier New", 10, "bold"), text_color=self.C_DIM).pack(
            anchor="w", padx=16, pady=(14, 10))
 
        # Background section
        bg_card = self._card(left)
        bg_card.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(bg_card, text="BACKGROUND", font=("Courier New", 9, "bold"),
                      text_color=self.C_DIM).pack(anchor="w", padx=14, pady=(10, 6))
 
        self.var_bg = ctk.StringVar(value="none")
        for val, lbl in [("none", "None"), ("blur", "Blur"), ("image", "Custom Image")]:
            ctk.CTkRadioButton(
                bg_card, text=lbl, variable=self.var_bg, value=val,
                font=("Segoe UI", 12), text_color=self.C_TEXT,
                fg_color=self.C_ACCENT, hover_color=self.C_ACCENT2,
                command=self.update_cam_settings).pack(anchor="w", padx=16, pady=3)
 
        ctk.CTkButton(bg_card, text="Choose image…", height=30,
                       font=("Segoe UI", 11), fg_color="#111",
                       hover_color="#1e1e1e", text_color=self.C_TEXT2,
                       border_width=1, border_color=self.C_BORDER,
                       command=self.select_bg_image).pack(
            fill="x", padx=14, pady=(6, 12))
 
        # Adjustments section
        adj_card = self._card(left)
        adj_card.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(adj_card, text="ADJUSTMENTS", font=("Courier New", 9, "bold"),
                      text_color=self.C_DIM).pack(anchor="w", padx=14, pady=(10, 6))
 
        self.var_mirror = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(adj_card, text="Mirror", variable=self.var_mirror,
                       font=("Segoe UI", 12), text_color=self.C_TEXT,
                       button_color=self.C_ACCENT, button_hover_color=self.C_ACCENT2,
                       progress_color=self.C_ACCENT,
                       command=self.update_cam_settings).pack(anchor="w", padx=14, pady=4)
        self.var_flip = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(adj_card, text="Flip Vertical", variable=self.var_flip,
                       font=("Segoe UI", 12), text_color=self.C_TEXT,
                       button_color=self.C_ACCENT, button_hover_color=self.C_ACCENT2,
                       progress_color=self.C_ACCENT,
                       command=self.update_cam_settings).pack(anchor="w", padx=14, pady=4)
 
        ctk.CTkLabel(adj_card, text="Brightness", font=("Segoe UI", 11),
                      text_color=self.C_TEXT2).pack(anchor="w", padx=14, pady=(6, 2))
        self.scale_bright = ctk.CTkSlider(adj_card, from_=-100, to=100,
                                           button_color=self.C_ACCENT,
                                           button_hover_color=self.C_ACCENT2,
                                           progress_color=self.C_ACCENT,
                                           command=lambda x: self.update_cam_settings())
        self.scale_bright.set(0)
        self.scale_bright.pack(fill="x", padx=14, pady=(0, 12))
 
        # Resolution
        res_card = self._card(left)
        res_card.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(res_card, text="RESOLUTION", font=("Courier New", 9, "bold"),
                      text_color=self.C_DIM).pack(anchor="w", padx=14, pady=(10, 6))
        self.combo_res = ctk.CTkOptionMenu(
            res_card,
            values=["1280x720 (16:9)", "1920x1080 (16:9)", "800x600 (4:3)", "720x720 (1:1)"],
            font=("Segoe UI", 12), fg_color="#111",
            button_color="#1e1e1e", button_hover_color="#252525",
            dropdown_fg_color="#141414", text_color=self.C_TEXT,
            command=self.change_aspect_ratio)
        self.combo_res.pack(fill="x", padx=14, pady=(0, 12))
 
        # Output control
        out_card = self._card(left)
        out_card.pack(fill="x", padx=10, pady=(0, 12))
        ctk.CTkLabel(out_card, text="OUTPUT", font=("Courier New", 9, "bold"),
                      text_color=self.C_DIM).pack(anchor="w", padx=14, pady=(10, 6))
 
        self.btn_cs_obs = ctk.CTkButton(
            out_card, text="▶  OBS Virtual Camera", height=36,
            font=("Segoe UI", 12), fg_color="#1a2a1f",
            hover_color="#1f3325", text_color=self.C_ACCENT,
            border_width=1, border_color="#2a3d2d",
            command=lambda: self.toggle_camera("OBS Virtual Camera"))
        self.btn_cs_obs.pack(fill="x", padx=14, pady=(0, 6))
 
        self.btn_cs_unity = ctk.CTkButton(
            out_card, text="▶  Unity Video Capture", height=36,
            font=("Segoe UI", 12), fg_color="#1a2a1f",
            hover_color="#1f3325", text_color=self.C_ACCENT,
            border_width=1, border_color="#2a3d2d",
            command=lambda: self.toggle_camera("Unity Video Capture"))
        self.btn_cs_unity.pack(fill="x", padx=14, pady=(0, 12))
 
        # Right — preview
        right = ctk.CTkFrame(parent, fg_color="#080808", corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_propagate(False)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)
 
        self.lbl_preview = ctk.CTkLabel(
            right, text="No signal\nWaiting for connection…",
            font=("Courier New", 13), text_color="#333",
            justify="center")
        # Pack it so it physically expands to fill the entire black frame
        self.lbl_preview.pack(fill="both", expand=True)
 
       
        badge = ctk.CTkFrame(right, fg_color="#0d0d0d", corner_radius=6, width=120, height=28)
        badge.place(x=12, y=12)
        self._preview_badge = ctk.CTkLabel(badge, text="● LIVE", font=("Courier New", 10, "bold"),
                                            text_color="#555")
        self._preview_badge.pack(padx=10, pady=4)
 
    # ── Sharing ────────────────────────────────────────────
    def _build_sharing(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color=self.C_BG,
                                          scrollbar_button_color="#1e1e1e",
                                          scrollbar_button_hover_color=self.C_ACCENT)
        scroll.grid(row=0, column=0, sticky="nsew", padx=24, pady=20)
        scroll.grid_columnconfigure(0, weight=1)
 
        ctk.CTkLabel(scroll, text="SHARING",
                      font=("Courier New", 13, "bold"), text_color=self.C_DIM).pack(anchor="w")
        ctk.CTkFrame(scroll, height=1, fg_color=self.C_BORDER).pack(fill="x", pady=(4, 16))
 
        # ─ Clipboard ───────────────────────────────────────
        self._section_label(scroll, "CLIPBOARD SYNC")
        clip_card = self._card(scroll)
        clip_card.pack(fill="x", pady=(0, 16))
 
        self.txt_clip = ctk.CTkTextbox(
            clip_card, height=90, font=("Courier New", 12),
            fg_color="#0d0d0d", text_color=self.C_ACCENT,
            scrollbar_button_color="#1e1e1e", corner_radius=6)
        self.txt_clip.pack(fill="x", padx=14, pady=(14, 8))
 
        btn_row = ctk.CTkFrame(clip_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=14, pady=(0, 14))
        self._pill_btn(btn_row, "Copy to PC", self.copy_to_pc).pack(side="left", expand=True, fill="x", padx=(0, 6))
        self._pill_btn(btn_row, "Send to Phone →", self.send_text_to_phone,
                       accent=True).pack(side="right", expand=True, fill="x")
 
        # ─ File transfer ───────────────────────────────────
        self._section_label(scroll, "FILE TRANSFER")
        file_card = self._card(scroll)
        file_card.pack(fill="both", expand=True, pady=(0, 4))
 
        self.list_files = ctk.CTkTextbox(
            file_card, font=("Courier New", 11),
            fg_color="#0d0d0d", text_color="#777",
            scrollbar_button_color="#1e1e1e",
            scrollbar_button_hover_color=self.C_ACCENT,
            height=180, corner_radius=6)
        self.list_files.pack(fill="both", expand=True, padx=14, pady=14)
 
        btn_row2 = ctk.CTkFrame(file_card, fg_color="transparent")
        btn_row2.pack(fill="x", padx=14, pady=(0, 14))
        self._pill_btn(btn_row2, "Open Folder", self.open_folder).pack(
            side="left", expand=True, fill="x", padx=(0, 6))
        self._pill_btn(btn_row2, "Send File…", self.send_file_pick,
                       accent=True).pack(side="right", expand=True, fill="x")
 
    # ── server logic ───────────────────────────────────────
    def toggle_server(self):
        if not self.is_running:
            self.btn_start.configure(text="STOP", fg_color=self.C_ERR, hover_color="#cc3333",
                                      text_color="#fff")
            self.lbl_status.configure(text="STARTING…", text_color=self.C_WARN)
            self._status_dot.configure(text_color=self.C_WARN)
            self.lbl_status_sub.configure(text="Initialising server…")
            threading.Thread(target=self.start_sequence, daemon=True).start()
        else:
            self.stop_server()
 
    def stop_server(self):
        if self.server: self.server.stop()
        self.is_running = False
        self.btn_start.configure(text="START", fg_color=self.C_ACCENT,
                                  hover_color=self.C_ACCENT2, text_color="#000")
        self.lbl_status.configure(text="OFFLINE", text_color=self.C_ERR)
        self._status_dot.configure(text_color=self.C_ERR)
        self.lbl_status_sub.configure(text="Server not running")
        self._set_ip("")
        for btn in [self.btn_obs, self.btn_unity, self.btn_audio]:
            btn.configure(state="disabled", fg_color=self.C_CARD, text_color=self.C_TEXT2)
        self._update_client_badge(0)
 
    def _set_ip(self, val):
        self.ent_ip.configure(state="normal")
        self.ent_ip.delete(0, "end")
        if val:
            self.ent_ip.insert(0, f"ws://{val}")
        self.ent_ip.configure(state="readonly")
 
    def start_sequence(self):
        try:
            if self.server: self.server.stop()
            port = self.prefs.get("port", 8080)
            gaming_mode = self.prefs.get("gaming_mode", True)
            self.server = UnifiedRemoteServer(port=port, gaming_mode=gaming_mode, update_queue=self.update_queue)
            self.server.start()
            for _ in range(20):
                if self.server._loop and self.server._loop.is_running():
                    ip = self.server.get_local_ip()
                    self.update_queue.put(("server_ready", f"{ip}:{port}"))
                    return
                time.sleep(0.1)
            raise Exception("Timeout waiting for server loop")
        except Exception as e:
            self.update_queue.put(("error", str(e)))
            self.stop_server()
 
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
            self._preview_badge.configure(text_color=self.C_ACCENT)
            if active_device == "OBS Virtual Camera":
                self.btn_cs_obs.configure(text="⏹  OBS Camera", fg_color="#2a1a1a",
                                           border_color="#5a2a2a", text_color=self.C_ERR)
                self.btn_cs_unity.configure(state="disabled", text_color=self.C_DIM)
                self.btn_obs.configure(text="⏹ OBS Camera", fg_color="#2a1a1a", text_color=self.C_ERR)
                self.btn_unity.configure(state="disabled")
            else:
                self.btn_cs_unity.configure(text="⏹  Unity Camera", fg_color="#2a1a1a",
                                             border_color="#5a2a2a", text_color=self.C_ERR)
                self.btn_cs_obs.configure(state="disabled", text_color=self.C_DIM)
                self.btn_unity.configure(text="⏹ Unity Camera", fg_color="#2a1a1a", text_color=self.C_ERR)
                self.btn_obs.configure(state="disabled")
        else:
            self._preview_badge.configure(text_color="#333")
            for b in [self.btn_cs_obs, self.btn_cs_unity]:
                b.configure(state="normal", fg_color="#1a2a1f",
                             border_color="#2a3d2d", text_color=self.C_ACCENT)
            self.btn_cs_obs.configure(text="▶  OBS Virtual Camera")
            self.btn_cs_unity.configure(text="▶  Unity Video Capture")
            self.btn_obs.configure(text="📹  OBS Camera", fg_color=self.C_CARD,
                                    text_color=self.C_TEXT2, state="normal")
            self.btn_unity.configure(text="🎥  Unity Camera", fg_color=self.C_CARD,
                                      text_color=self.C_TEXT2, state="normal")
 
    def toggle_audio(self):
        if not self.server: return
        if not self.server._streaming_audio:
            self.server.start_audio_streaming()
            self.btn_audio.configure(text="⏹ Audio Stream", fg_color="#2a1a1a", text_color=self.C_ERR)
        else:
            self.server.stop_audio_streaming()
            self.btn_audio.configure(text="🔊  Audio Stream", fg_color=self.C_CARD, text_color=self.C_TEXT2)
 
    def send_text_to_phone(self):
        text = self.txt_clip.get("1.0", "end").strip()
        if text and self.server: self.server.send_to_android("clipboard_text", text)
 
    def copy_to_pc(self):
        pyperclip.copy(self.txt_clip.get("1.0", "end").strip())
 
    def send_file_pick(self):
        paths = filedialog.askopenfilenames()
        if paths and self.server:
            threading.Thread(target=self.process_multiple_files, args=(paths,), daemon=True).start()
 
    def process_multiple_files(self, paths):
        for p in paths:
            self.server.send_file_to_phone_thread(p)
            time.sleep(0.5)
 
    def open_folder(self):
        os.startfile(SAVE_DIR)
 
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
 
    def change_aspect_ratio(self, selection):
        if not self.server: return
        try:
            w, h = map(int, selection.split(" ")[0].split("x"))
            self.server.target_w = w
            self.server.target_h = h
            if self.server._vcam_running:
                dev = self.server._vcam.device
                self.server.stop_virtual_camera()
                self.root.after(500, lambda: self.server.start_virtual_camera(dev))
        except: pass
 
    def update_preview(self):
        if self.server and self.server.latest_preview_frame is not None:
            try:
                frame = self.server.latest_preview_frame
                
                # Ensure the window is actually visible to prevent math errors
                pw = self.lbl_preview.winfo_width()
                ph = self.lbl_preview.winfo_height()
                
                if pw > 10 and ph > 10:
                    ih, iw = frame.shape[:2]
                    scale = min(pw / iw, ph / ih)
                    nw, nh = int(iw * scale), int(ih * scale)
                    
                    if nw > 0 and nh > 0:
                        img = Image.fromarray(frame)
                        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
                        
                        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(nw, nh))
                        self.lbl_preview.configure(image=ctk_img, text="")
                        
                        # 🟢 CRITICAL: Prevent Python's garbage collector from deleting the frame
                        self.lbl_preview.image = ctk_img 
            except Exception as e:
                print(f"Preview Drawing Error: {e}")
                
        self.root.after(33, self.update_preview)
 
    def _update_client_badge(self, count):
        self._client_count = count
        if count == 0:
            self.lbl_clients.configure(text="● 0 connected", text_color=self.C_DIM)
        else:
            self.lbl_clients.configure(text=f"● {count} connected", text_color=self.C_ACCENT)
 
    def process_queue(self):
        try:
            while True:
                kind, data = self.update_queue.get_nowait()
 
                if kind == "log":
                    self.log(data)
 
                elif kind == "server_ready":
                    self.is_running = True
                    self.lbl_status.configure(text="ONLINE", text_color=self.C_ACCENT)
                    self._status_dot.configure(text_color=self.C_ACCENT)
                    self.lbl_status_sub.configure(text="Accepting connections")
                    self._set_ip(data)
                    for btn in [self.btn_obs, self.btn_unity, self.btn_audio]:
                        btn.configure(state="normal")
 
                elif kind == "client_count":
                    self._update_client_badge(data)
 
                elif kind == "error":
                    messagebox.showerror("Server Error", data)
                    self.stop_server()
 
                elif kind == "clipboard":
                    self.txt_clip.delete("1.0", "end")
                    self.txt_clip.insert("end", data)
                    self._nav_go("sharing")
 
                elif kind == "file_received":
                    self.list_files.insert("end", f"📄 {os.path.basename(data)}\n")
                    self.list_files.see("end")
 
                elif kind == "audio_status":
                    if data:
                        self.btn_audio.configure(text="⏹ Audio Stream",
                                                  fg_color="#2a1a1a", text_color=self.C_ERR)
                    else:
                        self.btn_audio.configure(text="🔊  Audio Stream",
                                                  fg_color=self.C_CARD, text_color=self.C_TEXT2)
 
                elif kind == "camera_status":
                    self._update_cam_buttons(data)
 
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Queue Error: {e}")
 
        self.root.after(150, self.process_queue)
 
    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{ts}]  {msg}\n")
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


    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

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

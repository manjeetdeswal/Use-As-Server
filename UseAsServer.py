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

if sys.platform == "win32":
    import win32api
    import ctypes
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
        self.port = port
        self.running = True

    def get_broadcast_addresses(self):
        """Find broadcast address for every interface."""
        addresses = set()
        try:
            # Always try the generic broadcast
            addresses.add('<broadcast>')

            # Get all local IPs
            hostname = socket.gethostname()
            local_ips = socket.gethostbyname_ex(hostname)[2]

            for ip in local_ips:
                # Calculate broadcast for this IP (assuming standard /24 subnet)
                # e.g., 192.168.1.5 -> 192.168.1.255
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

        message = f"UNIFIED_REMOTE_SERVER:{self.port}".encode('utf-8')
        logging.info(f"Starting discovery broadcast on port 8888...")

        while self.running:
            targets = self.get_broadcast_addresses()
            for target in targets:
                try:
                    sock.sendto(message, (target, 8888))
                except:
                    pass  # Ignore errors on disconnected interfaces

            time.sleep(1)  # Announce every second

        sock.close()

    def stop(self):
        self.running = False


class UnifiedRemoteServer:
    def __init__(self, host="0.0.0.0", port=8080, update_queue: queue.Queue = None):
        import threading
        self.sending_lock = threading.Lock()
        self.host = host
        self.ack_event = threading.Event()
        self.port = port
        self.clients = set()
        self._loop = None
        self._ws_server = None
        self._thread = None
        self.update_queue = update_queue or queue.Queue()
        self._stop_event = threading.Event()
        self.discovery = DiscoveryServer(port)
        self.discovery.start()

        # START UDP MOUSE SERVER
        self.udp_mouse = UDPMouseServer(port=8081)
        self.udp_mouse.daemon = True  # Kill when app closes
        self.udp_mouse.start()

        # Thread-safe message queue for broadcasting
        self._broadcast_queue = asyncio.Queue()

        # Virtual camera
        self._vcam = None
        self._vcam_running = False
        self._last_frame = None
        self._frame_count = 0

        # Audio streaming
        self._audio_stream = None
        self._audio_thread = None
        self._streaming_audio = False
        self._display_active = False
        # Initialize this
        self._gamepad_active = False
        self.client_gamepads = {}

        # Display settings
        self._display_width = 1920
        self._display_height = 1080

        self._codec = None

        try:
            user32 = ctypes.windll.user32
            # SM_XVIRTUALSCREEN = 76, SM_YVIRTUALSCREEN = 77
            # SM_CXVIRTUALSCREEN = 78, SM_CYVIRTUALSCREEN = 79
            self.min_x = user32.GetSystemMetrics(76)
            self.min_y = user32.GetSystemMetrics(77)
            self.max_x = self.min_x + user32.GetSystemMetrics(78)
            self.max_y = self.min_y + user32.GetSystemMetrics(79)
        except:
            # Fallback to standard 1080p if fails
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
                        # FIX: Handle Binary Data (Video frames often come as bytes)
                        if isinstance(message, bytes):
                            try:
                                message = message.decode('utf-8')
                            except UnicodeDecodeError:
                                continue

                        data = json.loads(message)
                        msg_type = data.get("type", "unknown")
                        payload = data.get("payload", "")

                        if msg_type == "video_frame":
                            self._handle_video_frame(payload)
                        elif msg_type == "mouse_move":
                            self._handle_mouse_move(payload)
                        elif msg_type == "mouse_click":
                            self._handle_mouse_click(payload)
                        if msg_type == "ack":
                            self.ack_event.set()
                        elif msg_type == "clipboard_text":
                            try:
                                # Sometimes payload is double-encoded JSON
                                if isinstance(payload, str) and payload.startswith('{'):
                                    inner = json.loads(payload)
                                    text = inner.get("text", "")
                                else:
                                    text = str(payload)

                                if text:
                                    pyperclip.copy(text)
                                    self._put("clipboard", text)
                                    self._put("log", "📋 Text copied from phone")
                            except:
                                pass

                        elif msg_type == "file_transfer":
                           # print("DEBUG: Route file transfer")
                            if isinstance(payload, str):
                                payload = json.loads(payload)
                            self._handle_file_transfer(payload)

                        elif msg_type == "display_request":
                            self._handle_display_request(payload)
                        elif msg_type == "gamepad_state":
                            self._handle_gamepad_state(payload,websocket)
                        elif msg_type == "audio_frame":
                            self._handle_audio_frame(payload)
                        elif msg_type == "mouse_scroll":
                            self._handle_mouse_scroll(payload)
                        elif msg_type == "key_press":
                            self._handle_key_press(payload)
                        elif msg_type == "heartbeat":
                            await websocket.send(json.dumps({"type": "heartbeat", "payload": "pong"}))

                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        self._put("log", f"❌ Error: {e}")

            except websockets.ConnectionClosedOK:
                pass
            except Exception as e:
                self._put("log", f"⚠️ Handler error: {e}")
            finally:
                if websocket in self.clients:
                    self.clients.remove(websocket)

                    # REMOVE SPECIFIC GAMEPAD FOR THIS CLIENT
                if hasattr(self, 'client_gamepads') and websocket in self.client_gamepads:
                    try:
                        # Removing the object usually triggers vgamepad to unplug the virtual device
                        del self.client_gamepads[websocket]
                        self._put("log", f"🎮 Gamepad disconnected for {remote_addr}")
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

    def _handle_video_frame(self, payload):
        """Process video frame with STRICT rotation logic."""
        try:
            import json
            import base64
            import numpy as np
            import cv2

            # 1. Parse Payload
            if isinstance(payload, str):
                try:
                    frame_data = json.loads(payload)
                except json.JSONDecodeError:
                    frame_data = {'data': payload}
            else:
                frame_data = payload

            b64_data = frame_data.get('data')
            if not b64_data:
                return

            rotation = frame_data.get('rotation', 0)
            is_front = frame_data.get('is_front', False)

            # 2. Decode JPEG
            jpeg_bytes = base64.b64decode(b64_data)
            nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            bgr_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if bgr_frame is None:
                return

            # 3. Apply Rotation & Mirroring Logic
            # Front cameras are usually mirrored hardware-wise.
            # The logic below handles the standard Android behavior.

            if rotation == 90:
                bgr_frame = cv2.rotate(bgr_frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 270:
                bgr_frame = cv2.rotate(bgr_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotation == 180:
                bgr_frame = cv2.rotate(bgr_frame, cv2.ROTATE_180)

            # FIX: Apply mirroring AFTER rotation for front camera.
            # If it's upside down, we need to flip it vertically (0) or both (-1).
            # Standard front cam behavior after rotation usually requires a horizontal flip (1) to act like a mirror.
            # BUT if yours is upside down, let's try flipping both axes (-1) or vertical (0).

            if is_front:
                if rotation == 270 or rotation == 90:
                    # Portrait Mode Front Camera Fix
                    # Try converting the flip. If 1 (horizontal) was upside down,
                    # then 0 (vertical) or -1 (both) should fix it.
                    # Let's try -1 (Both) which is a common fix for "Upside Down & Mirrored"
                    bgr_frame = cv2.flip(bgr_frame, 0)
                else:
                    # Landscape Mode Front Camera
                    bgr_frame = cv2.flip(bgr_frame, 1)

            # 4. Convert to RGB
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            # 5. Fit into 1280x720
            TARGET_W, TARGET_H = 1280, 720
            h, w = rgb_frame.shape[:2]

            if w >= h:
                # Landscape
                rgb_frame = cv2.resize(rgb_frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
            else:
                # Portrait
                rgb_frame = self._resize_with_black_bars(rgb_frame, TARGET_W, TARGET_H)

            self._last_frame = rgb_frame
            self._frame_count += 1



        except Exception as e:
            if self._frame_count % 30 == 0:
                self._put("log", f"❌ Video Error: {e}")

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

    def start_virtual_camera(self):
        """Start virtual webcam that appears in Zoom/Teams/Discord."""
        if self._vcam_running:
            self._put("log", "📹 Virtual camera already running")
            return

        try:
            import pyvirtualcam
            import numpy as np
            import cv2

            # Create virtual camera (1280x720 @ 30fps)
            self._vcam = pyvirtualcam.Camera(
                width=1280,
                height=720,
                fps=30,
                fmt=pyvirtualcam.PixelFormat.RGB
            )
            self._vcam_running = True

            self._put("log", f"📹 Virtual Camera Started!")
            self._put("log", f"📱 Device: {self._vcam.device}")
            self._put("log", "💡 Select 'OBS Virtual Camera' in Zoom/Teams/Discord")
            self._put("log", "🎥 Waiting for Android camera stream...")

            # Start frame sender thread
            def send_frames():
                import time

                # Create waiting screen
                waiting_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(
                    waiting_frame,
                    "Waiting for Android Camera...",
                    (250, 350),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    waiting_frame,
                    "Start Camera on your Android device",
                    (280, 400),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (200, 200, 200),
                    1
                )

                while self._vcam_running:
                    try:
                        if self._last_frame is not None:
                            # Send Android camera frame
                            self._vcam.send(self._last_frame)
                        else:
                            # Send waiting screen
                            self._vcam.send(waiting_frame)

                        time.sleep(1 / 30)  # 30 FPS

                    except Exception as e:
                        if self._vcam_running:
                            self._put("log", f"⚠️ VCam send error: {e}")
                        break

                self._put("log", "📹 Frame sender stopped")

            self._vcam_thread = threading.Thread(target=send_frames, daemon=True)
            self._vcam_thread.start()

        except ImportError:
            self._put("log", "❌ pyvirtualcam not installed!")
            self._put("log", "💡 Run: pip install pyvirtualcam opencv-python")
        except Exception as e:
            self._put("log", f"❌ Virtual camera error: {e}")
            self._put("log", "💡 Install OBS Studio first (for virtual camera driver)")
            self._put("log", "   Download: https://obsproject.com/download")

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
                            if "CABLE Input" in name or "VB-Audio" in name or "Unified_Remote_Mic" in name:
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

                        # 3. Draw Mouse (OpenCV is faster than PIL)
                        try:
                            mx, my = pyautogui.position()
                            rel_x = mx - mon_left
                            rel_y = my - mon_top

                            # Scaling factor if monitor resolution != capture resolution
                            # (Usually MSS handles this, but raw coords need checking)
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
                        # Don't go unlimited; it floods the router buffer and increases latency.
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
        """Robust audio streaming:
           - Keeps worker alive even with 0 clients (avoids start/stop thrash)
           - Fuzzy loopback matching so Bluetooth/HDMI/USB choose correctly
           - Uses native sample rate and sends rate/channels to clients
           - Detects loopback disappearance and restarts gracefully
        """
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
        """Handle mouse click using Windows native API."""
        try:
            import ctypes
            event = json.loads(payload)
            button_id = event.get("button", 0)

            if platform.system() == "Windows":
                # Direct Windows API calls
                if button_id == 0:  # Left click
                    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
                    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP
                elif button_id == 1:  # Right click
                    ctypes.windll.user32.mouse_event(0x0008, 0, 0, 0, 0)  # MOUSEEVENTF_RIGHTDOWN
                    ctypes.windll.user32.mouse_event(0x0010, 0, 0, 0, 0)  # MOUSEEVENTF_RIGHTUP
                elif button_id == 2:  # Middle click
                    ctypes.windll.user32.mouse_event(0x0020, 0, 0, 0, 0)  # MOUSEEVENTF_MIDDLEDOWN
                    ctypes.windll.user32.mouse_event(0x0040, 0, 0, 0, 0)  # MOUSEEVENTF_MIDDLEUP
            else:
                # Fallback
                import pyautogui
                button_map = {0: "left", 1: "right", 2: "middle"}
                button = button_map.get(button_id, "left")
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
                    'win': 0x5B, 'menu': 0x5D
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
        self.sock = None  # Initialize as None

    def run(self):
        import ctypes

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # CRITICAL: Allow port reuse
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.setblocking(False)
            self.sock.bind(("0.0.0.0", self.port))
            print(f"🚀 UDP Mouse Server listening on port {self.port}")
        except Exception as e:
            print(f"❌ UDP Bind Failed: {e}")
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
                    ctypes.windll.user32.mouse_event(0x0001, dx, dy, 0, 0)
            except BlockingIOError:
                time.sleep(0.001)
            except Exception:
                pass

        # Cleanup when loop ends
        if self.sock:
            self.sock.close()

    def stop(self):
        self.running = False
        # Do NOT close socket here, let the loop finish and close it safely


class ServerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Use As Server")
        self.root.geometry("600x850")
        self.root.configure(bg=COLORS["bg"])

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
        self.root.after(100, self.process_queue)

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background=COLORS["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=COLORS["secondary"], foreground=COLORS["fg"], padding=[15, 5])
        style.map("TNotebook.Tab", background=[("selected", COLORS["accent"])], foreground=[("selected", "#000")])
        style.configure("TFrame", background=COLORS["bg"])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        self.tab_dashboard = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_dashboard, text="  Dashboard  ")
        self.build_dashboard(self.tab_dashboard)

        self.tab_sharing = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_sharing, text="  Sharing  ")
        self.build_sharing(self.tab_sharing)

    def build_dashboard(self, parent):
        header = tk.Frame(parent, bg=COLORS["bg"])
        header.pack(fill='x', pady=20)
        tk.Label(header, text="Use As Server", font=("Segoe UI", 24, "bold"), bg=COLORS["bg"], fg=COLORS["fg"]).pack()

        self.status_label = tk.Label(header, text="🔴 Offline", font=("Segoe UI", 10), bg=COLORS["secondary"],
                                     fg=COLORS["error"], padx=10, pady=5)
        self.status_label.pack(pady=10)

        self.ip_var = tk.StringVar(value="Not Running")
        tk.Entry(parent, textvariable=self.ip_var, font=("Consolas", 12), justify='center', bg=COLORS["secondary"],
                 fg=COLORS["accent"], relief='flat', state='readonly').pack(pady=5, ipadx=10, ipady=5)

        self.btn_start = tk.Button(parent, text="START SERVER", font=("Segoe UI", 12, "bold"), bg=COLORS["accent"],
                                   fg="black", activebackground=COLORS["accent_hover"], relief='flat',
                                   command=self.toggle_server)
        self.btn_start.pack(pady=15, ipadx=30, ipady=10)

        # Features
        features_frame = tk.LabelFrame(parent, text="Features", bg=COLORS["bg"], fg=COLORS["fg"],
                                       font=("Segoe UI", 10, "bold"))
        features_frame.pack(fill='x', padx=20, pady=10)

        def mk_btn(txt, cmd, color):
            return tk.Button(features_frame, text=txt, font=("Segoe UI", 10), bg=color, fg="white", relief='flat',
                             command=cmd, state='disabled')

        self.btn_vcam = mk_btn("📹 Virtual Camera", self.toggle_vcam, "#9c27b0")
        self.btn_vcam.pack(fill='x', pady=2, padx=5)

        self.btn_audio = mk_btn("🔊 Audio Streaming", self.toggle_audio, "#2196f3")
        self.btn_audio.pack(fill='x', pady=2, padx=5)

       

        log_frame = tk.LabelFrame(parent, text="Activity Log", bg=COLORS["bg"], fg=COLORS["text_dim"],
                                  font=("Consolas", 9))
        log_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=5, bg=COLORS["secondary"], fg=COLORS["fg"],
                                                  font=("Consolas", 9), relief='flat')
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)

    def build_sharing(self, parent):
        # Clipboard
        frame_clip = tk.LabelFrame(parent, text="Clipboard", bg=COLORS["bg"], fg=COLORS["fg"])
        frame_clip.pack(fill='x', padx=20, pady=10)
        self.txt_clip = tk.Text(frame_clip, height=3, bg=COLORS["secondary"], fg=COLORS["fg"], relief='flat')
        self.txt_clip.pack(fill='x', padx=10, pady=5)

        btn_frame = tk.Frame(frame_clip, bg=COLORS["bg"])
        btn_frame.pack(fill='x', padx=10, pady=5)
        tk.Button(btn_frame, text="Copy to PC", bg=COLORS["secondary"], fg=COLORS["fg"], relief='flat',
                  command=self.copy_to_pc).pack(side='left')
        tk.Button(btn_frame, text="Send Text to Phone", bg=COLORS["accent"], fg="black", relief='flat',
                  command=self.send_text_to_phone).pack(side='right')

        # Files
        frame_file = tk.LabelFrame(parent, text="File Transfer", bg=COLORS["bg"], fg=COLORS["fg"])
        frame_file.pack(fill='both', expand=True, padx=20, pady=10)

        self.list_files = tk.Listbox(frame_file, bg=COLORS["secondary"], fg=COLORS["fg"], relief='flat')
        self.list_files.pack(fill='both', expand=True, padx=10, pady=5)
        self.list_files.bind("<Double-Button-1>", self.open_file)

        btn_frame_f = tk.Frame(frame_file, bg=COLORS["bg"])
        btn_frame_f.pack(fill='x', padx=10, pady=10)

        # Button: Open Folder
        tk.Button(btn_frame_f, text="Open Received Files", bg=COLORS["secondary"], fg=COLORS["fg"], relief='flat',
                  command=self.open_folder).pack(side='left')

        # Button: Send File (THIS IS THE NEW ONE)
        tk.Button(btn_frame_f, text="Send File to Phone", bg=COLORS["accent"], fg="black", relief='flat',
                  command=self.send_file_pick).pack(side='right')

    # --- ACTIONS ---
    def toggle_server(self):
        if not self.is_running:
            self.btn_start.config(text="STOP SERVER", bg=COLORS["error"], fg="white")
            self.status_label.config(text="🟡 Starting...", fg="orange")
            threading.Thread(target=self.start_sequence, daemon=True).start()
        else:
            self.stop_server()

    def start_sequence(self):
        try:
            # Force cleanup of any lingering threads
            if self.server:
                self.server.stop()

            self.server = UnifiedRemoteServer(update_queue=self.update_queue)

            # Try to start. If ports are busy, this might log an error.
            self.server.start()

            # Wait up to 2 seconds for the server to actually be ready
            # We check if the loop is running to confirm success
            for _ in range(20):
                if self.server._loop and self.server._loop.is_running():
                    ip = self.server.get_local_ip()
                    self.update_queue.put(("server_ready", ip))
                    return
                time.sleep(0.1)

            # If we get here, it failed to start (likely port busy)
            raise Exception("Server timed out. Port 8080/8081 might be busy. Wait 5s and try again.")

        except Exception as e:
            self.update_queue.put(("error", str(e)))
            # Ensure we reset UI state
            if self.server:
                self.server.stop()

    def stop_server(self):
        if self.server: self.server.stop()
        self.is_running = False
        self.btn_start.config(text="START SERVER", bg=COLORS["accent"], fg="black")
        self.status_label.config(text="🔴 Offline", fg=COLORS["error"])
        self.ip_var.set("Not Running")
        for btn in [self.btn_vcam, self.btn_audio, self.btn_display]:
            btn.config(state='disabled', bg=COLORS["btn_disabled"])

    # --- FEATURE TOGGLES ---
    def toggle_vcam(self):
        if not self.server._vcam_running:
            self.server.start_virtual_camera()
            self.btn_vcam.config(text="⏹ Stop Camera", bg=COLORS["error"])
        else:
            self.server.stop_virtual_camera()
            self.btn_vcam.config(text="📹 Virtual Camera", bg="#9c27b0")

    def toggle_audio(self):
        if not self.server._streaming_audio:
            self.server.start_audio_streaming()
            self.btn_audio.config(text="🔇 Stop Audio", bg=COLORS["error"])
        else:
            self.server.stop_audio_streaming()
            self.btn_audio.config(text="🔊 Audio Streaming", bg="#2196f3")

    def toggle_display(self):
        self.log("Display toggle clicked")

    # --- FILE & TEXT LOGIC ---
    def send_text_to_phone(self):
        text = self.txt_clip.get("1.0", tk.END).strip()
        if text and self.server:
            self.server.send_to_android("clipboard_text", text)
            self.log("Sent text to phone")

    def send_file_pick(self):
        """Open file dialog and send selected files (Multiple supported)."""
        if not self.server or not self.is_running:
            messagebox.showwarning("Not Connected", "Please start the server first.")
            return

        # CHANGE: Use askopenfilenames (Plural) to select multiple
        file_paths = filedialog.askopenfilenames()

        if file_paths:
            # Run the loop in a background thread
            threading.Thread(
                target=self.process_multiple_files,
                args=(file_paths,),
                daemon=True
            ).start()

    def process_multiple_files(self, file_paths):
        """Queue multiple files safely."""
        total = len(file_paths)
        for i, file_path in enumerate(file_paths):
            self.log(f"--- Queueing File {i + 1}/{total} ---")

            # This function uses a Lock, so it will wait if a transfer is busy.
            # It sends File 1, finishes, releases lock, then sends File 2.
            self.server.send_file_to_phone_thread(file_path)

            # Tiny cool-down between files to let Android save final buffer
            time.sleep(0.5)

    def copy_to_pc(self):
        text = self.txt_clip.get("1.0", tk.END).strip()
        pyperclip.copy(text)
        self.log("Copied to PC Clipboard")

    def open_folder(self):
        try:
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            if platform.system() == "Windows":
                os.startfile(SAVE_DIR)
            else:
                subprocess.Popen(["xdg-open", str(SAVE_DIR)])
        except Exception as e:
            self.log(f"❌ Could not open folder: {e}")

    def open_file(self, event):
        selection = self.list_files.curselection()
        if selection:
            fname = self.list_files.get(selection[0])
            path = SAVE_DIR / fname
            try:
                if platform.system() == "Windows": os.startfile(path)
            except:
                pass

    # --- QUEUE & LOGGING ---
    def process_queue(self):
        try:
            while True:
                kind, data = self.update_queue.get_nowait()
                if kind == "log":
                    self.log(data)
                elif kind == "server_ready":
                    self.is_running = True
                    self.status_label.config(text="🟢 Online", fg=COLORS["accent"])
                    self.ip_var.set(f"ws://{data}:8080")
                    self.log(f"Server ready on {data}")
                    self.btn_vcam.config(state='normal', bg="#9c27b0")
                    self.btn_audio.config(state='normal', bg="#2196f3")
                    self.btn_display.config(state='normal', bg="#00bcd4")
                elif kind == "clipboard":
                    self.txt_clip.delete("1.0", tk.END)
                    self.txt_clip.insert("1.0", data)
                elif kind == "file_received":
                    # Update file list if file is in the SAVE_DIR
                    path = Path(data)
                    self.list_files.insert(0, path.name)
                elif kind == "error":
                    self.log(f"❌ Error: {data}")
                    messagebox.showerror("Server Error", f"Failed to start server:\n{data}")
                    # Reset UI to stopped state
                    self.is_running = False
                    self.btn_start.config(text="START SERVER", bg=COLORS["accent"], fg="black")
                    self.status_label.config(text="🔴 Offline", fg=COLORS["error"])
        except:
            pass
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
    # Fix for asyncio on Windows
    import multiprocessing

    multiprocessing.freeze_support()
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


    # Add pyautogui to requirements
    try:
        import pyautogui
        import websockets
    except ImportError:
        print("="*50)
        print("ERROR: Missing required libraries.")
        print("Please run: pip install websockets pyautogui")
        print("="*50)
        sys.exit(1)
    import threading

    for thread in threading.enumerate():
        if thread.name != "MainThread":
            print(f"⚠️ Warning: Found lingering thread {thread.name}")

    app = ServerGUI()
    app.run()

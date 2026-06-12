import customtkinter as ctk
from PIL import Image, ImageDraw
import threading
import asyncio
import json
import sys
import socket
import queue
import time
import websockets
import os
import pystray
from pystray import MenuItem as item
import pyautogui
import pyperclip
from pathlib import Path
from tkinter import filedialog, messagebox
import mss
import numpy as np
import cv2
import pyaudio
import signal
import subprocess

import base64
# --- OPTIONAL DEPENDENCIES CHECKS ---
try:
    import pyvirtualcam
    HAS_VCAM = True
except ImportError:
    HAS_VCAM = False
    print("⚠️ 'pyvirtualcam' not found. Camera features disabled.")
import numpy as np
import cv2
try:
    import pyfakewebcam
except ImportError:
    pyfakewebcam = None

try:
    import evdev
    from evdev import UInput, ecodes, AbsInfo
    HAS_EVDEV = True
except ImportError:
    HAS_EVDEV = False
    print("⚠️ 'evdev' not found. Gamepad features disabled.")

try:
    import pyaudio
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("⚠️ 'pyaudio' not found. Audio streaming disabled.")

try:
    from cvzone.SelfiSegmentationModule import SelfiSegmentation
    HAS_SEGMENTATION = True
except ImportError:
    HAS_SEGMENTATION = False


    

# Disable FailSafe for gaming
pyautogui.FAILSAFE = False

# --- CONSTANTS ---
APP_VERSION = "1.5"
GITHUB_REPO = "manjeetdeswal/Use-As-Server" 
GITHUB_URL = f"https://github.com/{GITHUB_REPO}"
SETTINGS_FILE = Path.home() / ".config" / "useas_server" / "settings.json"
SAVE_DIR = Path.home() / "Downloads" / "UseAs_Received"

# Ensure directories exist
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

# ============================================
# PART 1:  SERVER BACKEND (EXACT COPY)
# ============================================

class DiscoveryServer(threading.Thread):
    def __init__(self, port=8080):
        super().__init__()
        self.port = port
        self.running = True

    def get_broadcast_addresses(self):
        addresses = set()
        try:
            addresses.add('<broadcast>')
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            parts = local_ip.split('.')
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
        
        # This is what your Android app is listening for!
        message = f"UNIFIED_REMOTE_SERVER:{self.port}".encode('utf-8')

        while self.running:
            targets = self.get_broadcast_addresses()
            for target in targets:
                try:
                    # Broadcasting to port 8888
                    sock.sendto(message, (target, 8888))
                except:
                    pass
            time.sleep(1) # Broadcast every 1 second
            
        sock.close()

    def stop(self):
        self.running = False

class UDPMouseServer(threading.Thread):
    def __init__(self, port=8081):
        super().__init__()
        self.port = port
        self.running = True

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(("0.0.0.0", self.port))
            sock.setblocking(False)
        except: return 
        
        while self.running:
            try:
                data, _ = sock.recvfrom(1024)
                parts = data.decode('utf-8').split(',')
                if len(parts) == 2:
                    dx, dy = int(parts[0]), int(parts[1])
                    if dx != 0 or dy != 0:
                        pyautogui.moveRel(dx, dy, _pause=False)
            except BlockingIOError:
                time.sleep(0.002)
            except: pass
        sock.close()

    def stop(self):
        self.running = False

class UnifiedRemoteServer:
    def __init__(self, host="0.0.0.0", port=8080, update_queue: queue.Queue = None):
        self.sending_lock = threading.Lock()
        self.host = host
        self.port = port
        self.ack_event = threading.Event()
        self._setup_virtual_driver()
        self.clients = set()
        self._loop = None
        self._ws_server = None
        self._thread = None
        self.update_queue = update_queue or queue.Queue()
        self._stop_event = threading.Event()
        self.gamepad_history = {}
        # 1. START DISCOVERY
     

        # --- AUDIO SETUP ---
        self.p = pyaudio.PyAudio()
        self.audio_stream = None
        # Settings match Android: 16kHz, Mono, 16-bit
        self.AUDIO_FORMAT = pyaudio.paInt16
        self.AUDIO_CHANNELS = 1
        self.AUDIO_RATE = 16000

        

        self._broadcast_queue = asyncio.Queue()
        self.fake_cam = None
        self.show_preview = True
        self.cam_width = 0   # <--- ADD THIS
        self.cam_height = 0

        self.cam_settings = {
            "mirror": False,
            "flip": False,
            "brightness": 0,  
            "background": "none",
            "preview_on": True
        }
        self.segmentor = None
        if HAS_SEGMENTATION:
            try:
                self.segmentor = SelfiSegmentation()
                print("Background Blur Engine Loaded")
            except Exception as e:
                print(f"⚠️ Background Blur Disabled (MediaPipe Error): {e}")
                self.segmentor = None  # Fallback: Server starts without Blur





        self._vcam = None 
        self._vcam_running = False
        self._last_frame = None
        self._audio_stream = None
        self._audio_thread = None
        self._streaming_audio = False
        self._display_active = False
        self._gamepad_active = False
        self.client_gamepads = {} 
        self._display_width = 1920
        self._display_height = 1080
        self.mic_pipe = None

        # Screen metrics
        try:
            size = pyautogui.size()
            self.max_x, self.max_y = size.width, size.height
        except:
            self.max_x, self.max_y = 1920, 1080

    def get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except: return "127.0.0.1"

    def _put(self, kind, payload):
        try: self.update_queue.put((kind, payload))
        except: pass

    # --- CAM LOGIC ---
    def start_virtual_camera(self, device=None):
        if not HAS_VCAM:
            self._put("log", "❌ Error: 'pyvirtualcam' missing.")
            return
        if self._vcam_running: return

        try:
            # 1. FORCE DEVICE PATH
            target_device = device
            if target_device is None and os.path.exists("/dev/video10"):
                target_device = "/dev/video10"
            
            # 2. Start Camera
            self._vcam = pyvirtualcam.Camera(
                width=1280, height=720, fps=30, 
                fmt=pyvirtualcam.PixelFormat.RGB, 
                device=target_device
            )
            
            self._vcam_running = True
            self.vcam_error_shown = False # Reset error flag
            
            self._put("log", f"📹 VCam Started: {self._vcam.device}")
            self._put("log", "ℹ️ Set OBS Video Format to 'RGB'!")
            
           
            self._put("vcam_state", True) 
            
            threading.Thread(target=self._send_frames_loop, daemon=True).start()

        except Exception as e:
            self._put("log", f"❌ VCam Error: {e}")
            if "No such file" in str(e):
                self._put("log", "💡 Driver not loaded. Run: sudo modprobe v4l2loopback")
            
            self._vcam_running = False
            self._put("vcam_state", False)

    def stop_virtual_camera(self):
        """Stops the camera stream."""
        self._vcam_running = False
        time.sleep(0.1) 
        if self._vcam:
            try:
                self._vcam.close()
            except: pass
            self._vcam = None
        
        self._put("log", "📹 Virtual Camera Stopped")
        
      
        self._put("vcam_state", False)

    # --- UPDATED: DEBUG CAMERA LOGIC ---
    def _handle_video_frame(self, payload):
        """Optimized Video Handler with Auto-Start"""
        try:
            
            if not self._vcam_running and HAS_VCAM:
                # Check a flag so we don't spam start attempts if it fails
                if not getattr(self, "vcam_error_shown", False):
                    print("📷 Auto-starting Virtual Camera...")
                    self.start_virtual_camera()
                    # If it failed to start, mark it to stop trying
                    if not self._vcam_running:
                        self.vcam_error_shown = True
            # ---------------------------------------------------------

            # 1. Parse JSON
            if isinstance(payload, str): frame_data = json.loads(payload)
            else: frame_data = payload

            b64_data = frame_data.get('data')
            if not b64_data: return

            # 2. Decode Image
            img_bytes = base64.b64decode(b64_data)
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None: return

            # 3. Process Rotation/Mirroring (Standard Logic)
            rotation = frame_data.get('rotation', 0)
            if rotation == 90: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 270: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotation == 180: frame = cv2.rotate(frame, cv2.ROTATE_180)

            is_front = frame_data.get('is_front', False)
            if is_front:
                if rotation in [90, 270]: frame = cv2.flip(frame, 0)
                else: frame = cv2.flip(frame, 1)

            if self.cam_settings["mirror"]: frame = cv2.flip(frame, 1)
            if self.cam_settings["flip"]: frame = cv2.flip(frame, 0)

            # 4. Resize & Effects
            tw = getattr(self, 'target_w', 1280)
            th = getattr(self, 'target_h', 720)
            if frame.shape[1] != tw or frame.shape[0] != th:
                frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)

            if self.cam_settings["brightness"] != 0:
                frame = cv2.convertScaleAbs(frame, alpha=1, beta=self.cam_settings["brightness"])

            if self.cam_settings["background"] != "none":
                frame = self._process_background(frame)

            # 5. Send to VCam
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._last_frame = rgb_frame

            # 6. UI Preview
            if self.cam_settings["preview_on"]:
                h, w = rgb_frame.shape[:2]
                small_w = 320
                small_h = int(h * (small_w / w))
                preview_img = cv2.resize(rgb_frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
                pil_img = Image.fromarray(preview_img)
                self._put("video_frame", pil_img)

        except Exception as e:
            print(f"❌ VIDEO ERROR: {e}")

    # --- NEW: CLEANUP FUNCTION ---
    def _stop_camera(self):
        """Closes the preview window when phone disconnects"""
        try:
            cv2.destroyAllWindows() # Closes the popup window
            self._put("log", "📷 Camera Preview Closed")
        except: pass

    def _send_frames_loop(self):
        # Create a BLUE "Waiting" screen instead of black
        # This helps user confirm the camera is actually ON
        waiting_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        waiting_frame[:] = (0, 0, 100) # Dark Blue RGB
        
        # Add text "Waiting for Phone..."
        cv2.putText(waiting_frame, "UseAs: Waiting for Phone...", (350, 360), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        while self._vcam_running:
            try:
                if self._last_frame is not None:
                    self._vcam.send(self._last_frame)
                else:
                    self._vcam.send(waiting_frame)
                
                self._vcam.sleep_until_next_frame()
            except: break

    # --- NETWORKING HELPERS ---
    def send_to_android(self, msg_type, payload):
        if self._loop and self._loop.is_running() and self._broadcast_queue:
            message = json.dumps({"type": msg_type, "payload": payload})
            asyncio.run_coroutine_threadsafe(self._broadcast_queue.put(message), self._loop)

    def _send_to_clients_threadsafe(self, message: str):
        if self._loop and self._loop.is_running() and self._broadcast_queue:
            asyncio.run_coroutine_threadsafe(self._broadcast_queue.put(message), self._loop)

    async def _broadcast_worker(self):
        while True:
            try:
                if not self._broadcast_queue:
                    await asyncio.sleep(0.1)
                    continue
                message = await self._broadcast_queue.get()
                if message is None: break
                
                for client in list(self.clients):
                    try: await client.send(message)
                    except: pass
            except: pass

    # --- HANDLER ROUTING ---
    def make_handler(self):
        async def handler(websocket):
            # 1. Handle New Connection
            self.clients.add(websocket)
            self._put("log", f"Client connected")
            self._put("client_count", len(self.clients)) 
            
            try:
                # 2. Wrap the async loop to catch abrupt disconnects cleanly
                try:
                    async for message in websocket:
                        try:
                            # Ignore raw bytes (handled outside of JSON logic)
                            if isinstance(message, bytes): 
                                continue 
                            
                            # Safely parse JSON
                            try:
                                data = json.loads(message)
                                if not isinstance(data, dict):
                                    raise ValueError("Not a dictionary")
                            except (json.JSONDecodeError, ValueError):
                                # Fallback: if it's plain text, treat it as clipboard data
                                self._handle_clipboard(message)
                                continue

                            # 3. Extract Type and Payload
                            msg_type = data.get("type", "unknown")
                            payload = data.get("payload", "")

                            # 4. Route the message
                            if msg_type == "mouse_move": self._handle_mouse_move(payload)
                            elif msg_type == "mouse_click": self._handle_mouse_click(payload)
                            elif msg_type == "mouse_scroll": self._handle_mouse_scroll(payload)
                            elif msg_type == "key_press": self._handle_key_press(payload)
                            
                            # Audio Routing
                            elif msg_type == "audio_start" or msg_type == "audio_control": 
                                if "start" in str(payload) or msg_type == "audio_start":
                                    self.start_audio_streaming()
                                else:
                                    self.stop_audio_streaming()
                            elif msg_type == "audio":
                                action = data.get("action") or (payload.get("action") if isinstance(payload, dict) else "")
                                if action == "start": self.start_audio_streaming()
                                elif action == "stop": self.stop_audio_streaming()
                                
                            # Media & Transfer Routing
                            elif msg_type == "video_frame": self._handle_video_frame(payload)
                            elif msg_type == "audio_frame": self._handle_audio_frame(payload)
                            elif msg_type == "gamepad_state": self._handle_gamepad_state(payload, websocket)
                            elif msg_type == "display_request": self._handle_display_request(payload)
                            elif msg_type == "file_transfer": self._handle_file_transfer(payload)
                            elif msg_type in ["clipboard", "text_transfer", "clipboard_text"]:
                                self._handle_clipboard(payload)
                                
                            # Connection heartbeat
                            elif msg_type == "heartbeat": 
                                await websocket.send(json.dumps({"type": "heartbeat", "payload": "pong"}))
                                
                        except Exception as e: 
                            print(f"❌ Handler Error: {e}")
                            
                # Catch abrupt network drops (screen lock, app swipe, wifi drop)
                except websockets.exceptions.ConnectionClosed:
                    print("⚠️ Client dropped connection abruptly (Normal behavior)")
                    
            finally:
                # 5. Handle Disconnect (Clean or Abrupt)
                if websocket in self.clients: 
                    self.clients.remove(websocket)
                
                # Update UI count and logs
                self._put("client_count", len(self.clients))
                self._put("log", f"❌ Client disconnected")
                
                # Stop camera if it was running for this phone
                self._stop_camera() 
                
            return handler
        return handler
    

    
    def _process_background(self, frame):
        """Applies Blur or Image background using MediaPipe/cvzone"""
        if not self.segmentor or not HAS_SEGMENTATION:
            return frame

        try:
            mode = self.cam_settings.get("background", "none")
            
            if mode == "blur":
                # Create a blurred version of the current frame
                img_bg = cv2.GaussianBlur(frame, (55, 55), 0)
                return self.segmentor.removeBG(frame, imgBg=img_bg, cutThreshold=0.5)
            
            elif mode == "image":
                # Check if we have a loaded background image
                if hasattr(self, 'bg_image_cv2') and self.bg_image_cv2 is not None:
                    # Resize bg image to match frame if needed
                    h, w = frame.shape[:2]
                    if self.bg_image_cv2.shape[:2] != (h, w):
                        self.bg_image_cv2 = cv2.resize(self.bg_image_cv2, (w, h))
                    return self.segmentor.removeBG(frame, imgBg=self.bg_image_cv2, cutThreshold=0.5)
                else:
                    # Fallback to green screen if no image selected
                    return self.segmentor.removeBG(frame, imgBg=(0, 255, 0), cutThreshold=0.5)
                    
        except Exception as e:
            # If segmentation fails, just return original frame so stream doesn't crash
            return frame
            
        return frame


    def _force_create_virtual_display_sync(self):
        """Synchronously creates the virtual display (HDMI Hack) with pixel-perfect coordinates."""
        try:
            import subprocess
            import time
            import re
            self._put("log", "⚙️ Auto-creating Virtual Display (HDMI Hack)...")

            # 1. Find primary screen and its EXACT width
            try:
                out = subprocess.check_output("xrandr | grep ' primary'", shell=True).decode()
                primary_screen = out.split()[0]
                
                # Extract width (e.g., gets "1920" from "1920x1080+0+0")
                match = re.search(r'primary (\d+)x\d+\+\d+\+\d+', out)
                primary_width = int(match.group(1)) if match else 1920
            except Exception:
                primary_screen = "eDP" 
                primary_width = 1920

            target_port = "HDMI-A-0"
            mode_name = "1920x1080_Virtual"

            # 2. Cleanup
            subprocess.run(f"xrandr --output {target_port} --off", shell=True, stderr=subprocess.DEVNULL)
            subprocess.run(f"xrandr --delmode {target_port} {mode_name}", shell=True, stderr=subprocess.DEVNULL)
            subprocess.run(f"xrandr --rmmode {mode_name}", shell=True, stderr=subprocess.DEVNULL)

            # 3. Create Virtual Mode
            modeline = '173.00  1920 2048 2248 2576  1080 1083 1088 1120 -hsync +vsync'
            subprocess.run(f'xrandr --newmode "{mode_name}" {modeline}', shell=True, stderr=subprocess.DEVNULL)
            subprocess.run(f'xrandr --addmode {target_port} "{mode_name}"', shell=True, stderr=subprocess.DEVNULL)

            # 4. Activate using EXACT pixel position instead of --right-of
            cmd = f'xrandr --output {target_port} --mode "{mode_name}" --pos {primary_width}x0'
            result = subprocess.run(cmd, shell=True)

            if result.returncode == 0:
                self._put("log", f"✅ Auto-Virtual Display ready at {primary_width}x0!")
                time.sleep(1.5) 
            else:
                self._put("log", f"❌ Auto-create failed for {target_port}.")

        except Exception as e:
            self._put("log", f"❌ Auto-Virtual Display Error: {e}")

    def _handle_display_request(self, payload):
        """Handle display streaming request and auto-create screens if needed."""
        try:
            import json
            request = json.loads(payload) if isinstance(payload, str) else payload
            action = request.get('action')
            self._put("log", f"🖥️ Display request: {action}")

            if action in ['start_display', 'change_resolution']:
                self._display_width = int(request.get('width', 1280))
                self._display_height = int(request.get('height', 720))
                self._display_fps = int(request.get('fps', 30))
                self._display_quality = int(request.get('quality', 35))
                
                # Default to [0] (Primary/All) if missing
                self._display_monitor_indices = request.get('monitor_indices', [0])
                
                # Handle legacy single index if sent by old app version
                if 'monitor_index' in request and 'monitor_indices' not in request:
                     idx = int(request.get('monitor_index'))
                     self._display_monitor_indices = [idx] if idx >= 0 else [0, 1] 

                self._put("log", f"🖥️ Config: {self._display_width}x{self._display_height} (Monitors: {self._display_monitor_indices})")

            if action == 'start_display':
                if hasattr(self, '_display_thread') and getattr(self, '_display_thread') is not None and self._display_thread.is_alive():
                    self._stop_display_capture(wait_seconds=0.8)

                # ==========================================================
                # --- NEW: SMART AUTO VIRTUAL DISPLAY CHECK ---
                # ==========================================================
                import mss
                with mss.mss() as sct:
                    # sct.monitors[0] is all screens combined, so we subtract 1
                    real_monitor_count = len(sct.monitors) - 1 
                    
                # If they ask for index 1 (Monitor 2), but count is only 1, we must create it
                requested_max_index = max(self._display_monitor_indices)
                
                if requested_max_index >= real_monitor_count:
                    self._put("log", f"⚠️ Monitor {requested_max_index + 1} requested but not found.")
                    self._force_create_virtual_display_sync()
                # ==========================================================

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
            self._display_active = False
            if hasattr(self, '_display_thread') and self._display_thread is not None:
                if self._display_thread.is_alive():
                    self._put("log", "🔄 Stopping previous screen capture...")
                    self._display_thread.join(timeout=wait_seconds)

                if self._display_thread.is_alive():
                    self._put("log", "⚠️ Previous screen capture thread did not stop immediately (continuing).")
                else:
                    self._put("log", "Previous screen capture stopped.")
        except Exception as e:
            self._put("log", f"❌ Error stopping display capture: {e}")

    def _capture_screen_loop(self):
        """Continuously capture specific monitor or combined screens (Flicker-Free using MSS)."""
        try:
            import mss
            import cv2
            import time
            import pyautogui
            import numpy as np

            self._put("log", f"🖥️ Screen capture started via MSS.")

            current_quality = getattr(self, '_display_quality', 35)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), current_quality]
            
            last_mouse_pos = (0, 0)
            last_move_time = time.time()
            HIDE_TIMEOUT = 3.0

            with mss.mss() as sct:
                # Initialize an empty cache
                frame_cache = []

                while self._display_active:
                    start_time = time.time()
                    
                    # --- FIX: Read target_indices INSIDE the loop so it updates dynamically! ---
                    target_indices = self._display_monitor_indices
                    
                    # Resize cache if monitor selection changes (e.g. from 1 monitor to All)
                    if len(frame_cache) != len(target_indices):
                        frame_cache = [None] * len(target_indices)

                    target_w = self._display_width
                    target_h = self._display_height
                    target_fps = getattr(self, '_display_fps', 30)
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
                                # --- FIX: Fallback to primary monitor if requested monitor doesn't exist ---
                                if len(sct.monitors) > 1:
                                    # We only want to log this once per session so it doesn't spam
                                    if getattr(self, '_missing_mon_logged', None) != mss_idx:
                                        self._put("log", f"⚠️ Monitor {idx} missing. Falling back to Main Screen.")
                                        self._missing_mon_logged = mss_idx
                                        
                                    monitor = sct.monitors[1] # 1 is always the Primary Screen
                                    sct_img = sct.grab(monitor)
                                    img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                                    frame_cache[i] = img
                                    frames_ready.append(img)
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
                                    
                                    if 0 <= local_mx < final_frame.shape[1] and 0 <= local_my < final_frame.shape[0]:
                                        cv2.circle(final_frame, (local_mx, local_my), 8, (0, 0, 255), -1)
                                        cv2.circle(final_frame, (local_mx, local_my), 9, (255, 255, 255), 1)
                            except: pass

                        # 4. FIT TO SCREEN
                        h, w = final_frame.shape[:2]
                        scale = min(target_w / w, target_h / h)
                        
                        if scale < 1.0:
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            final_frame = cv2.resize(final_frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                        # 5. ENCODE & SEND RAW BINARY
                        success, buffer = cv2.imencode('.jpg', final_frame, encode_param)
                        if success:
                            header = b'\x03'
                            video_bytes = header + buffer.tobytes()
                            
                            if self._loop and self._loop.is_running():
                                import asyncio
                                asyncio.run_coroutine_threadsafe(
                                    self._broadcast_bytes(video_bytes), self._loop
                                )

                        # 6. FPS LIMIT
                        while (time.time() - start_time) < frame_duration:
                            pass

                    except Exception as e:
                        time.sleep(0.1)

            self._put("log", "🖥️ Screen capture stopped")

        except ImportError:
            self._put("log", "❌ Error: 'mss' or 'pyautogui' missing.")
        except Exception as e:
            self._put("log", f"❌ Capture Fatal Error: {e}")
        finally:
            self._display_active = False

    

    # --- INPUT HANDLERS ---
    def _handle_mouse_move(self, payload):
        try:
            if isinstance(payload, str): data = json.loads(payload)
            else: data = payload
            dx, dy = int(data.get('deltaX', 0)), int(data.get('deltaY', 0))
            if dx != 0 or dy != 0: pyautogui.moveRel(dx, dy, _pause=False)
        except: pass

    def _handle_mouse_click(self, payload):
        try:
            d = json.loads(payload)
            btn = {0:'left',1:'right',2:'middle'}.get(d.get("button", 0), 'left')
            act = d.get("action", "click")
            if act == "down": pyautogui.mouseDown(button=btn, _pause=False)
            elif act == "up": pyautogui.mouseUp(button=btn, _pause=False)
            else: pyautogui.click(button=btn, _pause=False)
        except: pass

    def _handle_mouse_scroll(self, payload):
        try:
            d = json.loads(payload)
            pyautogui.scroll(int(d.get("scrollDelta", 0) * 5), _pause=False)
        except: pass

    
    def _handle_key_press(self, payload):
        """Handle keyboard input (Respects Down/Up actions and Left/Right Modifiers)."""
        try:
            import subprocess
            import pyautogui
            import json

            if isinstance(payload, str): d = json.loads(payload)
            else: d = payload

            key = d.get("key", "")
            raw_key = key.lower().replace(" ", "").replace("_", "") 
            modifiers = [m.lower() for m in d.get("modifiers", [])]
            action = d.get("action", "press") 

            # --- AUTO-SHIFT FIX ---
            if len(key) == 1 and key.isupper() and key.isalpha():
                if "shift" not in modifiers:
                    modifiers.append("shift")

            # --- 1. LINUX SPECIAL COMMANDS (Brightness & Media) ---
            linux_special_map = {
                "brightnessup": "XF86MonBrightnessUp", "brightnessdown": "XF86MonBrightnessDown",
                "volumemute": "XF86AudioMute", "volumedown": "XF86AudioLowerVolume", "volumeup": "XF86AudioRaiseVolume",
                "playpause": "XF86AudioPlay", "nexttrack": "XF86AudioNext", "prevtrack": "XF86AudioPrev",
                "stop": "XF86AudioStop", "search": "XF86Search", "home": "Home", "end": "End",
                "pageup": "Page_Up", "pagedown": "Page_Down", "insert": "Insert", "delete": "Delete",
                "printscreen": "Print", "numlock": "Num_Lock", "scrolllock": "Scroll_Lock", "pause": "Pause",
                "menu": "Menu", "capslock": "Caps_Lock"
            }

            if raw_key in linux_special_map:
                key_code = linux_special_map[raw_key]
                if action == "down": subprocess.Popen(["xdotool", "keydown", key_code])
                elif action == "up": subprocess.Popen(["xdotool", "keyup", key_code])
                else: subprocess.Popen(["xdotool", "key", key_code])
                return

            # --- 2. NUMPAD MAPPING ---
            numpad_map = {
                "numpad0": "0", "num0": "0", "numpad1": "1", "num1": "1", "numpad2": "2", "num2": "2",
                "numpad3": "3", "num3": "3", "numpad4": "4", "num4": "4", "numpad5": "5", "num5": "5",
                "numpad6": "6", "num6": "6", "numpad7": "7", "num7": "7", "numpad8": "8", "num8": "8",
                "numpad9": "9", "num9": "9", "numpadenter": "enter", "numpadadd": "+", "numpadsubtract": "-", 
                "numpadmultiply": "*", "numpaddivide": "/", "numpaddecimal": "."
            }

            if raw_key in numpad_map:
                final_key = numpad_map[raw_key]
                if action == "down": pyautogui.keyDown(final_key)
                elif action == "up": pyautogui.keyUp(final_key)
                else: pyautogui.press(final_key)
                return

            # --- 3. WINDOWS / SUPER KEY (Linux uses Super_L) ---
            if raw_key in ["win", "windows", "super", "meta", "cmd"]:
                if action == "up": return
                subprocess.Popen(["xdotool", "key", "Super_L"])
                return

            # --- 4. ARROWS ---
            arrow_map = {
                "up": "up", "down": "down", "left": "left", "right": "right",
                "↑": "up", "↓": "down", "←": "left", "→": "right"
            }
            if raw_key in arrow_map:
                k = arrow_map[raw_key]
                if action == "down": pyautogui.keyDown(k)
                elif action == "up": pyautogui.keyUp(k)
                else: pyautogui.press(k)
                return

            # --- 5. STANDARD KEYS & LEFT/RIGHT MODIFIERS ---
            key_map = {
                "enter": "enter", "return": "enter", "backspace": "backspace", 
                "tab": "tab", "space": "space", "esc": "escape",
                "-": "-", "=": "=", "[": "[", "]": "]", "\\": "\\", 
                ";": ";", "'": "'", ",": ",", ".": ".", "/": "/", "`": "`"
            }
            
            # Map specific left/right modifiers to PyAutoGUI readable strings
            pyautogui_mod_map = {
                'shift_l': 'shiftleft', 'shift_r': 'shiftright',
                'ctrl_l': 'ctrlleft', 'ctrl_r': 'ctrlright',
                'alt_l': 'altleft', 'alt_r': 'altright',
                'shift': 'shift', 'ctrl': 'ctrl', 'alt': 'alt'
            }

            final_key = key_map.get(raw_key, raw_key)
            
            # Filter out the meta keys (handled above) and map the rest
            mapped_mods = [
                pyautogui_mod_map.get(m, m) 
                for m in modifiers if m not in ["win", "meta", "super", "win_l", "win_r"]
            ]
            
            keys_to_press = mapped_mods + [final_key]

            if action == "down":
                for k in keys_to_press: pyautogui.keyDown(k)
            elif action == "up":
                for k in reversed(keys_to_press): pyautogui.keyUp(k)
            else:
                pyautogui.hotkey(*keys_to_press)

        except Exception as e:
            self._put("log", f"❌ Key Handler Error: {e}")

  
    # --- GAMEPAD ---
   
   
    # ============================================
    # 🎮 GAMEPAD HANDLER (VERIFIED & FIXED)
    # ============================================

    def _create_uinput_device(self):
        """Creates a Virtual Xbox 360 Controller."""
        if not HAS_EVDEV: return None
        
        # Define Controller Capabilities (Xbox 360 Standard)
        cap = {
            ecodes.EV_KEY: [
                ecodes.BTN_A, ecodes.BTN_B, ecodes.BTN_X, ecodes.BTN_Y,
                ecodes.BTN_TL, ecodes.BTN_TR, ecodes.BTN_SELECT, ecodes.BTN_START,
                ecodes.BTN_MODE, ecodes.BTN_THUMBL, ecodes.BTN_THUMBR,
                ecodes.BTN_DPAD_UP, ecodes.BTN_DPAD_DOWN, ecodes.BTN_DPAD_LEFT, ecodes.BTN_DPAD_RIGHT
            ],
            ecodes.EV_ABS: [
                # Left Stick (16-bit signed)
                (ecodes.ABS_X, AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=16, resolution=0)),
                (ecodes.ABS_Y, AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=16, resolution=0)),
                # Right Stick
                (ecodes.ABS_RX, AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=16, resolution=0)),
                (ecodes.ABS_RY, AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=16, resolution=0)),
                # Triggers (0-255)
                (ecodes.ABS_Z, AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0)),  # Left Trigger
                (ecodes.ABS_RZ, AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0)), # Right Trigger
            ]
        }

        try:
            # FIX: Explicitly unpacking events is safer for some evdev versions
            return UInput(events=cap, name="UseAs-Virtual-Controller", version=0x1)
        except PermissionError:
            self._put("log", "❌ Gamepad Error: Permission denied for /dev/uinput")
            self._put("log", "💡 FIX: Run 'sudo chmod 666 /dev/uinput' or setup udev rules.")
            return "FALLBACK" # Signal to use Keyboard instead
        except Exception as e:
            self._put("log", f"❌ Gamepad Create Error: {e}")
            return None

    def _handle_gamepad_state(self, payload, client_ws):
        """Handles Gamepad with FULL PlayStation/Xbox Button Support."""
        try:
            state = json.loads(payload)
            buttons = state.get('buttons', {})
            
            # 1. Setup Device
            if client_ws not in self.client_gamepads:
                dev = self._create_uinput_device()
                if dev:
                    self.client_gamepads[client_ws] = dev
                    if dev != "FALLBACK": self._put("log", "🎮 Virtual Controller Connected")
                    else: self._put("log", "⚠️ Fallback: Keyboard Mode")
                else: return

            device = self.client_gamepads[client_ws]
            
            # Get previous state for edge detection (Important for HOLDING buttons)
            prev_state = self.gamepad_history.get(client_ws, {'buttons': {}})
            prev_btns = prev_state['buttons']

            # =================================================
            # MODE A: VIRTUAL CONTROLLER (Linux UInput)
            # =================================================
            if device != "FALLBACK":
                btn_map = {
                    # --- Face Buttons ---
                    'a': ecodes.BTN_A, 'cross': ecodes.BTN_A,
                    'b': ecodes.BTN_B, 'circle': ecodes.BTN_B,
                    'x': ecodes.BTN_X, 'square': ecodes.BTN_X,
                    'y': ecodes.BTN_Y, 'triangle': ecodes.BTN_Y,
                    
                    # --- Shoulders / Bumpers ---
                    'lb': ecodes.BTN_TL, 'l1': ecodes.BTN_TL, 'l_shoulder': ecodes.BTN_TL,
                    'rb': ecodes.BTN_TR, 'r1': ecodes.BTN_TR, 'r_shoulder': ecodes.BTN_TR,
                    
                    # --- Digital Triggers (Fallback if axis missing) ---
                    'lt': ecodes.BTN_TL2, 'l2': ecodes.BTN_TL2,
                    'rt': ecodes.BTN_TR2, 'r2': ecodes.BTN_TR2,

                    # --- Center Buttons ---
                    'start': ecodes.BTN_START, 'options': ecodes.BTN_START, 'menu': ecodes.BTN_START,
                    'select': ecodes.BTN_SELECT, 'share': ecodes.BTN_SELECT, 'back': ecodes.BTN_SELECT, 'view': ecodes.BTN_SELECT,
                    'mode': ecodes.BTN_MODE, 'xbox': ecodes.BTN_MODE, 'ps': ecodes.BTN_MODE, 'center': ecodes.BTN_MODE,
                    
                    # --- Thumbs ---
                    'l_thumb': ecodes.BTN_THUMBL, 'thumbl': ecodes.BTN_THUMBL,
                    'r_thumb': ecodes.BTN_THUMBR, 'thumbr': ecodes.BTN_THUMBR,
                    
                    # --- D-Pad ---
                    'dpad_up': ecodes.BTN_DPAD_UP, 'up': ecodes.BTN_DPAD_UP,
                    'dpad_down': ecodes.BTN_DPAD_DOWN, 'down': ecodes.BTN_DPAD_DOWN,
                    'dpad_left': ecodes.BTN_DPAD_LEFT, 'left': ecodes.BTN_DPAD_LEFT,
                    'dpad_right': ecodes.BTN_DPAD_RIGHT, 'right': ecodes.BTN_DPAD_RIGHT
                }

                # Update Buttons
                for name, code in btn_map.items():
                    curr = buttons.get(name, False)
                    # Only write if state changed to prevent spamming
                    if curr != prev_btns.get(name, False):
                        device.write(ecodes.EV_KEY, code, 1 if curr else 0)

                # Update Axes
                def scale(val): return int(val * 32767)
                def scale_trig(val): return int(val * 255)

                device.write(ecodes.EV_ABS, ecodes.ABS_X, scale(float(state.get('leftStickX', 0))))
                device.write(ecodes.EV_ABS, ecodes.ABS_Y, scale(float(state.get('leftStickY', 0))))
                device.write(ecodes.EV_ABS, ecodes.ABS_RX, scale(float(state.get('rightStickX', 0))))
                device.write(ecodes.EV_ABS, ecodes.ABS_RY, scale(float(state.get('rightStickY', 0))))
                
                # Triggers: Prefer Axis, Fallback to Digital Button
                lt_val = float(state.get('leftTrigger', 0))
                # If axis is 0 but button is pressed (digital mode), set to max
                if lt_val == 0 and (buttons.get('l2') or buttons.get('lt')): lt_val = 1.0
                
                rt_val = float(state.get('rightTrigger', 0))
                if rt_val == 0 and (buttons.get('r2') or buttons.get('rt')): rt_val = 1.0

                device.write(ecodes.EV_ABS, ecodes.ABS_Z, scale_trig(lt_val))
                device.write(ecodes.EV_ABS, ecodes.ABS_RZ, scale_trig(rt_val))
                
                device.syn()

            # =================================================
            # MODE B: KEYBOARD FALLBACK (WASD + Utilities)
            # =================================================
            else:
                import pyautogui
                
                # Map Start/Select to Esc/Tab (Menu/Map)
                if (buttons.get('start') or buttons.get('options') or buttons.get('menu')) and not prev_btns.get('start'):
                    pyautogui.press('esc')
                
                if (buttons.get('select') or buttons.get('share') or buttons.get('back')) and not prev_btns.get('select'):
                    pyautogui.press('tab')

                # WASD Logic (Stick)
                ly = float(state.get('leftStickY', 0))
                lx = float(state.get('leftStickX', 0))
                
                # Use hold logic for WASD to allow walking
                if ly < -0.5: pyautogui.keyDown('w')
                else: pyautogui.keyUp('w')
                
                if ly > 0.5: pyautogui.keyDown('s')
                else: pyautogui.keyUp('s')

                if lx < -0.5: pyautogui.keyDown('a')
                else: pyautogui.keyUp('a')

                if lx > 0.5: pyautogui.keyDown('d')
                else: pyautogui.keyUp('d')

                # Action Buttons (Hold Logic)
                # A/Cross -> Space (Jump)
                is_a = buttons.get('a') or buttons.get('cross')
                was_a = prev_btns.get('a') or prev_btns.get('cross')
                if is_a and not was_a: pyautogui.keyDown('space')
                elif not is_a and was_a: pyautogui.keyUp('space')

                # B/Circle -> Enter (Interact)
                is_b = buttons.get('b') or buttons.get('circle')
                was_b = prev_btns.get('b') or prev_btns.get('circle')
                if is_b and not was_b: pyautogui.keyDown('enter')
                elif not is_b and was_b: pyautogui.keyUp('enter')

            # 3. Save History
            self.gamepad_history[client_ws] = {'buttons': buttons}

        except Exception as e:
            pass

    def _init_audio_stream(self):
        """Opens default stream, then moves it to Virtual Mic"""
        if self.audio_stream is None:
            try:
                # 1. Open Default Stream (Plays to Speakers initially)
                self.audio_stream = self.p.open(
                    format=self.AUDIO_FORMAT,
                    channels=self.AUDIO_CHANNELS,
                    rate=self.AUDIO_RATE,
                    output=True,
                    frames_per_buffer=1024
                )
                self._put("log", "🎤 Audio Started. Routing to Mic...")

                # 2. Trigger the "Mover" in background
                # We use a thread so it doesn't freeze the app while waiting
                threading.Thread(target=self._force_route_audio, daemon=True).start()
                
            except Exception as e:
                self._put("log", f"❌ Audio Init Failed: {e}")

    def _force_route_audio(self):
        """
        Aggressively finds the Python audio stream and moves it.
        """
        # Wait a bit longer for the stream to fully register in PulseAudio
        time.sleep(1.5) 
        
        try:
            # Get all sink inputs
            result = subprocess.run(
                "pactl list sink-inputs", 
                shell=True, 
                stdout=subprocess.PIPE, 
                text=True
            )
            output = result.stdout
            
            current_id = None
            
            # Loop through the output line by line
            for line in output.split('\n'):
                # capture the ID
                if "Sink Input #" in line:
                    current_id = line.split("#")[1].strip()
                
                # CHECK 1: Is it our app? (Usually shows as 'python3' or 'ALSA plug-in')
                # CHECK 2: Is it ALREADY on the Mic? (If so, skip)
                # We look for "media.name" or "application.name"
                if current_id and ("python" in line.lower() or "alsa" in line.lower() or "audio stream" in line.lower()):
                    
                    self._put("log", f"🔍 Found Candidate Stream #{current_id}")
                    
                    # Try to move it blindly. If it's not ours, it might move another 
                    # system sound, but that's a rare risk.
                    move_cmd = f"pactl move-sink-input {current_id} UseAs_Mic"
                    move_result = subprocess.run(move_cmd, shell=True, stderr=subprocess.DEVNULL)
                    
                    if move_result.returncode == 0:
                        self._put("log", f"Moved Stream #{current_id} to Virtual Mic")
                        return # Success, stop looking
            
            self._put("log", "⚠️ No movable audio stream found.")
            
        except Exception as e:
            self._put("log", f"❌ Routing Error: {e}")


    def _setup_virtual_driver(self):
        """Creates the Virtual Mic if missing"""
        try:
            # Check if exists
            check = subprocess.run("pactl list short sinks | grep UseAs_Mic", shell=True, stdout=subprocess.DEVNULL)
            
            if check.returncode != 0:
                self._put("log", "⚙️ Creating Virtual Microphone...")
                cmd = "pactl load-module module-null-sink sink_name=UseAs_Mic sink_properties=device.description=\"UseAs_Virtual_Microphone\""
                subprocess.run(cmd, shell=True)
                subprocess.run("pactl set-sink-volume UseAs_Mic 100%", shell=True)
                self._put("log", "Virtual Microphone Created")
            
            # We do NOT set os.environ here anymore.
            # We will find the ID manually in the next step.
            
        except Exception as e:
            self._put("log", f"⚠️ Driver Setup Error: {e}")

    # --- 2. FIND ID & CONNECT ---
    

    def _handle_audio_frame(self, payload):
        """Plays received audio chunk"""
        try:
            # 1. Initialize stream on first packet
            if self.audio_stream is None:
                self._init_audio_stream()

            # 2. Decode Base64 -> Raw PCM Bytes
            audio_data = base64.b64decode(payload)

            # 3. Write to Speakers
            if self.audio_stream:
                # exception_on_underflow=False prevents crash if network lags
                self.audio_stream.write(audio_data, exception_on_underflow=False)

        except Exception as e:
            print(f"Audio Error: {e}")

    # OPTIONAL: Cleanup method if you want to close stream strictly
    def close(self):
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        self.p.terminate()
    
    def _handle_audio_stop(self):
        """Closes the audio stream to release speakers"""
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self._put("log", "🔇 Audio Stream Stopped")
            except: pass
            self.audio_stream = None

    # --- AUDIO & SCREEN ---
    # --- AUDIO STREAMING (PC -> PHONE) ---
    def start_audio_streaming(self):
        if not HAS_AUDIO:
            self._put("log", "❌ Error: 'pyaudio' missing.")
            return
        if self._streaming_audio: return
        
        self._streaming_audio = True
        
        def worker():
            p = pyaudio.PyAudio()
            try:
                # FIX: Use System Default (None) instead of forcing ID 3
                # This lets Linux (PulseAudio/PipeWire) handle the routing safely.
                self._put("log", "🔊 PC Audio Stream Started (Using Default)")
                
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=2,
                    rate=48000,
                    input=True,
                    input_device_index=None, # <--- CRITICAL CHANGE: Let OS decide
                    frames_per_buffer=1024
                )
                
                while self._streaming_audio:
                    try:
                        data = stream.read(1024, exception_on_overflow=False)
                        b64 = base64.b64encode(data).decode('utf-8')
                        msg = json.dumps({"type": "audio_frame", "rate": 48000, "payload": b64})
                        if self._loop: 
                            asyncio.run_coroutine_threadsafe(self._broadcast_text(msg), self._loop)
                    except Exception as e:
                        # SHOW THE ERROR if it crashes
                        self._put("log", f"❌ Stream Crash: {e}") 
                        break
            except Exception as e:
                self._put("log", f"❌ Audio Init Error: {e}")
            finally:
                p.terminate()
                self._streaming_audio = False
                self._put("log", "🔇 Audio Stream Stopped")

        threading.Thread(target=worker, daemon=True).start()

    def stop_audio_streaming(self):
        self._streaming_audio = False

    async def _broadcast_text(self, message):
        """Helper to broadcast JSON messages to all phones"""
        for client in list(self.clients):
            try: await client.send(message)
            except: pass


    


    def _handle_clipboard(self, payload):
        try:
            text_to_copy = ""
            
            # 1. Try to parse JSON payload (e.g., '{"text": "Hello"}')
            if isinstance(payload, str):
                try:
                    data = json.loads(payload)
                    if isinstance(data, dict) and "text" in data:
                        text_to_copy = data["text"]
                    else:
                        text_to_copy = payload # It was just a raw string
                except:
                    text_to_copy = payload # Not JSON, use as is
            else:
                text_to_copy = str(payload)

            print(f"📋 COPYING: {text_to_copy}") # Debug log

            # 2. Update Linux Clipboard
            pyperclip.copy(text_to_copy)
            self._put("log", f"📋 Clipboard updated ({len(text_to_copy)} chars)")
            
            # 3. Update GUI Textbox
            self._put("clip", text_to_copy) 
            
        except Exception as e:
            print(f"❌ CLIPBOARD ERROR: {e}")
            self._put("log", f"❌ Clip Error: {e}")

    # --- FIXED FILE RECEIVER (Robust) ---
    def _handle_file_transfer(self, payload):
        try:
            data = json.loads(payload) if isinstance(payload, str) else payload
            filename = data.get("filename", "unknown.dat")
            b64_data = data.get("data")
            is_end = data.get("is_end", False)
            
            file_path = SAVE_DIR / filename
            
            if b64_data:
                try:
                    with open(file_path, "ab") as f:
                        f.write(base64.b64decode(b64_data))
                except Exception as e:
                    self._put("log", f"❌ File Write Error: {e}")

            if is_end:
                self._put("log", f"📥 Received: {filename}")
                self._put("refresh_files", "") # <--- NEW: Tells GUI to update list
                
        except Exception as e:
            self._put("log", f"❌ File Recv Error: {e}")

    async def _broadcast_bytes(self, data):
        """Helper to send raw binary data to all clients"""
        for client in list(self.clients):
            try: 
                await client.send(data)
            except: 
                pass

    def send_file_to_phone_thread(self, file_path):
        def worker():
            try:
                original_name = os.path.basename(file_path)
                # Remove protocol characters from filename (Fix for Linux timestamps)
                filename = original_name.replace(":", "_")
                file_size = os.path.getsize(file_path)

                self._put("log", f"📤 Starting transfer: {filename}")
                
                # Acquire lock to prevent mixing files
                self.sending_lock.acquire()

                # 1. SEND START SIGNAL (As JSON Text)
                # Payload matches your Android 'substringAfter' logic
                start_payload = json.dumps({"filename": filename, "size": file_size})
                
                # Send JSON via the existing helper
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

                        # Send Binary directly to clients
                        if self._loop and self._loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                self._broadcast_bytes(packet),
                                self._loop
                            )

                        total_sent += len(chunk)
                        chunk_counter += 1

                        # Tiny sleep to prevent router buffer overflow
                        time.sleep(0.005)

                # 3. SEND END SIGNAL (As Binary Frame)
                # Header 0x02 means "End of File"
                end_packet = b'\x02'
                if self._loop:
                    asyncio.run_coroutine_threadsafe(
                        self._broadcast_bytes(end_packet), 
                        self._loop
                    )

                self._put("log", f" Sent: {filename}")

            except Exception as e:
                self._put("log", f"❌ Error: {e}")
            finally:
                if self.sending_lock.locked():
                    self.sending_lock.release()

        threading.Thread(target=worker, daemon=True).start()

    # --- SERVER STARTUP ---
    async def _async_starter(self):
        handler = self.make_handler()
        self._ws_server = await websockets.serve(handler, self.host, self.port, max_size=20*1024*1024, ping_interval=None)
        self._broadcast_task = asyncio.create_task(self._broadcast_worker())
        return self._ws_server

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._broadcast_queue = asyncio.Queue()
        try:
            self._ws_server = self._loop.run_until_complete(self._async_starter())
            self._put("log", f"🌐 WebSocket listening on 0.0.0.0:{self.port}")
            self._loop.run_forever()
        except Exception as e: 
            self._put("log", f"Err: {e}")
        finally: 
            # Force clean up any remaining async generators before closing
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    def start(self):
        # 1. Start Discovery Server
        self.discovery = DiscoveryServer(port=self.port)
        self.discovery.daemon = True
        self.discovery.start()

        # 2. Start UDP Mouse Server
        self.udp_port = self.port + 1
        self.udp_mouse = UDPMouseServer(port=self.udp_port)
        self.udp_mouse.daemon = True
        self.udp_mouse.start()

        # 3. Start Main WebSocket Event Loop
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        
        self._put("log", "⏳ Shutting down server and releasing ports...")
        
        # 1. Stop External Services
        if hasattr(self, 'discovery') and self.discovery: 
            self.discovery.stop()
        if hasattr(self, 'udp_mouse') and self.udp_mouse: 
            self.udp_mouse.stop()
            
        # 2. Stop AV Streams
        self._vcam_running = False
        self._streaming_audio = False
        self._display_active = False
        self.fake_cam = None 
        self._handle_audio_stop()
        
        # 3. Nuke the WebSocket Server & Asyncio Loop
        if self._loop and self._loop.is_running():
            try:
                # Synchronously force the server socket to close instantly
                if hasattr(self, '_ws_server') and self._ws_server:
                    self._ws_server.close()

                # Schedule the loop to stop immediately without waiting for clients
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception as e:
                print(f"Error during async shutdown: {e}")

        # 4. Wait for the main thread to die
        if hasattr(self, '_thread') and self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            
        self._put("log", " Server fully shut down.")

# ============================================
# PART 2: MODERN UI (CustomTkinter)
# ============================================



class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, parent, prefs, callback_save):
        super().__init__(parent)
        self.title("Configuration")
        
        # 1. Make window larger and allow resizing
        self.geometry("550x750")
        self.resizable(True, True) 
        
        self.prefs = prefs
        self.callback_save = callback_save
        self.transient(parent)
        self.configure(fg_color="#0F0F0F")
        
        # 2. Anchor buttons to the bottom FIRST so they never get cut off
        self.bottom_bar = ctk.CTkFrame(self, fg_color="transparent")
        self.bottom_bar.pack(side="bottom", fill="x", padx=40, pady=20)
        
        ctk.CTkButton(self.bottom_bar, text="Cancel", command=self.destroy, height=45, 
                      fg_color="#cf6679", hover_color="#b00020").pack(side="left", expand=True, padx=(0, 10))
                      
        ctk.CTkButton(self.bottom_bar, text="Save Settings", command=self.save, height=45, 
                      fg_color="#00E676", text_color="black", hover_color="#00c853").pack(side="right", expand=True, padx=(10, 0))

        # 3. Put all settings inside a Scrollable Frame
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(self.scroll, text="Configuration", font=("Courier New", 24, "bold"), text_color="white").pack(pady=(10, 20))

        # --- General ---
        self.frame_gen = ctk.CTkFrame(self.scroll, fg_color="#141414", corner_radius=5)
        self.frame_gen.pack(fill="x", padx=20, pady=(0, 10))
        ctk.CTkLabel(self.frame_gen, text="General", font=("Courier New", 14, "bold"), text_color="#00E676").pack(anchor="w", padx=15, pady=5)
        
        self.var_autostart = ctk.BooleanVar(value=prefs.get("autostart", False))
        ctk.CTkSwitch(self.frame_gen, text="Auto-start with Linux Login", variable=self.var_autostart, progress_color="#00E676").pack(anchor="w", padx=15, pady=8)
        
        self.var_auto_server = ctk.BooleanVar(value=prefs.get("auto_server", True))
        ctk.CTkSwitch(self.frame_gen, text="Auto-start Server on launch", variable=self.var_auto_server, progress_color="#00E676").pack(anchor="w", padx=15, pady=8)

        # --- Input & Gaming ---
        self.frame_game = ctk.CTkFrame(self.scroll, fg_color="#141414", corner_radius=5)
        self.frame_game.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(self.frame_game, text="Input & Gaming", font=("Courier New", 14, "bold"), text_color="#00E676").pack(anchor="w", padx=15, pady=5)
        
        self.var_gaming = ctk.BooleanVar(value=prefs.get("gaming_mode", True))
        ctk.CTkSwitch(self.frame_game, text="Gaming Mode (Low Latency)", variable=self.var_gaming, progress_color="#00E676").pack(anchor="w", padx=15, pady=8)
        
        # --- UI Scaling ---
        self.frame_ui = ctk.CTkFrame(self.scroll, fg_color="#141414", corner_radius=5)
        self.frame_ui.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(self.frame_ui, text="UI Scaling(Restart Required)", font=("Courier New", 14, "bold"), text_color="#00E676").pack(anchor="w", padx=15, pady=5)
        
        current_scale = prefs.get("scale", 1.4)
        self.opt_scale = ctk.CTkOptionMenu(self.frame_ui, values=["1.0", "1.2", "1.4", "1.5", "1.6", "1.8", "2.0", "2.2"], fg_color="#0a0a0a", button_color="#222")
        self.opt_scale.set(str(current_scale))
        self.opt_scale.pack(fill="x", padx=15, pady=10)

        # --- Server Port ---
        self.frame_port = ctk.CTkFrame(self.scroll, fg_color="#141414", corner_radius=5)
        self.frame_port.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(self.frame_port, text="Server Port:", font=("Courier New", 13, "bold"), text_color="white").pack(side="left", padx=20, pady=15)
        
        self.ent_port = ctk.CTkEntry(self.frame_port, width=80, justify="center", fg_color="#0F0F0F", border_color="#333", text_color="white")
        self.ent_port.insert(0, str(prefs.get("port", 8089)))
        self.ent_port.pack(side="right", padx=20, pady=15)

        self.after(100, self.safe_grab)

    def safe_grab(self):
        try:
            self.grab_set()
            self.focus_set()
        except: pass

    def save(self):
        try:
            port = int(self.ent_port.get())
            scale = float(self.opt_scale.get())
            new_prefs = {
                "autostart": self.var_autostart.get(),
                "auto_server": self.var_auto_server.get(),
                "gaming_mode": self.var_gaming.get(),
                "port": port,
                "scale": scale 
            }
            self.callback_save(new_prefs)
            self.destroy()
        except ValueError:
            pass

class ServerGUI:
    def __init__(self):
        self.prefs = {"autostart": False, "auto_server": True, "port": 8089, "scale": 1.4}
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, 'r') as f: 
                    self.prefs.update(json.load(f))
        except: pass

        ctk.set_appearance_mode("Dark")
        scaling_factor = self.prefs.get("scale", 1.4)
        ctk.set_widget_scaling(scaling_factor)  
        ctk.set_window_scaling(scaling_factor)

        self.root = ctk.CTk()
        self.root.title(f"Use As Server v{APP_VERSION}")
        self.root.geometry("1100x750")
        self.root.configure(fg_color="#0B0B0B")
        
        self.update_queue = queue.Queue()
        self.server = None
        self.is_running = False
        self.tray_icon = None
        self.client_count = 0

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(100, self.process_queue)
        
        # Start server automatically if requested (matches the "Always On" nature of image)
        if self.prefs.get("auto_server", True):
            self.toggle_server(force_start=True)

    def load_preferences(self):
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, 'r') as f: self.prefs.update(json.load(f))
        except: pass

    def save_preferences(self, new_prefs):
        self.prefs.update(new_prefs)
        try:
            with open(SETTINGS_FILE, 'w') as f: json.dump(self.prefs, f)
        except: pass
        if self.server:
            self.server.gaming_mode = self.prefs["gaming_mode"]

    def minimize_to_tray(self):
        self.root.withdraw() 
        image = Image.new('RGB', (64, 64), "#00e676")
        ImageDraw.Draw(image).rectangle((16, 16, 48, 48), fill="#1e1e1e")
        menu = (item('Restore', self.restore_from_tray, default=True), item('Quit', self.quit_app))
        self.tray_icon = pystray.Icon("UseAs", image, "Use As Server", menu)
        threading.Thread(target=self.tray_icon.run, daemon=True).start()


    def check_for_updates(self):
        """Starts the update check in a background thread."""
        self.btn_update.configure(text="Checking...", state="disabled")
        threading.Thread(target=self._fetch_latest_version, daemon=True).start()

    def _fetch_latest_version(self):
        """Fetches the latest release tag from GitHub API."""
        try:
            # FIX: Access the global GITHUB_REPO directly, without 'self.'
            url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
            
            # Use a generic User-Agent (GitHub API requires it)
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                # Clean up the tag (e.g., 'v1.6' -> '1.6')
                latest_version = data.get("tag_name", "").replace("v", "").strip()
                release_url = data.get("html_url", GITHUB_URL)
                
                if latest_version and latest_version != APP_VERSION:
                    # Update available
                    self.root.after(0, lambda: self._prompt_update(latest_version, release_url))
                else:
                    # Up to date
                    self.root.after(0, lambda: self._reset_update_button("Up to date!"))
                    self.root.after(2000, lambda: self._reset_update_button())
                    
        except Exception as e:
            print(f"Update Check Error: {e}")
            self.root.after(0, lambda: self._reset_update_button("Update Check Failed"))
            self.root.after(3000, lambda: self._reset_update_button())

    def _prompt_update(self, latest_version, release_url):
        """Shows a dialog asking the user if they want to download the update."""
        self._reset_update_button()
        dialog = messagebox.askyesno(
            "Update Available", 
            f"Version {latest_version} is available!\n(You are on v{APP_VERSION})\n\nWould you like to download it now?"
        )
        if dialog:
            webbrowser.open(release_url)
            
    def _reset_update_button(self, text="↺ Check for Update"):
        """Helper to reset the button state on the main thread."""
        if hasattr(self, 'btn_update'):
            self.btn_update.configure(text=text, state="normal")

    def restore_from_tray(self, icon=None, item=None):
        if self.tray_icon: 
            self.tray_icon.stop()
            self.tray_icon = None
        self.root.after(0, self.root.deiconify)

    def quit_app(self, icon=None, item=None):
        if self.tray_icon: 
            self.tray_icon.stop()
        self.root.after(0, lambda: os._exit(0))

    def setup_ui(self):
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # --- SIDEBAR ---
        self.sidebar = ctk.CTkFrame(self.root, width=220, corner_radius=0, fg_color="#121212")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Header section
        header_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        header_frame.pack(fill="x", padx=15, pady=(15, 10))
        
        # Icon + Title
        ctk.CTkLabel(header_frame, text="■ Use As Server · v1.5", font=("Courier New", 10), text_color="#00A2ED").pack(anchor="w")
        ctk.CTkLabel(self.sidebar, text="USE AS", font=("Courier New", 12, "bold"), text_color="#E0E0E0").pack(anchor="w", padx=20, pady=(10,0))
        ctk.CTkLabel(self.sidebar, text="SERVER", font=("Courier New", 20, "bold"), text_color="#00E676").pack(anchor="w", padx=20, pady=(0, 10))
        
        # Divider
        ctk.CTkFrame(self.sidebar, height=2, fg_color="#00E676").pack(fill="x", padx=15, pady=(0, 15))

        # Nav Buttons (styled to match)
        self.btn_dash = ctk.CTkButton(self.sidebar, text="■  Dashboard", fg_color="#1C2722", text_color="#00E676", anchor="w", corner_radius=5, hover_color="#24332D", command=lambda: self.select_tab("dash"))
        self.btn_dash.pack(fill="x", padx=15, pady=5)
        
        self.btn_cam = ctk.CTkButton(self.sidebar, text="○  Camera Studio", fg_color="transparent", text_color="#A0A0A0", anchor="w", hover_color="#1A1A1A", command=lambda: self.select_tab("cam"))
        self.btn_cam.pack(fill="x", padx=15, pady=5)
        
        self.btn_share = ctk.CTkButton(self.sidebar, text="⇆  Sharing", fg_color="transparent", text_color="#A0A0A0", anchor="w", hover_color="#1A1A1A", command=lambda: self.select_tab("share"))
        self.btn_share.pack(fill="x", padx=15, pady=5)

        # Bottom section logic
        bottom_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bottom_frame.pack(side="bottom", fill="x", padx=15, pady=(0, 20))

        self.lbl_connected = ctk.CTkLabel(bottom_frame, text="○ 0 connected", font=("Courier New", 10), text_color="#A0A0A0")
        self.lbl_connected.pack(anchor="w", pady=(0, 10))

        btn_settings = ctk.CTkButton(bottom_frame, text="⚙ Settings", fg_color="#000", border_color="#00E676", border_width=1, text_color="#00E676", hover_color="#111", command=self.open_settings)
        btn_settings.pack(fill="x", pady=4)
        
        self.btn_update = ctk.CTkButton(
            bottom_frame, 
            text="↺ Check for Update", 
            fg_color="#000", 
            border_color="#00E676", 
            border_width=1, 
            text_color="#00E676", 
            hover_color="#111", 
            command=self.check_for_updates  # Bind the function here
        )
        self.btn_update.pack(fill="x", pady=4)
        
        btn_tray = ctk.CTkButton(bottom_frame, text="□ Tray", fg_color="#000", border_color="#00E676", border_width=1, text_color="#00E676", hover_color="#111", command=self.minimize_to_tray)
        btn_tray.pack(fill="x", pady=4)

        # --- MAIN CONTENT AREA ---
        self.main_container = ctk.CTkFrame(self.root, fg_color="#0B0B0B", corner_radius=0)
        self.main_container.grid(row=0, column=1, sticky="nsew")
        
        self.frame_dash = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.frame_cam = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.frame_share = ctk.CTkFrame(self.main_container, fg_color="transparent")

        self.build_dash(self.frame_dash)
        self.build_cam(self.frame_cam)
        self.build_share(self.frame_share)
        
        self.select_tab("dash")

    def select_tab(self, name):
        # Reset nav buttons
        self.btn_dash.configure(fg_color="transparent", text_color="#A0A0A0")
        self.btn_cam.configure(fg_color="transparent", text_color="#A0A0A0")
        self.btn_share.configure(fg_color="transparent", text_color="#A0A0A0")
        
        self.frame_dash.pack_forget()
        self.frame_cam.pack_forget()
        self.frame_share.pack_forget()
        
        if name == "dash":
            self.btn_dash.configure(fg_color="#1C2722", text_color="#00E676")
            self.frame_dash.pack(fill="both", expand=True)
        elif name == "cam":
            self.btn_cam.configure(fg_color="#1C2722", text_color="#00E676")
            self.frame_cam.pack(fill="both", expand=True)
        elif name == "share":
            self.btn_share.configure(fg_color="#1C2722", text_color="#00E676")
            self.frame_share.pack(fill="both", expand=True)

    def build_dash(self, parent):
        # DASHBOARD Header
        ctk.CTkLabel(parent, text="DASHBOARD", text_color="white", font=("Courier New", 12)).pack(anchor="w", padx=30, pady=(30, 10))
        
        # ONLINE Panel
        self.panel_online = ctk.CTkFrame(parent, fg_color="#121212", corner_radius=5)
        self.panel_online.pack(fill="x", padx=30, pady=5)
        
        top_online = ctk.CTkFrame(self.panel_online, fg_color="transparent")
        top_online.pack(fill="x", padx=15, pady=(15, 5))
        
        # --- FIX: Add dedicated toggle button ---
        self.btn_toggle_server = ctk.CTkButton(
            top_online, 
            text="STOP", 
            width=150, 
            height=50, 
            font=("Courier New", 12, "bold"),
            fg_color="#cf6679", 
            hover_color="#b00020",
            command=lambda: self.toggle_server()
        )
        self.btn_toggle_server.pack(side="right", padx=10)
        
        self.lbl_status = ctk.CTkLabel(top_online, text="● ONLINE", text_color="#00E676", font=("Courier New", 16, "bold"))
        self.lbl_status.pack(anchor="w")
        
        self.lbl_sub_status = ctk.CTkLabel(top_online, text="Accepting connections", text_color="#A0A0A0", font=("Courier New", 12))
        self.lbl_sub_status.pack(anchor="w")

        endpoint_frame = ctk.CTkFrame(self.panel_online, fg_color="#0B0B0B", corner_radius=3)
        endpoint_frame.pack(fill="x", padx=15, pady=(10, 15))
        
        ctk.CTkLabel(endpoint_frame, text="ENDPOINT", text_color="white", font=("Courier New", 10)).pack(anchor="w", padx=10, pady=(5,0))
        self.lbl_endpoint = ctk.CTkLabel(endpoint_frame, text=f"ws://...:{self.prefs.get('port', 8089)}", text_color="#00E676", font=("Courier New", 12))
        self.lbl_endpoint.pack(anchor="w", padx=10, pady=(0,10))

      

        # ACTIVITY LOG Section
        log_header_frame = ctk.CTkFrame(parent, fg_color="transparent")
        log_header_frame.pack(fill="x", padx=30, pady=(25, 5))
        ctk.CTkLabel(log_header_frame, text="ACTIVITY LOG", text_color="white", font=("Courier New", 10)).pack(side="left")

        self.log_box = ctk.CTkTextbox(parent, fg_color="#090909", border_color="#222", border_width=1, text_color="#00E676", font=("Courier New", 12), state="disabled")
        self.log_box.pack(fill="both", expand=True, padx=30, pady=(0, 30))

    

    def build_cam(self, parent):
        parent.columnconfigure(0, weight=0) 
        parent.columnconfigure(1, weight=1) 
        parent.rowconfigure(0, weight=1)
        
        panel_left = ctk.CTkFrame(parent, width=250, corner_radius=5, fg_color="#121212")
        panel_left.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        ctk.CTkLabel(panel_left, text="Background", font=("Courier New", 14, "bold"), text_color="white").pack(anchor="w", padx=15, pady=(15, 5))
        self.var_bg = ctk.StringVar(value="none")
        ctk.CTkRadioButton(panel_left, text="None", variable=self.var_bg, value="none", fg_color="#00E676", text_color="#A0A0A0", command=self.update_cam_settings).pack(anchor="w", padx=15, pady=2)
        ctk.CTkRadioButton(panel_left, text="Blur", variable=self.var_bg, value="blur", fg_color="#00E676", text_color="#A0A0A0", command=self.update_cam_settings).pack(anchor="w", padx=15, pady=2)
        ctk.CTkRadioButton(panel_left, text="Image", variable=self.var_bg, value="image", fg_color="#00E676", text_color="#A0A0A0", command=self.update_cam_settings).pack(anchor="w", padx=15, pady=2)
        ctk.CTkButton(panel_left, text="Select Image...", height=24, fg_color="#222", text_color="#00E676", command=self.select_bg_image).pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(panel_left, text="Adjustments", font=("Courier New", 14, "bold"), text_color="white").pack(anchor="w", padx=15, pady=(20, 5))
        self.sw_mirror = ctk.CTkSwitch(panel_left, text="Mirror Video", progress_color="#00E676", text_color="#A0A0A0", command=self.update_cam_settings)
        self.sw_mirror.pack(anchor="w", padx=15, pady=5)
        self.sw_flip = ctk.CTkSwitch(panel_left, text="Flip Vertical", progress_color="#00E676", text_color="#A0A0A0", command=self.update_cam_settings)
        self.sw_flip.pack(anchor="w", padx=15, pady=5)
        
        ctk.CTkLabel(panel_left, text="Brightness", text_color="#A0A0A0").pack(anchor="w", padx=15, pady=(10, 0))
        self.sld_bright = ctk.CTkSlider(panel_left, from_=-100, to=100, button_color="#00E676", progress_color="#00E676", command=lambda v: self.update_cam_settings())
        self.sld_bright.set(0)
        self.sld_bright.pack(fill="x", padx=15, pady=5)

        self.sw_preview = ctk.CTkSwitch(panel_left, text="Show Preview", progress_color="#00E676", text_color="#A0A0A0", command=self.update_cam_settings)
        self.sw_preview.select() 
        self.sw_preview.pack(anchor="w", padx=15, pady=(30, 5))

        panel_right = ctk.CTkFrame(parent, fg_color="#050505", border_color="#222", border_width=1, corner_radius=5)
        panel_right.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=20)
        self.lbl_preview = ctk.CTkLabel(panel_right, text="Waiting for connection...", text_color="#333", font=("Courier New", 14))
        self.lbl_preview.place(relx=0.5, rely=0.5, anchor="center")

    def build_share(self, parent):
        ctk.CTkLabel(parent, text="Clipboard Sync", font=("Courier New", 16, "bold"), text_color="white").pack(anchor="w", padx=30, pady=(30, 5))
        self.txt_clip = ctk.CTkTextbox(parent, height=80, fg_color="#121212", border_color="#222", border_width=1, text_color="white")
        self.txt_clip.pack(fill="x", padx=30, pady=5)
        ctk.CTkButton(parent, text="Send Text to Phone", fg_color="#000", border_color="#00E676", border_width=1, text_color="#00E676", hover_color="#111", command=self.send_text).pack(anchor="w", padx=30, pady=5)
        
        ctk.CTkLabel(parent, text="File Transfer", font=("Courier New", 16, "bold"), text_color="white").pack(anchor="w", padx=30, pady=(30, 5))
        ctk.CTkButton(parent, text="📤 Send File to Phone", fg_color="#000", border_color="#00E676", border_width=1, text_color="#00E676", hover_color="#111", command=self.send_file).pack(anchor="w", padx=30, pady=5)

        ctk.CTkLabel(parent, text="Received Files:", font=("Courier New", 12), text_color="#A0A0A0").pack(anchor="w", padx=30, pady=(20,0))
        self.files_frame = ctk.CTkScrollableFrame(parent, fg_color="#121212", corner_radius=5)
        self.files_frame.pack(fill="both", expand=True, padx=30, pady=5)
        
        ctk.CTkButton(parent, text="📂 Open Received Folder", fg_color="#222", text_color="#00E676", hover_color="#333", command=lambda: subprocess.Popen(['xdg-open', str(SAVE_DIR)])).pack(fill="x", padx=30, pady=(10, 30))
        self.refresh_file_list()

    def select_bg_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if path and self.server:
            try:
                img = cv2.imread(path)
                if img is not None:
                    self.server.bg_image_cv2 = img 
                    self.var_bg.set("image") 
                    self.update_cam_settings() 
            except Exception as e:
                pass
    
    def update_cam_settings(self):
        if self.server:
            self.server.cam_settings["mirror"] = bool(self.sw_mirror.get())
            self.server.cam_settings["flip"] = bool(self.sw_flip.get())
            self.server.cam_settings["brightness"] = int(self.sld_bright.get())
            self.server.cam_settings["background"] = self.var_bg.get()
            
            is_preview = bool(self.sw_preview.get())
            self.server.cam_settings["preview_on"] = is_preview
            
            if not is_preview:
                try:
                    self.lbl_preview.configure(image=None, text="Preview Paused")
                    self.current_image = None 
                except: pass
            else:
                try:
                    self.lbl_preview.configure(image=None, text="Resuming...")
                except: pass

    def toggle_vcam(self):
        if not self.server: return
        if not self.server._vcam_running:
            self.server.start_virtual_camera()
            self.server.show_preview = self.sw_preview.get()
        else:
            self.server.stop_virtual_camera()

        is_running = self.server._vcam_running
        self.btn_vcam.configure(
            fg_color="#00E676" if is_running else "#000",
            text_color="black" if is_running else "#00E676"
        )

    def toggle_audio(self):
        if not self.server: return
        if not self.server._streaming_audio:
            self.server.start_audio_streaming()
        else:
            self.server.stop_audio_streaming()

        is_running = self.server._streaming_audio
        self.btn_audio.configure(
            fg_color="#00E676" if is_running else "#000",
            text_color="black" if is_running else "#00E676"
        )

    def refresh_file_list(self):
        for widget in self.files_frame.winfo_children():
            widget.destroy()

        if not SAVE_DIR.exists(): return
        try:
            files = sorted(SAVE_DIR.glob("*"), key=os.path.getmtime, reverse=True)
            if not files:
                ctk.CTkLabel(self.files_frame, text="No files yet", text_color="#A0A0A0").pack(pady=10)
                return

            for f in files:
                if f.is_file():
                    row = ctk.CTkFrame(self.files_frame, fg_color="transparent")
                    row.pack(fill="x", pady=2)
                    ctk.CTkLabel(row, text=f"📄 {f.name}", text_color="white", anchor="w").pack(side="left", padx=5)
                    ctk.CTkButton(row, text="Open", width=60, height=24, fg_color="#222", text_color="#00E676", command=lambda p=f: subprocess.Popen(['xdg-open', str(p)])).pack(side="right", padx=5)
        except Exception as e:
            pass

    def toggle_server(self, force_start=False):
        if not self.is_running or force_start:
            self.server = UnifiedRemoteServer(port=self.prefs.get("port", 8089), update_queue=self.update_queue)
            self.server.gaming_mode = self.prefs.get("gaming_mode", True)
            self.server.start()
            self.is_running = True
            
            self.lbl_status.configure(text="● ONLINE", text_color="#00E676")
            self.lbl_sub_status.configure(text="Accepting connections", text_color="#A0A0A0")
            
            # --- FIX: Update button to say STOP ---
            if hasattr(self, 'btn_toggle_server'):
                self.btn_toggle_server.configure(text="STOP", fg_color="#cf6679", hover_color="#b00020")
            
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(('10.255.255.255', 1))
                ip = s.getsockname()[0]
                s.close()
                self.lbl_endpoint.configure(text=f"ws://{ip}:{self.prefs.get('port', 8089)}")
            except: pass
            
        else:
            self.lbl_status.configure(text="○ OFFLINE", text_color="#A0A0A0")
            self.lbl_sub_status.configure(text="Click to start server", text_color="#555")
            self.lbl_connected.configure(text="○ 0 connected") 
            
            # --- FIX: Update button to say START ---
            if hasattr(self, 'btn_toggle_server'):
                self.btn_toggle_server.configure(text="START", fg_color="#00E676", text_color="black", hover_color="#00c853")
            
            def stopper():
                if self.server: 
                    self.server.stop()
                self.is_running = False
                self.server = None
            threading.Thread(target=stopper, daemon=True).start()

    def send_text(self):
        if self.server: self.server.send_to_android("clipboard_text", self.txt_clip.get("1.0", "end").strip())

    def send_file(self):
        if self.server:
            paths = filedialog.askopenfilenames()
            if paths: threading.Thread(target=self.process_files, args=(paths,), daemon=True).start()

    def process_files(self, paths):
        for p in paths: self.server.send_file_to_phone_thread(p); time.sleep(0.5)

    def process_queue(self):
        try:
            while True:
                kind, data = self.update_queue.get_nowait()
                
                if kind == "video_frame":
                    self.current_image = ctk.CTkImage(light_image=data, dark_image=data, size=(640, 360))
                    if hasattr(self, 'lbl_preview'):
                        self.lbl_preview.configure(image=self.current_image, text="")
                        
                elif kind == "log":
                    # --- FIX: Use time.strftime instead of datetime to prevent crashes ---
                    ts = time.strftime("[%H:%M:%S]")
                    self.log_box.configure(state="normal")
                    self.log_box.insert("end", f"{ts}  {data}\n")
                    self.log_box.see("end")
                    self.log_box.configure(state="disabled")

                elif kind == "client_count":
                    # --- FIX: Update the connected devices label ---
                    if hasattr(self, 'lbl_connected'):
                        self.lbl_connected.configure(text=f"○ {data} connected")

                elif kind == "vcam_state":
                    is_running = data
                    if hasattr(self, 'btn_vcam'):
                        self.btn_vcam.configure(
                            fg_color="#00E676" if is_running else "#000",
                            text_color="black" if is_running else "#00E676"
                        )

                elif kind == "status_update":
                    pass 
                    
                elif kind == "clip":
                    self.txt_clip.delete("1.0", "end")
                    self.txt_clip.insert("end", data)
                    
                elif kind == "refresh_files":
                    self.refresh_file_list()
                    
        except queue.Empty:
            pass
        except Exception as e:
            # --- FIX: Print errors so they don't fail silently! ---
            print(f"UI Queue Error: {e}")
            pass
            
        self.root.after(15, self.process_queue)

    def open_settings(self): 
        SettingsDialog(self.root, self.prefs, self.save_preferences)

    def on_closing(self):
        try:
            if self.server: self.server.stop()
        except: pass
        os._exit(0)

if __name__ == "__main__":
    app = ServerGUI()
    app.root.mainloop()

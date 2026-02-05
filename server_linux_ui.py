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
APP_VERSION = "1.3-Linux-Merged"
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
        message = f"UNIFIED_REMOTE_SERVER:{self.port}".encode('utf-8')

        while self.running:
            targets = self.get_broadcast_addresses()
            for target in targets:
                try:
                    sock.sendto(message, (target, 8888))
                except:
                    pass
            time.sleep(1)
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
        self.discovery = DiscoveryServer(port=self.port)
        self.discovery.start()


        # --- AUDIO SETUP ---
        self.p = pyaudio.PyAudio()
        self.audio_stream = None
        # Settings match Android: 16kHz, Mono, 16-bit
        self.AUDIO_FORMAT = pyaudio.paInt16
        self.AUDIO_CHANNELS = 1
        self.AUDIO_RATE = 16000

        # 2. START UDP MOUSE
        self.udp_port = self.port + 1
        self.udp_mouse = UDPMouseServer(port=self.udp_port)
        self.udp_mouse.daemon = True
        self.udp_mouse.start()

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
                print("✅ Background Blur Engine Loaded")
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
            
            # ✅ NOTIFY GUI TO CHANGE BUTTON COLOR
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
        
        # ✅ NOTIFY GUI
        self._put("vcam_state", False)

    # --- UPDATED: DEBUG CAMERA LOGIC ---
    def _handle_video_frame(self, payload):
        """Optimized Video Handler with Auto-Start"""
        try:
            # ---------------------------------------------------------
            # ✅ AUTO-START LOGIC
            # ---------------------------------------------------------
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
            self.clients.add(websocket)
            self._put("log", f"✅ Client connected")
            try:
                async for message in websocket:
                    try:
                        # 🔍 DEBUG: Print the raw message type and first 50 chars
                        # print(f"👉 RAW MSG TYPE: {type(message)}")
                        
                        if isinstance(message, bytes): 
                            # Bytes are usually file data or audio chunks (ignore logging them)
                            continue 
                        
                        
                        # ---------------------------------------------------------
                        # ✅ LOGIC: Try JSON, Fallback to Text
                        # ---------------------------------------------------------
                        try:
                            data = json.loads(message)
                            # If it parses as JSON but is just a string (e.g. "Hello"), treating as data dict will fail later.
                            # So we check if 'data' is actually a dictionary.
                            if not isinstance(data, dict):
                                raise ValueError("Not a dictionary")
                                
                        except (json.JSONDecodeError, ValueError):
                         
                            self._handle_clipboard(message)
                            continue
                        # ---------------------------------------------------------

                        msg_type = data.get("type", "unknown")
                        payload = data.get("payload", "")
                        
                        

                        # Routing
                        if msg_type == "mouse_move": self._handle_mouse_move(payload)
                        elif msg_type == "mouse_click": self._handle_mouse_click(payload)
                        elif msg_type == "mouse_scroll": self._handle_mouse_scroll(payload)
                        elif msg_type == "key_press": self._handle_key_press(payload)
                        
                        # Audio
                        elif msg_type == "audio_start" or msg_type == "audio_control": 
                            if "start" in str(payload) or msg_type == "audio_start":
                                self.start_audio_streaming()
                            else:
                                self.stop_audio_streaming()
                        elif msg_type == "audio":
                            action = data.get("action") or (payload.get("action") if isinstance(payload, dict) else "")
                            if action == "start": self.start_audio_streaming()
                            elif action == "stop": self.stop_audio_streaming()

                        # Features
                        elif msg_type == "video_frame": self._handle_video_frame(payload)
                        elif msg_type == "audio_frame": self._handle_audio_frame(payload)
                        elif msg_type == "gamepad_state": self._handle_gamepad_state(payload, websocket)
                        elif msg_type == "display_request": self._handle_display_request(payload)
                        elif msg_type == "file_transfer": self._handle_file_transfer(payload)
                        
                        # Clipboard (Explicit JSON type)
                        elif msg_type == "clipboard" or msg_type == "text_transfer" or msg_type == "clipboard_text":
                            print("📋 JSON Clipboard detected") # DEBUG
                            self._handle_clipboard(payload)
                            
                        elif msg_type == "heartbeat": await websocket.send(json.dumps({"type": "heartbeat", "payload": "pong"}))
                        
                    except Exception as e: 
                        print(f"❌ Handler Error: {e}")
            finally:
                if websocket in self.clients: self.clients.remove(websocket)
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

    def _handle_display_request(self, payload):
        """Handles Start/Stop requests for Screen Mirroring"""
        try:
            data = json.loads(payload) if isinstance(payload, str) else payload
            action = data.get("action", "stop_display")
            
            if action == "start_display":
                if self._display_active: return
                
                # Get resolution (Default to HD if missing)
                width = int(data.get("width", 1280))
                height = int(data.get("height", 720))
                
                self._display_active = True
                
                threading.Thread(
                    target=self._display_stream_worker, 
                    args=(width, height), 
                    daemon=True
                ).start()
                
            elif action == "change_resolution":
                self._display_width = int(data.get("width", 1280))
                self._display_height = int(data.get("height", 720))
                self._put("log", f"🖥️ Res Changed: {self._display_width}x{self._display_height}")

            else: # "stop_display"
                self._display_active = False
                self._put("log", "🖥️ Display Streaming Stopped")

        except Exception as e:
            self._put("log", f"❌ Display Req Error: {e}")

    # --- SMART STREAM WORKER (Auto-Switching) ---
    def _display_stream_worker(self, width, height):
        try:
            import mss
            import pyautogui  # Needed for Virtual Mode cursor
            
            self._display_width = width
            self._display_height = height

            with mss.mss() as sct:
                # 1. DECIDE MODE: Virtual vs Mirror
                # If we have more than 2 monitors (0=All, 1=Main, 2=Virtual), implies Virtual exists.
                use_virtual_mode = len(sct.monitors) > 2
                target_monitor = None
                
                if use_virtual_mode:
                    # --- VIRTUAL MODE (Advanced) ---
                    # Find the monitor furthest to the right
                    max_left = 0
                    for m in sct.monitors[1:]:
                        if m["left"] > max_left:
                            max_left = m["left"]
                            target_monitor = m
                    
                    self._put("log", f"✅ Virtual Monitor Detected ({target_monitor['width']}x{target_monitor['height']})")
                    self._put("log", "✨ Mode: Extended Desktop (With Mouse Cursor)")
                else:
                    # --- MIRROR MODE (Simple Fallback) ---
                    # Uses the exact logic you requested for mirroring
                    target_monitor = sct.monitors[1]
                    self._put("log", "⚠️ No Virtual Monitor Found")
                    self._put("log", "🔄 Mode: Screen Mirroring")

                # 2. START STREAMING LOOP
                while self._display_active:
                    loop_start = time.time()
                    
                    # Capture
                    raw_img = sct.grab(target_monitor)
                    frame = np.array(raw_img)
                    
                    if frame.size == 0: continue

                    # OPTIONAL: Draw Cursor ONLY if in Virtual Mode
                    # (Mirror mode usually includes cursor naturally, or you strictly wanted the simple code)
                    if use_virtual_mode:
                        mx, my = pyautogui.position()
                        rel_x = mx - target_monitor["left"]
                        rel_y = my - target_monitor["top"]
                        
                        # Draw cursor if inside the virtual screen
                        if 0 <= rel_x < target_monitor["width"] and 0 <= rel_y < target_monitor["height"]:
                            cv2.circle(frame, (rel_x, rel_y), 15, (0, 0, 0), 4)       # Black Border
                            cv2.circle(frame, (rel_x, rel_y), 15, (255, 255, 255), -1) # White Fill

                    # Resize & Convert
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    frame = cv2.resize(frame, (self._display_width, self._display_height), interpolation=cv2.INTER_LINEAR)
                    
                    # Compress
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                    b64_data = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send
                    msg = json.dumps({
                        "type": "video_frame", 
                        "payload": b64_data
                    })
                    
                    if self._loop:
                        asyncio.run_coroutine_threadsafe(self._broadcast_text(msg), self._loop)
                    
                    # FPS Limit
                    process_time = time.time() - loop_start
                    sleep_time = max(0, 0.033 - process_time)
                    time.sleep(sleep_time)
                
        except ImportError:
            self._put("log", "❌ Error: 'mss' or 'pyautogui' missing.")
        except Exception as e:
            self._put("log", f"❌ Stream Error: {e}")
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
        try:
            import subprocess
            from pynput.keyboard import Controller, Key
            import pyautogui

            # 1. Parse Data
            if isinstance(payload, str):
                d = json.loads(payload)
            else:
                d = payload

            # Normalize key: remove spaces, lowercase
            raw_key = d.get("key", "").lower().replace(" ", "").replace("_", "") 
            modifiers = [m.lower() for m in d.get("modifiers", [])]
            action = d.get("action", "press") 

            # =========================================================
            # 1. SPECIAL LINUX COMMANDS (Brightness & Media)
            # xdotool is more reliable for these on Linux
            # =========================================================
            linux_special_map = {
                "brightnessup": "XF86MonBrightnessUp",
                "brightnessdown": "XF86MonBrightnessDown",
                "volumemute": "XF86AudioMute",
                "volumedown": "XF86AudioLowerVolume",
                "volumeup": "XF86AudioRaiseVolume",
                "playpause": "XF86AudioPlay",
                "nexttrack": "XF86AudioNext",
                "prevtrack": "XF86AudioPrev",
                "stop": "XF86AudioStop",
                "search": "XF86Search",
                "home": "Home",
                "end": "End",
                "pageup": "Page_Up",
                "pagedown": "Page_Down",
                "insert": "Insert",
                "delete": "Delete",
                "printscreen": "Print",
                "numlock": "Num_Lock",
                "scrolllock": "Scroll_Lock",
                "pause": "Pause",
                "menu": "Menu",
                "capslock": "Caps_Lock"
            }

            if raw_key in linux_special_map:
                key_code = linux_special_map[raw_key]
                # We use xdotool for these specific keys
                if action == "down":
                    subprocess.Popen(["xdotool", "keydown", key_code])
                elif action == "up":
                    subprocess.Popen(["xdotool", "keyup", key_code])
                else:
                    subprocess.Popen(["xdotool", "key", key_code])
                return

            # =========================================================
            # 2. NUMPAD MAPPING
            # =========================================================
            numpad_map = {
                "numpad0": "0", "num0": "0",
                "numpad1": "1", "num1": "1",
                "numpad2": "2", "num2": "2",
                "numpad3": "3", "num3": "3",
                "numpad4": "4", "num4": "4",
                "numpad5": "5", "num5": "5",
                "numpad6": "6", "num6": "6",
                "numpad7": "7", "num7": "7",
                "numpad8": "8", "num8": "8",
                "numpad9": "9", "num9": "9",
                "numpadenter": "enter", 
                "numpadadd": "+", 
                "numpadsubtract": "-", 
                "numpadmultiply": "*", 
                "numpaddivide": "/", 
                "numpaddecimal": "."
            }

            if raw_key in numpad_map:
                final_key = numpad_map[raw_key]
                if action == "down": pyautogui.keyDown(final_key)
                elif action == "up": pyautogui.keyUp(final_key)
                else: pyautogui.press(final_key)
                return

            # =========================================================
            # 3. WINDOWS / SUPER KEY
            # =========================================================
            if raw_key in ["win", "windows", "super", "meta", "cmd"]:
                if action == "up": return
                subprocess.Popen(["xdotool", "key", "Super_L"])
                return

            # =========================================================
            # 4. ARROWS
            # =========================================================
            arrow_map = {
                "up": "Up", "down": "Down", "left": "Left", "right": "Right",
                "↑": "Up", "↓": "Down", "←": "Left", "→": "Right"
            }
            if raw_key in arrow_map:
                k = arrow_map[raw_key]
                if action == "down": subprocess.Popen(["xdotool", "keydown", k])
                elif action == "up": subprocess.Popen(["xdotool", "keyup", k])
                else: subprocess.Popen(["xdotool", "key", k])
                return

            # =========================================================
            # 5. STANDARD KEYS (Fallback to PyAutoGUI)
            # =========================================================
            key_map = {
                "enter": "enter", "return": "enter",
                "backspace": "backspace", 
                "tab": "tab", "space": "space", "esc": "escape",
                "-": "-", "=": "=", "[": "[", "]": "]", "\\": "\\", 
                ";": ";", "'": "'", ",": ",", ".": ".", "/": "/", "`": "`"
            }

            final_key = key_map.get(raw_key, raw_key)
            
            # Handle modifiers sent from Android
            py_mods = [m for m in modifiers if m not in ["win", "meta", "super"]]

            if action == "down":
                if not py_mods: pyautogui.keyDown(final_key)
            elif action == "up":
                if not py_mods: pyautogui.keyUp(final_key)
            else:
                if py_mods:
                    pyautogui.hotkey(*py_mods + [final_key])
                else:
                    pyautogui.press(final_key)

        except Exception as e:
            print(f"Key Handler Error: {e}")

  
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
                        self._put("log", f"✅ Moved Stream #{current_id} to Virtual Mic")
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
                self._put("log", "✅ Virtual Microphone Created")
            
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

                self._put("log", f"✅ Sent: {filename}")

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
            self._put("log", f"🌐 WebSocket listening on {self.port}")
            self._loop.run_forever()
        except Exception as e: self._put("log", f"Err: {e}")
        finally: self._loop.close()

    def start(self):
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Aggressive Stop: Kills all loops immediately"""
        print("Stopping Server...")
        
        # 1. Stop External Services
        if hasattr(self, 'discovery'): self.discovery.stop()
        if hasattr(self, 'udp_mouse'): self.udp_mouse.stop()
        
        # 2. Kill Loops Immediately
        self._vcam_running = False
        self._streaming_audio = False
        self.fake_cam = None  # Break driver link
        
        # 3. Stop Audio
        self._handle_audio_stop()
        
        # 4. Stop Network Loop
        if self._loop and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except: pass

# ============================================
# PART 2: MODERN UI (CustomTkinter)
# ============================================

class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, parent, prefs, callback_save):
        super().__init__(parent)
        self.title("Configuration")
        self.geometry("500x720") # Increased height for new options
        self.resizable(False, False)
        self.prefs = prefs
        self.callback_save = callback_save
        self.transient(parent)
        
        # --- UI LAYOUT ---
        ctk.CTkLabel(self, text="Configuration", font=("Segoe UI", 24, "bold")).pack(pady=(25, 20))

        # 1. General Settings
        self.frame_gen = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_gen.pack(fill="x", padx=40, pady=(0, 10))
        ctk.CTkLabel(self.frame_gen, text="General", font=("Segoe UI", 14, "bold"), text_color="#00e676").pack(anchor="w")
        
        self.var_autostart = ctk.BooleanVar(value=prefs.get("autostart", False))
        ctk.CTkSwitch(self.frame_gen, text="Auto-start with Linux Login", variable=self.var_autostart).pack(anchor="w", pady=8)
        
        self.var_auto_server = ctk.BooleanVar(value=prefs.get("auto_server", False))
        ctk.CTkSwitch(self.frame_gen, text="Auto-start Server on launch", variable=self.var_auto_server).pack(anchor="w", pady=8)

        # 2. Input & Gaming
        self.frame_game = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_game.pack(fill="x", padx=40, pady=20)
        ctk.CTkLabel(self.frame_game, text="Input & Gaming", font=("Segoe UI", 14, "bold"), text_color="#00e676").pack(anchor="w")
        
        self.var_gaming = ctk.BooleanVar(value=prefs.get("gaming_mode", True))
        ctk.CTkSwitch(self.frame_game, text="Gaming Mode (Low Latency)", variable=self.var_gaming).pack(anchor="w", pady=5)
        
        # 3. UI Scaling (NEW SECTION)
        self.frame_ui = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_ui.pack(fill="x", padx=40, pady=10)
        ctk.CTkLabel(self.frame_ui, text="UI Scaling (Restart Required)", font=("Segoe UI", 14, "bold"), text_color="#00e676").pack(anchor="w")
        
        # Default to 1.4 if not set
        current_scale = prefs.get("scale", 1.8)
        self.opt_scale = ctk.CTkOptionMenu(self.frame_ui, 
                                           values=["1.0", "1.2", "1.4", "1.5", "1.6", "1.8", "2.0", "2.2"],
                                           command=None)
        self.opt_scale.set(str(current_scale))
        self.opt_scale.pack(fill="x", pady=5)

        # 4. Port Configuration
        self.frame_port = ctk.CTkFrame(self, fg_color="#212121", corner_radius=8)
        self.frame_port.pack(fill="x", padx=40, pady=10)
        ctk.CTkLabel(self.frame_port, text="Server Port:", font=("Segoe UI", 13, "bold")).pack(side="left", padx=20, pady=15)
        
        self.ent_port = ctk.CTkEntry(self.frame_port, width=80, justify="center", fg_color="#1a1a1a")
        self.ent_port.insert(0, str(prefs.get("port", 8080)))
        self.ent_port.pack(side="right", padx=20, pady=15)

        # Save Button
        ctk.CTkButton(self, text="Save & Close", command=self.save, height=50,
                      fg_color="#00e676", hover_color="#00c853", text_color="black").pack(side="bottom", fill="x", padx=40, pady=40)

        # Wait for window to render before grabbing focus (Prevents Linux Crash)
        self.after(100, self.safe_grab)

    def safe_grab(self):
        try:
            self.grab_set()
            self.focus_set()
        except: pass

    def save(self):
        try:
            port = int(self.ent_port.get())
            scale = float(self.opt_scale.get()) # Get scale value
            
            new_prefs = {
                "autostart": self.var_autostart.get(),
                "auto_server": self.var_auto_server.get(),
                "gaming_mode": self.var_gaming.get(),
                "port": port,
                "scale": scale # Save scale preference
            }
            # Send data back to main app
            self.callback_save(new_prefs)
            self.destroy()
        except ValueError:
            pass # Ignore invalid inputs







class ServerGUI:
    def __init__(self):
        # UI Scaling FIX for Linux
        self.prefs = {"autostart": False, "port": 8080, "scale": 1.4}
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, 'r') as f: 
                    self.prefs.update(json.load(f))
        except: pass

        # 2. Apply UI Scaling
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("green")
        
        scaling_factor = self.prefs.get("scale", 1.4)
        ctk.set_widget_scaling(scaling_factor)  
        ctk.set_window_scaling(scaling_factor)
        self.root = ctk.CTk()
        self.root.title(f"Use As Server (Linux) v{APP_VERSION}")
        self.root.geometry("1100x850") # Larger default window
        
        self.prefs = {"autostart": False, "port": 8080}
        self.load_preferences()
        self.update_queue = queue.Queue()
        self.server = None
        self.is_running = False
        self.tray_icon = None
        
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(100, self.process_queue)

    def load_preferences(self):
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, 'r') as f: self.prefs.update(json.load(f))
        except: pass

    def save_preferences(self, new_prefs):
        """Handles saving JSON and setting up Linux Autostart"""
        self.prefs.update(new_prefs)
        
        # 1. Save to JSON file
        try:
            with open(SETTINGS_FILE, 'w') as f: json.dump(self.prefs, f)
        except: pass
        
        # 2. Linux Autostart Logic (.desktop file)
        autostart_dir = Path.home() / ".config" / "autostart"
        dfile = autostart_dir / "useas_server.desktop"
        
        if self.prefs["autostart"]:
            try:
                autostart_dir.mkdir(parents=True, exist_ok=True)
                with open(dfile, "w") as f:
                    # We use sys.executable to ensure we run with the correct python (venv)
                    f.write(f"""[Desktop Entry]
Type=Application
Name=Use As Server
Exec={sys.executable} {os.path.abspath(__file__)}
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
""")
            except Exception as e: print(f"Autostart Error: {e}")
        elif dfile.exists():
            try: os.remove(dfile)
            except: pass
            
        # 3. Apply Gaming Mode immediately if server is running
        if self.server:
            self.server.gaming_mode = self.prefs["gaming_mode"]


    def minimize_to_tray(self):
        """Hides window and creates Tray Icon"""
        self.root.withdraw() # Hide window
        
        # Create a simple green icon
        image = Image.new('RGB', (64, 64), "#00e676")
        ImageDraw.Draw(image).rectangle((16, 16, 48, 48), fill="#1e1e1e")
        
        menu = (item('Restore', self.restore_from_tray), item('Quit', self.quit_app))
        self.tray_icon = pystray.Icon("UseAs", image, "Use As Server", menu)
        
        # Run tray in separate thread so it doesn't block
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def restore_from_tray(self, icon=None, item=None):
        """Restores window from tray"""
        if self.tray_icon: 
            self.tray_icon.stop()
            self.tray_icon = None
        
        # Schedule GUI update on main thread
        self.root.after(0, self.root.deiconify)

    def quit_app(self, icon=None, item=None):
        """Fully quits the app from the Tray"""
        if self.tray_icon: 
            self.tray_icon.stop()
        
        # Force kill
        self.root.after(0, lambda: os._exit(0))

    

    def setup_ui(self):
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # --- SIDEBAR ---
        self.sidebar = ctk.CTkFrame(self.root, width=140, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        # Logo
        ctk.CTkLabel(self.sidebar, text="Use As\nLinux", font=("Segoe UI", 20, "bold"), text_color="#00e676").pack(pady=(30, 20))
        
        # Navigation Buttons
        self.btn_dash = ctk.CTkButton(self.sidebar, text="Dashboard", fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), 
                      command=lambda: self.select_tab("Dashboard"))
        self.btn_dash.pack(pady=10, padx=20, fill="x")
        
        self.btn_cam = ctk.CTkButton(self.sidebar, text="Camera", fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), 
                      command=lambda: self.select_tab("Camera"))
        self.btn_cam.pack(pady=10, padx=20, fill="x")
        
        self.btn_share = ctk.CTkButton(self.sidebar, text="Sharing", fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), 
                      command=lambda: self.select_tab("Sharing"))
        self.btn_share.pack(pady=10, padx=20, fill="x")

        # --- MINIMIZE TO TRAY BUTTON (Always Visible) ---
        # I placed it at the bottom of the sidebar for clean layout
        self.btn_tray = ctk.CTkButton(self.sidebar, text="⬇ Minimize to Tray", fg_color="#2d2d2d", hover_color="#333",
                                      command=self.minimize_to_tray)
        self.btn_tray.pack(side="bottom", pady=20, padx=20, fill="x")

        # --- MAIN CONTENT AREA ---
        self.tabview = ctk.CTkTabview(self.root, fg_color="transparent")
        self.tabview.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")
        self.tabview.add("Dashboard")
        self.tabview.add("Camera")
        self.tabview.add("Sharing")
        
        # Build Tabs
        self.build_dash(self.tabview.tab("Dashboard"))
        self.build_cam(self.tabview.tab("Camera"))
        self.build_share(self.tabview.tab("Sharing"))
        
        # Select default
        self.select_tab("Dashboard")

    def select_tab(self, name):
        self.tabview.set(name)

    def build_dash(self, parent):
        # 1. Start/Stop Button
        self.btn_start = ctk.CTkButton(parent, text="START SERVER", height=50, font=("Segoe UI", 16, "bold"),
                                     command=self.toggle_server, fg_color="#00e676", text_color="black")
        self.btn_start.pack(pady=40, padx=40, fill="x")
        
        # 2. Status Labels
        self.lbl_status = ctk.CTkLabel(parent, text="🔴 Offline", font=("Segoe UI", 18, "bold"))
        self.lbl_status.pack()
        
        self.ent_ip = ctk.CTkEntry(parent, placeholder_text="IP Address", justify="center")
        self.ent_ip.pack(pady=10)
        
        # 3. BUTTON ROW (Minimize + Settings)
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=20, padx=40)
        
        # Minimize Button (Packed Left)
        ctk.CTkButton(row, text="⬇ Minimize to Tray", command=self.minimize_to_tray, 
                      fg_color="#333", hover_color="#444").pack(side="left", expand=True, fill="x", padx=(0, 5))
        
        # Settings Button (Packed Right, so it ends up to the right of Minimize)
        ctk.CTkButton(row, text="⚙ Settings", command=self.open_settings, 
                      fg_color="#333", hover_color="#444").pack(side="right", expand=True, fill="x", padx=(5, 0))
        
        # 4. Logs
        self.log_box = ctk.CTkTextbox(parent)
        self.log_box.pack(fill="both", expand=True, padx=20, pady=10)

    def build_cam(self, parent):
        # Grid Layout: 2 Columns
        parent.columnconfigure(0, weight=0) # Left Sidebar (Fixed width)
        parent.columnconfigure(1, weight=1) # Right Preview (Expand)
        parent.rowconfigure(0, weight=1)

        # ============================================
        # 1. LEFT SIDEBAR (Controls)
        # ============================================
        panel_left = ctk.CTkFrame(parent, width=250, corner_radius=0, fg_color="transparent")
        panel_left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # --- Background Section ---
        ctk.CTkLabel(panel_left, text="Background", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 5))
        
        # Use 'value' to match the checks in _handle_video_frame
        self.var_bg = ctk.StringVar(value="none")
        
        # Add 'command=self.update_cam_settings' so it updates INSTANTLY when clicked
        ctk.CTkRadioButton(panel_left, text="None", variable=self.var_bg, value="none", 
                          command=self.update_cam_settings).pack(anchor="w", pady=2)
                          
        ctk.CTkRadioButton(panel_left, text="Blur", variable=self.var_bg, value="blur", 
                          command=self.update_cam_settings).pack(anchor="w", pady=2)
                          
        ctk.CTkRadioButton(panel_left, text="Image", variable=self.var_bg, value="image", 
                          command=self.update_cam_settings).pack(anchor="w", pady=2)
        
        ctk.CTkButton(panel_left, text="Select Image...", height=24, fg_color="#333", 
                      command=self.select_bg_image).pack(fill="x", pady=5)

        # --- Adjustments Section ---
        ctk.CTkLabel(panel_left, text="Adjustments", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(20, 5))
        
        # Mirror Switch
        self.sw_mirror = ctk.CTkSwitch(panel_left, text="Mirror Video", command=self.update_cam_settings)
        self.sw_mirror.pack(anchor="w", pady=5)
        
        # Flip Switch
        self.sw_flip = ctk.CTkSwitch(panel_left, text="Flip Vertical", command=self.update_cam_settings)
        self.sw_flip.pack(anchor="w", pady=5)
        
        # Brightness Slider
        ctk.CTkLabel(panel_left, text="Brightness").pack(anchor="w", pady=(10, 0))
        self.sld_bright = ctk.CTkSlider(panel_left, from_=-100, to=100, command=lambda v: self.update_cam_settings())
        self.sld_bright.set(0)
        self.sld_bright.pack(fill="x", pady=5)

        # Resolution Dropdown (Visual only for now, phone dictates res usually)
        ctk.CTkLabel(panel_left, text="Resolution").pack(anchor="w", pady=(10, 0))
        self.opt_res = ctk.CTkOptionMenu(panel_left, values=["1280x720 (16:9)", "1920x1080 (16:9)", "640x480 (4:3)"])
        self.opt_res.pack(fill="x", pady=5)

        # --- Output Control Section ---
        ctk.CTkLabel(panel_left, text="Output Control", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(20, 5))
        
        # Preview Toggle (Crucial Request)
        self.sw_preview = ctk.CTkSwitch(panel_left, text="Show Preview", command=self.update_cam_settings)
        self.sw_preview.select() # Default ON
        self.sw_preview.pack(anchor="w", pady=5)

        # Buttons
        ctk.CTkButton(panel_left, text="Start Virtual Camera", fg_color="#3B8ED0", 
                      command=lambda: self.server.start_virtual_camera() if self.server else None).pack(fill="x", pady=5)
        
        # Audio Button (Moved here to match "Controls")
        ctk.CTkButton(panel_left, text="Start Audio Stream", fg_color="#3B8ED0",
                      command=lambda: self.server.start_audio_streaming() if self.server else None).pack(fill="x", pady=5)


        # ============================================
        # 2. RIGHT PANEL (Preview)
        # ============================================
        panel_right = ctk.CTkFrame(parent, fg_color="#000", corner_radius=10)
        panel_right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Preview Label (Holds the image)
        self.lbl_preview = ctk.CTkLabel(panel_right, text="Waiting for connection...", text_color="gray")
        self.lbl_preview.place(relx=0.5, rely=0.5, anchor="center")

    
    
    
    
    # --- ADD THIS NEW FUNCTION TO ServerGUI ---
    def select_bg_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if path and self.server:
            try:
                # Load image immediately into OpenCV format for the server
                img = cv2.imread(path)
                if img is not None:
                    self.server.bg_image_cv2 = img # Store directly in server instance
                    self.var_bg.set("image") # Auto-select the radio button
                    self.update_cam_settings() # Apply changes
                    print(f"✅ Background loaded: {path}")
            except Exception as e:
                print(f"❌ Failed to load image: {e}")
    
    
    
    def update_cam_settings(self):
        """Sends UI settings to the Server Logic"""
        if self.server:
            # ... (your existing settings updates for mirror/flip/etc) ...
            self.server.cam_settings["mirror"] = bool(self.sw_mirror.get())
            self.server.cam_settings["flip"] = bool(self.sw_flip.get())
            self.server.cam_settings["brightness"] = int(self.sld_bright.get())
            self.server.cam_settings["background"] = self.var_bg.get()
            
            # Preview Toggle Logic
            is_preview = bool(self.sw_preview.get())
            self.server.cam_settings["preview_on"] = is_preview
            
            if not is_preview:
                # Turn OFF: Clear image, show "Paused"
                try:
                    self.lbl_preview.configure(image=None, text="Preview Paused")
                    self.current_image = None # Clear memory
                except: pass
            else:
                # Turn ON: Show "Resuming" immediately
                # The video will overwrite this text in a split second
                try:
                    self.lbl_preview.configure(image=None, text="Resuming...")
                except: pass
    # --- LOGIC HANDLERS ---
    def update_preview_setting(self):
        """Live update for the preview window toggle"""
        if self.server:
            self.server.show_preview = self.var_preview.get()
            # If user turns OFF preview while running, close window immediately
            if not self.server.show_preview:
                cv2.destroyAllWindows()

    def update_status_indicators(self, is_video, is_audio):
        """Called by the Server thread to update UI labels"""
        if is_video and is_audio:
            self.lbl_cam_status.configure(text="🔴 Live (Video+Audio)", text_color="#cf6679")
        elif is_video:
            self.lbl_cam_status.configure(text="📷 Video Live", text_color="#00e676")
        elif is_audio:
            self.lbl_cam_status.configure(text="🎤 Audio Live", text_color="#00e676")
        else:
            self.lbl_cam_status.configure(text="⚪ Waiting for Phone...", text_color="gray")

    def toggle_vcam(self):
        if not self.server: return
        
        # 1. Toggle Server State
        if not self.server._vcam_running:
            self.server.start_virtual_camera()
            self.server.show_preview = self.var_preview.get() # Apply checkbox state
        else:
            self.server.stop_virtual_camera()

        # 2. Update UI
        is_running = self.server._vcam_running
        
        self.btn_vcam.configure(
            text="Stop Virtual Camera" if is_running else "Start Virtual Camera",
            fg_color="#cf6679" if is_running else "#3B8ED0", 
            hover_color="#b00020" if is_running else "#1F6AA5"
        )
        
        # Sync Quick Button (on Dashboard) if it exists
        if hasattr(self, 'btn_quick_cam'):
            self.btn_quick_cam.configure(
                text="Stop Camera" if is_running else "Start Virtual Camera",
                fg_color="#cf6679" if is_running else ["#3B8ED0", "#1F6AA5"]
            )

        self.update_status_label()

    def toggle_audio(self):
        if not self.server: return

        # 1. Toggle Server State
        if not self.server._streaming_audio:
            self.server.start_audio_streaming()
        else:
            self.server.stop_audio_streaming()

        # 2. Update UI
        is_running = self.server._streaming_audio
        
        self.btn_audio.configure(
            text="Stop Audio Receiver" if is_running else "Start Audio Receiver",
            fg_color="#cf6679" if is_running else "#3B8ED0",
            hover_color="#b00020" if is_running else "#1F6AA5"
        )
        
        self.update_status_label()

    def update_status_label(self):
        """Updates the small label in the top right corner"""
        v_on = self.server._vcam_running if self.server else False
        a_on = self.server._streaming_audio if self.server else False

        if v_on and a_on:
            self.lbl_cam_status.configure(text="🔴 Live (Video+Audio)", text_color="#cf6679")
        elif v_on:
            self.lbl_cam_status.configure(text="📷 Video Live", text_color="#00e676")
        elif a_on:
            self.lbl_cam_status.configure(text="🎤 Audio Live", text_color="#00e676")
        else:
            self.lbl_cam_status.configure(text="⚪ Inactive", text_color="gray")

    def build_share(self, parent):
        # 1. Clipboard Section
        ctk.CTkLabel(parent, text="Clipboard Sync", font=("Segoe UI", 16, "bold")).pack(pady=(10, 5))
        self.txt_clip = ctk.CTkTextbox(parent, height=60)
        self.txt_clip.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(parent, text="Send Text to Phone", command=self.send_text).pack(fill="x", padx=10, pady=5)
        
        # 2. File Transfer Section
        ctk.CTkLabel(parent, text="File Transfer", font=("Segoe UI", 16, "bold")).pack(pady=(20, 5))
        
        # Send Button
        ctk.CTkButton(parent, text="📤 Send File to Phone", fg_color="#3B8ED0", 
                      command=self.send_file).pack(fill="x", padx=10, pady=5)

        # 3. Received Files List (NEW)
        ctk.CTkLabel(parent, text="Received Files:", font=("Segoe UI", 12)).pack(anchor="w", padx=15, pady=(10,0))
        
        self.files_frame = ctk.CTkScrollableFrame(parent, height=200, label_text="Downloads Folder")
        self.files_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Open Folder Button
        ctk.CTkButton(parent, text="📂 Open Received Folder", fg_color="#2d2d2d", 
                      command=lambda: subprocess.Popen(['xdg-open', str(SAVE_DIR)])).pack(fill="x", padx=10, pady=10)

        # Populate list immediately
        self.refresh_file_list()

    # --- NEW: REFRESH FILE LIST ---
    def refresh_file_list(self):
        # Clear existing buttons
        for widget in self.files_frame.winfo_children():
            widget.destroy()

        # Scan folder
        if not SAVE_DIR.exists(): return
        
        try:
            files = sorted(SAVE_DIR.glob("*"), key=os.path.getmtime, reverse=True)
            if not files:
                ctk.CTkLabel(self.files_frame, text="No files yet").pack(pady=10)
                return

            for f in files:
                if f.is_file():
                    # Create a row for each file
                    row = ctk.CTkFrame(self.files_frame, fg_color="transparent")
                    row.pack(fill="x", pady=2)
                    
                    # File Icon/Name
                    ctk.CTkLabel(row, text=f"📄 {f.name}", anchor="w").pack(side="left", padx=5)
                    
                    # Open Button
                    ctk.CTkButton(row, text="Open", width=60, height=24, 
                                  command=lambda p=f: subprocess.Popen(['xdg-open', str(p)])).pack(side="right", padx=5)
        except Exception as e:
            print(f"List Error: {e}")

    def toggle_server(self):
        if not self.is_running:
            # --- START (Normal Logic) ---
            self.server = UnifiedRemoteServer(port=self.prefs.get("port", 8080), update_queue=self.update_queue)
            # Apply Gaming Mode setting on start
            self.server.gaming_mode = self.prefs.get("gaming_mode", True)
            self.server.start()
            self.is_running = True
            
            self.btn_start.configure(text="STOP SERVER", fg_color="#cf6679", hover_color="#b00020")
            self.lbl_status.configure(text="🟢 Online", text_color="#00e676")
            
            # Show IP Logic
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(('10.255.255.255', 1))
                ip = s.getsockname()[0]
                s.close()
                self.ent_ip.configure(state="normal")
                self.ent_ip.delete(0, "end")
                self.ent_ip.insert(0, f"ws://{ip}:{self.prefs.get('port')}")
                self.ent_ip.configure(state="readonly")
            except: pass
            
        else:
            # --- STOP (Background Thread Logic) ---
            self.btn_start.configure(text="Stopping...", state="disabled")
            
            def stopper():
                if self.server: 
                    self.server.stop()
                
                # Update UI back on main thread logic is safe here
                self.is_running = False
                self.btn_start.configure(text="START SERVER", state="normal", fg_color="#00e676", hover_color="#00c853")
                self.lbl_status.configure(text="🔴 Offline", text_color="gray")
                self.server = None
            
            # Run the stop function in a thread so UI doesn't hang
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
                    self.log_box.insert("end", f"> {data}\n")
                    self.log_box.see("end")

                # ✅ NEW: Handle Button Updates from Auto-Start
                elif kind == "vcam_state":
                    is_running = data
                    # Update Camera Tab Button
                    if hasattr(self, 'btn_vcam'):
                        self.btn_vcam.configure(
                            text="Stop Virtual Camera" if is_running else "Start Virtual Camera",
                            fg_color="#cf6679" if is_running else "#3B8ED0", 
                            hover_color="#b00020" if is_running else "#1F6AA5"
                        )
                    # Update Status Label
                    self.update_status_label()

                elif kind == "status_update":
                    if hasattr(self, 'lbl_cam_status'):
                        self.update_status_indicators(data["video"], data["audio"])
                    
                elif kind == "clip":
                    self.txt_clip.delete("1.0", "end")
                    self.txt_clip.insert("end", data)
                    
                elif kind == "refresh_files":
                    self.refresh_file_list()
                    
        except queue.Empty:
            pass
        except Exception as e:
            pass
            
        self.root.after(15, self.process_queue)

    def minimize_to_tray(self):
        """Hides window and creates Tray Icon with Double-Click Restore"""
        self.root.withdraw() # Hide window
        
        # Create a simple green icon
        image = Image.new('RGB', (64, 64), "#00e676")
        ImageDraw.Draw(image).rectangle((16, 16, 48, 48), fill="#1e1e1e")
        
        # Define Menu: Restore and Quit
        menu = (
            item('Restore', self.restore_from_tray, default=True), # default=True makes it the double-click action
            item('Quit', self.quit_app)
        )
        
        # Create Icon
        # action=... binds the left-click/double-click to restore
        self.tray_icon = pystray.Icon("UseAs", image, "Use As Server", menu)
        
        # Run tray in separate thread
        threading.Thread(target=self.tray_icon.run, daemon=True).start()
    

    def restore(self, icon=None, item=None):
        if self.tray_icon: self.tray_icon.stop()
        self.root.after(0, self.root.deiconify)

    def quit(self, icon=None, item=None):
        if self.tray_icon: self.tray_icon.stop()
        self.root.after(0, self.on_closing)

    def open_settings(self): 
        SettingsDialog(self.root, self.prefs, self.save_preferences)

    def on_closing(self):
        """Force kills the application immediately"""
        try:
            # Try to stop nicely first (optional)
            if self.server: self.server.stop()
        except: 
            pass
        
        # Hard Exit: Don't wait for threads to join.
        # This solves the "App won't close" issue.
        print("Force Exiting...")
        os._exit(0)

if __name__ == "__main__":
    app = ServerGUI()
    app.root.mainloop()

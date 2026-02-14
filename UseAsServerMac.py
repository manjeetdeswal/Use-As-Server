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
import subprocess
import base64
from pynput.keyboard import Controller, Key

# --- OPTIONAL DEPENDENCIES CHECKS ---
try:
    import pyvirtualcam
    HAS_VCAM = True
except ImportError:
    HAS_VCAM = False
    print("⚠️ 'pyvirtualcam' not found. Camera features disabled.")

try:
    import pyaudio
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("⚠️ 'pyaudio' not found. Audio streaming disabled.")

try:
    from cvzone.SelfiSegmentationModule import SelfiSegmentation
    HAS_SEGMENTATION = True
    print("AI Background Removal loaded successfully!")
except ImportError:
    HAS_SEGMENTATION = False
    print("⚠️ 'cvzone' or 'mediapipe' missing. Background effects disabled.")

# --- CONSTANTS ---

APP_VERSION = "1.0-macOS"
SETTINGS_DIR = Path.home() / "Library" / "Application Support" / "UseAsServer"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"
SAVE_DIR = Path.home() / "Downloads" / "UseAs_Received"

# Ensure directories exist
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# PART 1:  SERVER BACKEND (MACOS OPTIMIZED)
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
                        # macOS often requires smaller multipliers for smooth movement
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
        self.clients = set()
        self._loop = None
        self._ws_server = None
        self._thread = None
        self.update_queue = update_queue or queue.Queue()
        self.gamepad_history = {}
        
        # Keyboard Controller for media keys
        self.keyboard = Controller()

        # 1. START DISCOVERY
        self.discovery = DiscoveryServer(port=self.port)
        self.discovery.start()

        # --- AUDIO SETUP ---
        self.p = pyaudio.PyAudio()
        self.audio_stream = None
        self.AUDIO_FORMAT = pyaudio.paInt16
        self.AUDIO_CHANNELS = 1
        self.AUDIO_RATE = 16000

        # 2. START UDP MOUSE
        self.udp_port = self.port + 1
        self.udp_mouse = UDPMouseServer(port=self.udp_port)
        self.udp_mouse.daemon = True
        self.udp_mouse.start()

        self._broadcast_queue = asyncio.Queue()
        self.show_preview = True
        self.cam_width = 0   
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
                print(f"⚠️ Background Blur Disabled: {e}")
                self.segmentor = None

        self._vcam = None 
        self._vcam_running = False
        self._last_frame = None
        self._streaming_audio = False
        self._display_active = False
        self._display_width = 1920
        self._display_height = 1080

    def _put(self, kind, payload):
        try: self.update_queue.put((kind, payload))
        except: pass

    # --- CAM LOGIC (OBS Virtual Camera) ---
    def start_virtual_camera(self):
        if not HAS_VCAM:
            self._put("log", "❌ Error: 'pyvirtualcam' missing.")
            return
        if self._vcam_running: return

        try:
            # macOS: We rely on OBS Virtual Camera or similar plugins
            # 'obs' backend is usually auto-detected if OBS is installed
            self._vcam = pyvirtualcam.Camera(
                width=1280, height=720, fps=30, 
                fmt=pyvirtualcam.PixelFormat.RGB,
                # backend='obs' # Auto-detection usually works best on Mac
            )
            
            self._vcam_running = True
            self.vcam_error_shown = False 
            
            self._put("log", f"📹 VCam Started via {self._vcam.backend}")
            self._put("log", "ℹ️ Ensure OBS Virtual Camera is running if using OBS backend")
            
            self._put("vcam_state", True) 
            threading.Thread(target=self._send_frames_loop, daemon=True).start()

        except Exception as e:
            self._put("log", f"❌ VCam Error: {e}")
            self._put("log", "💡 Install OBS + Virtual Camera Plugin")
            
            self._vcam_running = False
            self._put("vcam_state", False)

    def stop_virtual_camera(self):
        self._vcam_running = False
        time.sleep(0.1) 
        if self._vcam:
            try: self._vcam.close()
            except: pass
            self._vcam = None
        
        self._put("log", "📹 Virtual Camera Stopped")
        self._put("vcam_state", False)

    def _handle_video_frame(self, payload):
        try:
            # Auto-start logic for OBS Virtual Camera
            if not getattr(self, "_vcam_running", False) and HAS_VCAM:
                if not getattr(self, "vcam_error_shown", False):
                    print("📷 Auto-starting Virtual Camera...")
                    self.start_virtual_camera()
                    if not getattr(self, "_vcam_running", False):
                        self.vcam_error_shown = True

            if isinstance(payload, str): frame_data = json.loads(payload)
            else: frame_data = payload

            b64_data = frame_data.get('data')
            if not b64_data: return

            # Decode from Base64
            img_bytes = base64.b64decode(b64_data)
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None: return

            # --- 1. ORIENTATION & RESOLUTION ---
            # We removed the 'is_front' and 'rotation' logic here because your 
            # Android Kotlin code ALREADY rotated and mirrored the image perfectly!
            
            # We also removed the hardcoded 1280x720 resize. The frame is now 
            # exactly the size you requested in the Android App settings!

            # PC-side manual overrides (if you click flip/mirror on the Mac UI)
            if self.cam_settings.get("mirror", False): frame = cv2.flip(frame, 1)
            if self.cam_settings.get("flip", False): frame = cv2.flip(frame, 0)

            # --- 2. EFFECTS (Brightness & Background Blur) ---
            if self.cam_settings.get("brightness", 0) != 0:
                frame = cv2.convertScaleAbs(frame, alpha=1, beta=self.cam_settings["brightness"])

            if self.cam_settings.get("background", "none") != "none":
                frame = self._process_background(frame)

            # --- 3. SEND TO VIRTUAL CAMERA THREAD ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._last_frame = rgb_frame

            # --- 4. PYTHON UI PREVIEW ---
            if self.cam_settings.get("preview_on", True):
                h, w = rgb_frame.shape[:2]
                small_w = 320
                small_h = int(h * (small_w / w))
                
                # Make sure dimensions are valid before resizing preview
                if small_w > 0 and small_h > 0:
                    preview_img = cv2.resize(rgb_frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
                    pil_img = Image.fromarray(preview_img)
                    self._put("video_frame", pil_img)

        except Exception as e:
            print(f"❌ VIDEO ERROR: {e}")

    def _stop_camera(self):
        try:
            cv2.destroyAllWindows()
            self._put("log", "📷 Camera Preview Closed")
        except: pass

    def _send_frames_loop(self):
        waiting_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        waiting_frame[:] = (0, 0, 100) # Dark Blue RGB
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
            self._put("log", f" Client connected")
            try:
                async for message in websocket:
                    try:
                        if isinstance(message, bytes): 
                            continue 
                        
                        try:
                            data = json.loads(message)
                            if not isinstance(data, dict): raise ValueError("Not dict")
                        except (json.JSONDecodeError, ValueError):
                            self._handle_clipboard(message)
                            continue

                        msg_type = data.get("type", "unknown")
                        payload = data.get("payload", "")

                        if msg_type == "mouse_move": self._handle_mouse_move(payload)
                        elif msg_type == "mouse_click": self._handle_mouse_click(payload)
                        elif msg_type == "mouse_scroll": self._handle_mouse_scroll(payload)
                        elif msg_type == "key_press": self._handle_key_press(payload)
                        
                        elif msg_type == "audio_start" or msg_type == "audio_control": 
                            if "start" in str(payload) or msg_type == "audio_start":
                                self.start_audio_streaming()
                            else:
                                self.stop_audio_streaming()
                        elif msg_type == "audio":
                            action = data.get("action") or (payload.get("action") if isinstance(payload, dict) else "")
                            if action == "start": self.start_audio_streaming()
                            elif action == "stop": self.stop_audio_streaming()

                        elif msg_type == "video_frame": self._handle_video_frame(payload)
                        elif msg_type == "audio_frame": self._handle_audio_frame(payload)
                        elif msg_type == "gamepad_state": self._handle_gamepad_state(payload, websocket)
                        elif msg_type == "display_request": self._handle_display_request(payload)
                        elif msg_type == "file_transfer": self._handle_file_transfer(payload)
                        
                        elif msg_type in ["clipboard", "text_transfer", "clipboard_text"]:
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
        # 1. Check if the AI segmentor is actually loaded
        if not getattr(self, 'segmentor', None) or not HAS_SEGMENTATION: 
            return frame
            
        try:
            mode = self.cam_settings.get("background", "none")
            if mode == "none": 
                return frame
                
            h, w = frame.shape[:2]

            # --- BLUR MODE ---
            if mode == "blur":
                # Increased kernel size (95,95) for a much deeper, professional blur
                img_bg = cv2.GaussianBlur(frame, (95, 95), 0)
                return self.segmentor.removeBG(frame, imgBg=img_bg, cutThreshold=0.5)
                
            # --- CUSTOM IMAGE MODE ---
            elif mode == "image":
                if hasattr(self, 'bg_image_cv2') and self.bg_image_cv2 is not None:
                    
                    # FIX 1: Strip Alpha Channel 
                    # If you uploaded a transparent PNG, convert it to a flat BGR image
                    if len(self.bg_image_cv2.shape) == 3 and self.bg_image_cv2.shape[2] == 4:
                        self.bg_image_cv2 = cv2.cvtColor(self.bg_image_cv2, cv2.COLOR_BGRA2BGR)

                    # FIX 2: Dynamic Dimension Matching
                    # If you rotate your phone, force the background to stretch and match the new shape
                    bg_h, bg_w = self.bg_image_cv2.shape[:2]
                    if bg_h != h or bg_w != w:
                        self.bg_image_cv2 = cv2.resize(self.bg_image_cv2, (w, h), interpolation=cv2.INTER_AREA)

                    # Apply the image background
                    return self.segmentor.removeBG(frame, imgBg=self.bg_image_cv2, cutThreshold=0.5)
                else:
                    # Fallback to a solid Green Screen if you click "Image" but haven't uploaded one yet
                    return self.segmentor.removeBG(frame, imgBg=(0, 255, 0), cutThreshold=0.5)

        except Exception as e:
            # Replaced the silent fail so we can actually see the error if it breaks!
            print(f"❌ Background Effect Error: {e}")
            return frame
            
        return frame

    # --- SCREEN MIRRORING ---
    def _handle_display_request(self, payload):
        try:
            data = json.loads(payload) if isinstance(payload, str) else payload
            action = data.get("action", "stop_display")
            
            if action == "start_display":
                if self._display_active: return
                
                # Dynamically read what the Android app requested
                width = int(data.get("width", 1280))
                height = int(data.get("height", 720))
                fps = int(data.get("fps", 30)) # <--- NEW: Read FPS from Android
                
                self._display_active = True
                
                # Pass the FPS to the worker thread
                threading.Thread(
                    target=self._display_stream_worker, 
                    args=(width, height, fps), 
                    daemon=True
                ).start()
                
            elif action == "change_resolution":
                self._display_width = int(data.get("width", 1280))
                self._display_height = int(data.get("height", 720))
                if "fps" in data:
                    self._display_fps = int(data["fps"])
                self._put("log", f"🖥️ Stream Updated: {self._display_width}x{self._display_height} @ {getattr(self, '_display_fps', 30)} FPS")

            else:
                self._display_active = False
                self._put("log", "🖥️ Display Streaming Stopped")

        except Exception as e:
            self._put("log", f"❌ Display Req Error: {e}")

    def _display_stream_worker(self, width, height, fps=30):
        try:
            self._display_width = width
            self._display_height = height
            self._display_fps = fps # Store requested FPS
            
            with mss.mss() as sct:
                self._put("log", f"🔍 Detected {len(sct.monitors)-1} display(s).")
                
                is_extended = False
                target_monitor = sct.monitors[1] 
                
                if len(sct.monitors) > 2:
                    for m in sct.monitors[1:]:
                        if m["left"] != 0 or m["top"] != 0:
                            target_monitor = m
                            is_extended = True
                            break
                    
                    if is_extended: self._put("log", f"✨ Mode: Extended Display ({width}x{height} @ {fps} FPS)")
                    else: self._put("log", "⚠️ 2 displays found, but macOS is Mirroring")

                if not is_extended and len(sct.monitors) <= 2:
                    self._put("log", f"🔄 Mode: Screen Mirroring ({width}x{height} @ {fps} FPS)")

                # JPEG Quality (Lower is faster. 45 is a great balance for 60fps)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 45]

                while self._display_active:
                    loop_start = time.time()
                    
                    # 1. Grab Frame
                    raw_img = sct.grab(target_monitor)
                    frame = np.array(raw_img)
                    
                    if frame.size == 0: continue

                    # 2. Fast Alpha Drop + Fix the Memory Crash!
                    frame = np.ascontiguousarray(frame[:, :, :3])

                    # 3. Draw fake cursor on extended display
                    if is_extended:
                        mx, my = pyautogui.position()
                        rel_x = mx - target_monitor["left"]
                        rel_y = my - target_monitor["top"]
                        
                        if 0 <= rel_x < target_monitor["width"] and 0 <= rel_y < target_monitor["height"]:
                            cv2.circle(frame, (rel_x, rel_y), 8, (0, 0, 0), 2)
                            cv2.circle(frame, (rel_x, rel_y), 6, (255, 255, 255), -1)

                    # 4. Resize to exactly what Android requested
                    if frame.shape[1] != self._display_width or frame.shape[0] != self._display_height:
                        frame = cv2.resize(frame, (self._display_width, self._display_height), interpolation=cv2.INTER_NEAREST)
                    
                    # 5. Encode and Send
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)
                    b64_data = base64.b64encode(buffer).decode('utf-8')
                    msg = json.dumps({"type": "video_frame", "payload": b64_data})
                    
                    if self._loop:
                        asyncio.run_coroutine_threadsafe(self._broadcast_text(msg), self._loop)
                    
                    # 6. DYNAMIC FPS PACING
                    # Calculates exactly how long to sleep to hit the requested 30 or 60 FPS
                    target_frame_time = 1.0 / self._display_fps
                    process_time = time.time() - loop_start
                    sleep_time = max(0, target_frame_time - process_time)
                    time.sleep(sleep_time)
                
        except Exception as e:
            self._put("log", f"❌ Stream Error: {e}")
        finally:
            self._display_active = False


    def _handle_mouse_click(self, payload):
        """Hardware-level macOS Quartz Clicker with explicit Click State"""
        try:
            if isinstance(payload, str): 
                import json
                d = json.loads(payload)
            else: 
                d = payload
            
            import Quartz
            import time
            
            mac_event_source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
            btn_id = d.get("button", 0)
            act = d.get("action", "click")
            loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
            
            if btn_id == 0:  # --- LEFT BUTTON ---
                if act == "down":
                    self._is_left_down = True
                    event = Quartz.CGEventCreateMouseEvent(mac_event_source, Quartz.kCGEventLeftMouseDown, loc, Quartz.kCGMouseButtonLeft)
                    # THE MAGIC FIX 1: Force macOS to recognize this as a real window grab!
                    Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, 1)
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
                    time.sleep(0.02)
                    
                elif act == "up":
                    self._is_left_down = False
                    event = Quartz.CGEventCreateMouseEvent(mac_event_source, Quartz.kCGEventLeftMouseUp, loc, Quartz.kCGMouseButtonLeft)
                    Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, 1)
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
                    
                else: # Click
                    event_down = Quartz.CGEventCreateMouseEvent(mac_event_source, Quartz.kCGEventLeftMouseDown, loc, Quartz.kCGMouseButtonLeft)
                    Quartz.CGEventSetIntegerValueField(event_down, Quartz.kCGMouseEventClickState, 1)
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_down)
                    time.sleep(0.01)
                    event_up = Quartz.CGEventCreateMouseEvent(mac_event_source, Quartz.kCGEventLeftMouseUp, loc, Quartz.kCGMouseButtonLeft)
                    Quartz.CGEventSetIntegerValueField(event_up, Quartz.kCGMouseEventClickState, 1)
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)

        except Exception as e:
            print(f"❌ Mouse Click Error: {e}")

    def _handle_mouse_move(self, payload):
        """Hardware-level macOS Quartz Mover with Drag State Persistence"""
        try:
            if isinstance(payload, str):
                import json
                data = json.loads(payload)
            else:
                data = payload
                
            dx = float(data.get('deltaX', 0))
            dy = float(data.get('deltaY', 0))

            if dx == 0 and dy == 0: return

            sensitivity = 1.6 
            final_dx = dx * sensitivity
            final_dy = dy * sensitivity

            import Quartz
            
            mac_event_source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
            loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
            new_pos = (loc.x + final_dx, loc.y + final_dy)

            if getattr(self, '_is_left_down', False):
                # We are dragging!
                event = Quartz.CGEventCreateMouseEvent(mac_event_source, Quartz.kCGEventLeftMouseDragged, new_pos, Quartz.kCGMouseButtonLeft)
                # THE MAGIC FIX 2: Maintain the Click State during the drag!
                Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, 1)
            else:
                # We are just moving.
                event = Quartz.CGEventCreateMouseEvent(mac_event_source, Quartz.kCGEventMouseMoved, new_pos, Quartz.kCGMouseButtonLeft)
            
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

        except Exception:
            pass

    def _handle_mouse_scroll(self, payload):
        try:
            if isinstance(payload, str): d = json.loads(payload)
            else: d = payload
            # MacOS scroll sensitivity is different
            pyautogui.scroll(int(d.get("scrollDelta", 0) * 10), _pause=False)
        except Exception as e:
            print(f"❌ Mouse Scroll Error: {e}")

    def _handle_key_press(self, payload):
        try:
            if isinstance(payload, str): d = json.loads(payload)
            else: d = payload

            raw_key = d.get("key", "").lower().replace(" ", "").replace("_", "") 
            modifiers = [m.lower() for m in d.get("modifiers", [])]
            action = d.get("action", "press") 

            # --- FIX 2: MAC BRIGHTNESS BYPASS ---
            # Bypasses Python entirely and forces macOS to change brightness natively
            if raw_key == "brightnessup":
                subprocess.run(["osascript", "-e", 'tell application "System Events" to key code 144'])
                return
            elif raw_key == "brightnessdown":
                subprocess.run(["osascript", "-e", 'tell application "System Events" to key code 145'])
                return

            # macOS Media Keys
            macos_special = {
                "volumemute": Key.media_volume_mute,
                "volumedown": Key.media_volume_down,
                "volumeup": Key.media_volume_up,
                "playpause": Key.media_play_pause,
                "nexttrack": Key.media_next,
                "prevtrack": Key.media_previous
            }

            if raw_key in macos_special:
                k = macos_special[raw_key]
                if action == "press" or action == "down":
                    self.keyboard.press(k)
                    self.keyboard.release(k)
                return

            # Map Windows/Linux 'super' to Command on Mac
            if "win" in modifiers or "super" in modifiers:
                modifiers = [m for m in modifiers if m not in ["win", "super"]]
                modifiers.append("command")
            
            if raw_key in ["win", "windows", "super", "meta", "cmd"]:
                final_key = "command"
            else:
                final_key = raw_key

            # PyAutoGUI Handler
            if action == "down":
                if not modifiers: pyautogui.keyDown(final_key)
            elif action == "up":
                if not modifiers: pyautogui.keyUp(final_key)
            else:
                if modifiers:
                    pyautogui.hotkey(*modifiers + [final_key])
                else:
                    pyautogui.press(final_key)

        except Exception as e:
            print(f"❌ Keyboard Error: {e}")


    # --- GAMEPAD (KEYBOARD FALLBACK FOR MACOS) ---
    def _handle_gamepad_state(self, payload, client_ws):
        """MacOS does not support virtual controllers natively. Mapping to Keyboard (WASD)."""
        try:
            state = json.loads(payload)
            buttons = state.get('buttons', {})
            
            prev_state = self.gamepad_history.get(client_ws, {'buttons': {}})
            prev_btns = prev_state['buttons']

            # Map Start/Select
            if (buttons.get('start') or buttons.get('options')) and not prev_btns.get('start'):
                pyautogui.press('esc')
            
            if (buttons.get('select') or buttons.get('share')) and not prev_btns.get('select'):
                pyautogui.press('tab')

            # Left Stick -> WASD
            ly = float(state.get('leftStickY', 0))
            lx = float(state.get('leftStickX', 0))
            
            if ly < -0.5: pyautogui.keyDown('w')
            else: pyautogui.keyUp('w')
            
            if ly > 0.5: pyautogui.keyDown('s')
            else: pyautogui.keyUp('s')

            if lx < -0.5: pyautogui.keyDown('a')
            else: pyautogui.keyUp('a')

            if lx > 0.5: pyautogui.keyDown('d')
            else: pyautogui.keyUp('d')

            # Right Stick -> Arrow Keys
            ry = float(state.get('rightStickY', 0))
            rx = float(state.get('rightStickX', 0))
            
            if ry < -0.5: pyautogui.keyDown('up')
            else: pyautogui.keyUp('up')
            if ry > 0.5: pyautogui.keyDown('down')
            else: pyautogui.keyUp('down')
            if rx < -0.5: pyautogui.keyDown('left')
            else: pyautogui.keyUp('left')
            if rx > 0.5: pyautogui.keyDown('right')
            else: pyautogui.keyUp('right')

            # Face Buttons
            # A/Cross -> Space
            is_a = buttons.get('a') or buttons.get('cross')
            was_a = prev_btns.get('a') or prev_btns.get('cross')
            if is_a and not was_a: pyautogui.keyDown('space')
            elif not is_a and was_a: pyautogui.keyUp('space')

            # B/Circle -> Enter
            is_b = buttons.get('b') or buttons.get('circle')
            was_b = prev_btns.get('b') or prev_btns.get('circle')
            if is_b and not was_b: pyautogui.keyDown('enter')
            elif not is_b and was_b: pyautogui.keyUp('enter')

            self.gamepad_history[client_ws] = {'buttons': buttons}

        except Exception as e:
            pass

    # --- AUDIO HANDLING ---
    def _handle_audio_frame(self, payload):
        """Plays received audio from phone (Mic Mode) into BlackHole"""
        try:
            # 1. Initialize stream on first packet
            if self.audio_stream is None:
                target_out_index = None
                
                # Hunt for BlackHole's OUTPUT so we can inject the voice into it
                for i in range(self.p.get_device_count()):
                    dev_info = self.p.get_device_info_by_index(i)
                    if "BlackHole" in dev_info.get("name", "") and dev_info.get("maxOutputChannels", 0) > 0:
                        target_out_index = i
                        self._put("log", "🎤 Routing Phone Mic into BlackHole...")
                        break

                if target_out_index is None:
                    self._put("log", "⚠️ BlackHole not found. Playing Mic through Mac Speakers instead.")

                # Android usually sends mic data at 16000Hz (Mono). 
                # We will force the Mac stream to Stereo (2) so BlackHole accepts it cleanly.
                self.AUDIO_FORMAT = pyaudio.paInt16
                self.AUDIO_CHANNELS = 2 # Stereo
                self.AUDIO_RATE = 16000

                self.audio_stream = self.p.open(
                    format=self.AUDIO_FORMAT,
                    channels=self.AUDIO_CHANNELS,
                    rate=self.AUDIO_RATE,
                    output=True,
                    output_device_index=target_out_index,
                    frames_per_buffer=1024
                )

            # 2. Decode Base64 -> Raw PCM Bytes
            audio_data = base64.b64decode(payload)

            # 3. Upmix Mono to Stereo (Prevents the audio from only playing in the "Left Ear")
            mono_array = np.frombuffer(audio_data, dtype=np.int16)
            stereo_array = np.repeat(mono_array, 2)
            final_audio = stereo_array.tobytes()

            # 4. Write to BlackHole (or Speakers)
            if self.audio_stream:
                self.audio_stream.write(final_audio, exception_on_underflow=False)

        except Exception as e:
            print(f"Mic Audio Error: {e}")

    def _handle_audio_stop(self):
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except: pass
            self.audio_stream = None

    def start_audio_streaming(self):
        """PC -> Phone (Speaker Mode)"""
        if not HAS_AUDIO or self._streaming_audio: return
        self._streaming_audio = True
        
        def worker():
            p = pyaudio.PyAudio()
            try:
                # 1. HUNT FOR BLACKHOLE
                target_device_index = None
                native_rate = 48000
                capture_channels = 2
                
                # Scan all audio devices connected to the Mac
                for i in range(p.get_device_count()):
                    dev_info = p.get_device_info_by_index(i)
                    dev_name = dev_info.get("name", "")
                    
                    # If we find BlackHole and it is an Input device
                    if "BlackHole" in dev_name and dev_info.get("maxInputChannels", 0) > 0:
                        target_device_index = i
                        native_rate = int(dev_info.get("defaultSampleRate", 48000))
                        capture_channels = int(dev_info.get("maxInputChannels", 2))
                        self._put("log", f"✅ Found BlackHole! Hooking into internal audio...")
                        break

                # 2. FALLBACK (If BlackHole is missing)
                if target_device_index is None:
                    self._put("log", "⚠️ BlackHole not found. Falling back to default microphone.")
                    default_device = p.get_default_input_device_info()
                    target_device_index = default_device["index"]
                    native_rate = int(default_device.get("defaultSampleRate", 44100))
                    capture_channels = int(default_device.get("maxInputChannels", 1))

                CHUNK_SIZE = 2048 
                self._put("log", f"🔊 Audio Active: {native_rate}Hz, {capture_channels} ch")
                
                # 3. OPEN THE SPECIFIC DEVICE
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=capture_channels,
                    rate=native_rate,
                    input=True,
                    input_device_index=target_device_index, # <--- FORCE PYTHON TO USE BLACKHOLE
                    frames_per_buffer=CHUNK_SIZE
                )
                
                while self._streaming_audio:
                    try:
                        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        
                        # Upmix to stereo if we fell back to the mono mic
                        if capture_channels == 1:
                            mono_array = np.frombuffer(data, dtype=np.int16)
                            data = np.repeat(mono_array, 2).tobytes()

                        b64 = base64.b64encode(data).decode('utf-8')
                        
                        msg = json.dumps({
                            "type": "audio_frame", 
                            "rate": native_rate, 
                            "payload": b64
                        })
                        
                        if self._loop: 
                            asyncio.run_coroutine_threadsafe(self._broadcast_text(msg), self._loop)
                            
                    except Exception as e:
                        print(f"Audio Read Error: {e}")
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
        for client in list(self.clients):
            try: await client.send(message)
            except: pass


    def _handle_camera_frame(self, payload):
        """Receives live camera frames from the Android phone and displays them."""
        try:
            # 1. Decode the Base64 image from the Android app
            img_bytes = base64.b64decode(payload)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            
            # 2. Convert raw bytes into a visual OpenCV frame
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                # 3. Display the live feed in a Mac window
                cv2.imshow("Android Wireless Camera", frame)
                
                # 4. CRITICAL MAC FIX: waitKey(1) tells macOS to refresh the window UI
                # Without this exact line, the Mac window will permanently freeze!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyWindow("Android Wireless Camera")

        except Exception as e:
            self._put("log", f"❌ Camera Stream Error: {e}")

    # --- CLIPBOARD & FILES ---
    def _handle_clipboard(self, payload):
        try:
            text_to_copy = ""
            
            # 1. If payload is already a dictionary
            if isinstance(payload, dict):
                text_to_copy = payload.get("text", str(payload))
            
            # 2. If payload is a string (could be JSON or raw text)
            elif isinstance(payload, str):
                try:
                    data = json.loads(payload)
                    if isinstance(data, dict):
                        text_to_copy = data.get("text", payload)
                    else:
                        text_to_copy = payload
                except:
                    text_to_copy = payload # It's just normal text
            else:
                text_to_copy = str(payload)

            # Update Mac Clipboard and UI
            if text_to_copy:
                pyperclip.copy(text_to_copy)
                self._put("log", f"📋 Clipboard synced: {text_to_copy[:15]}...")
                self._put("clip", text_to_copy) 
                
        except Exception as e:
            self._put("log", f"❌ Clip Error: {e}")

    def _handle_file_transfer(self, payload):
        try:
            data = json.loads(payload) if isinstance(payload, str) else payload
            filename = data.get("filename", "unknown.dat")
            b64_data = data.get("data")
            is_end = data.get("is_end", False)
            
            # Ensure the directory exists (macOS sometimes deletes empty folders)
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            file_path = SAVE_DIR / filename
            
            if b64_data:
                try:
                    with open(file_path, "ab") as f:
                        f.write(base64.b64decode(b64_data))
                except PermissionError:
                    self._put("log", "❌ Error: Terminal needs permission to access Downloads folder.")
                    return

            if is_end:
                self._put("log", f"📥 Received: {filename}")
                self._put("refresh_files", "")
                
        except Exception as e:
            self._put("log", f"❌ File Recv Error: {e}")

    async def _broadcast_bytes(self, data):
        for client in list(self.clients):
            try: await client.send(data)
            except: pass

    def send_file_to_phone_thread(self, file_path):
        def worker():
            try:
                original_name = os.path.basename(file_path)
                filename = original_name.replace(":", "_")
                file_size = os.path.getsize(file_path)

                self._put("log", f"📤 Sending: {filename}")
                self.sending_lock.acquire()

                start_payload = json.dumps({"filename": filename, "size": file_size})
                self.send_to_android("file_start", start_payload)
                time.sleep(0.5)

                with open(file_path, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk: break
                        
                        packet = b'\x01' + chunk
                        if self._loop and self._loop.is_running():
                            asyncio.run_coroutine_threadsafe(self._broadcast_bytes(packet), self._loop)
                        time.sleep(0.005)

                end_packet = b'\x02'
                if self._loop:
                    asyncio.run_coroutine_threadsafe(self._broadcast_bytes(end_packet), self._loop)

                self._put("log", f"✅ Sent: {filename}")

            except Exception as e:
                self._put("log", f"❌ Error: {e}")
            finally:
                if self.sending_lock.locked(): self.sending_lock.release()

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
        print("Stopping Server...")
        if hasattr(self, 'discovery'): self.discovery.stop()
        if hasattr(self, 'udp_mouse'): self.udp_mouse.stop()
        self._vcam_running = False
        self._streaming_audio = False
        self._handle_audio_stop()
        
        # --- THE PORT RELEASE FIX ---
        # Explicitly close the WebSocket server before stopping the loop
        if self._loop and self._loop.is_running():
            async def shutdown():
                if self._ws_server:
                    self._ws_server.close()
                    await self._ws_server.wait_closed()
                self._loop.stop()
            
            try:
                asyncio.run_coroutine_threadsafe(shutdown(), self._loop)
            except Exception as e:
                pass

# ============================================
# PART 2: UI (MACOS ADJUSTMENTS)
# ============================================
class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, parent, prefs, callback_save):
        super().__init__(parent)
        self.title("Configuration")
        self.geometry("500x550") # Slightly taller to fit new settings
        self.resizable(False, False)
        self.prefs = prefs
        self.callback_save = callback_save
        self.transient(parent)
        
        ctk.CTkLabel(self, text="Configuration", font=("Segoe UI", 24, "bold")).pack(pady=(25, 20))

        # --- 1. AUTOMATION SETTINGS (NEW) ---
        self.frame_auto = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_auto.pack(fill="x", padx=40, pady=(0, 10))
        ctk.CTkLabel(self.frame_auto, text="Automation", font=("Segoe UI", 14, "bold"), text_color="#00e676").pack(anchor="w")
        
        self.sw_auto_app = ctk.CTkSwitch(self.frame_auto, text="Start App automatically on Mac Login")
        if self.prefs.get("auto_start_app", False): self.sw_auto_app.select()
        self.sw_auto_app.pack(anchor="w", pady=5)

        self.sw_auto_server = ctk.CTkSwitch(self.frame_auto, text="Auto-Start Server when App opens")
        if self.prefs.get("auto_start_server", False): self.sw_auto_server.select()
        self.sw_auto_server.pack(anchor="w", pady=5)

        # --- 2. GENERAL SETTINGS ---
        self.frame_gen = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_gen.pack(fill="x", padx=40, pady=10)
        ctk.CTkLabel(self.frame_gen, text="UI Scaling (Restart Required)", font=("Segoe UI", 14, "bold"), text_color="#00e676").pack(anchor="w")
        current_scale = prefs.get("scale", 1.0)
        self.opt_scale = ctk.CTkOptionMenu(self.frame_gen, values=["1.0", "1.2", "1.5", "2.0"], command=None)
        self.opt_scale.set(str(current_scale))
        self.opt_scale.pack(fill="x", pady=5)

        # Port
        self.frame_port = ctk.CTkFrame(self, fg_color="#212121", corner_radius=8)
        self.frame_port.pack(fill="x", padx=40, pady=10)
        ctk.CTkLabel(self.frame_port, text="Server Port:", font=("Segoe UI", 13, "bold")).pack(side="left", padx=20, pady=15)
        self.ent_port = ctk.CTkEntry(self.frame_port, width=80, justify="center", fg_color="#1a1a1a")
        self.ent_port.insert(0, str(prefs.get("port", 8080)))
        self.ent_port.pack(side="right", padx=20, pady=15)

        ctk.CTkButton(self, text="Save & Close", command=self.save, height=50,
                      fg_color="#00e676", hover_color="#00c853", text_color="black").pack(side="bottom", fill="x", padx=40, pady=30)
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
                "port": port, 
                "scale": scale,
                "auto_start_app": bool(self.sw_auto_app.get()),
                "auto_start_server": bool(self.sw_auto_server.get())
            }
            self.callback_save(new_prefs)
            self.destroy()
        except ValueError: pass

class ServerGUI:
    def __init__(self):
        self.prefs = {"port": 8080, "scale": 1.0, "auto_start_app": False, "auto_start_server": False}
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, 'r') as f: self.prefs.update(json.load(f))
        except: pass

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("green")
        
        scaling_factor = self.prefs.get("scale", 1.0)
        ctk.set_widget_scaling(scaling_factor)  
        ctk.set_window_scaling(scaling_factor)
        
        self.root = ctk.CTk()
        self.root.title(f"Use As Server (macOS) v{APP_VERSION}")
        self.root.geometry("1100x850")
        
        self.update_queue = queue.Queue()
        self.server = None
        self.is_running = False
        self.tray_icon = None
        
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(100, self.process_queue)

        # --- AUTO-START SERVER LOGIC ---
        if self.prefs.get("auto_start_server", False):
            # Wait 500ms for UI to load, then virtually 'click' the start button
            self.root.after(500, self.toggle_server)

    def toggle_mac_startup(self, enabled):
        """Creates or destroys the Apple LaunchAgent to start the app on login"""
        plist_path = Path.home() / "Library" / "LaunchAgents" / "com.useas.server.plist"
        
        if enabled:
            plist_path.parent.mkdir(parents=True, exist_ok=True)
            # Find exact path whether running as Python script or Compiled Mac .app
            app_path = sys.executable if getattr(sys, 'frozen', False) else os.path.abspath(__file__)
            
            plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.useas.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>{app_path}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>"""
            try:
                with open(plist_path, "w") as f:
                    f.write(plist_content.strip())
            except Exception as e:
                self.update_queue.put(("log", f"❌ Failed to set Mac startup: {e}"))
        else:
            if plist_path.exists():
                try: os.remove(plist_path)
                except: pass

    def save_preferences(self, new_prefs):
        old_auto_app = self.prefs.get("auto_start_app", False)
        self.prefs.update(new_prefs)
        try:
            with open(SETTINGS_FILE, 'w') as f: json.dump(self.prefs, f)
        except: pass

        # If the user toggled the Mac Startup setting, update the OS
        if new_prefs.get("auto_start_app", False) != old_auto_app:
            self.toggle_mac_startup(new_prefs.get("auto_start_app", False))
            if new_prefs.get("auto_start_app", False):
                self.update_queue.put(("log", "✅ App will now start automatically when you log into your Mac."))
            else:
                self.update_queue.put(("log", "🛑 Removed App from Mac startup sequence."))

    def setup_ui(self):
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        self.sidebar = ctk.CTkFrame(self.root, width=140, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="Use As\nmacOS", font=("Segoe UI", 20, "bold"), text_color="#00e676").pack(pady=(30, 20))
        
        self.btn_dash = ctk.CTkButton(self.sidebar, text="Dashboard", fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), 
                      command=lambda: self.select_tab("Dashboard"))
        self.btn_dash.pack(pady=10, padx=20, fill="x")
        
        self.btn_cam = ctk.CTkButton(self.sidebar, text="Camera", fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), 
                      command=lambda: self.select_tab("Camera"))
        self.btn_cam.pack(pady=10, padx=20, fill="x")
        
        self.btn_share = ctk.CTkButton(self.sidebar, text="Sharing", fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), 
                      command=lambda: self.select_tab("Sharing"))
        self.btn_share.pack(pady=10, padx=20, fill="x")

        self.btn_tray = ctk.CTkButton(self.sidebar, text="⬇ Minimize to Tray", fg_color="#2d2d2d", hover_color="#333",
                                      command=self.minimize_to_tray)
        self.btn_tray.pack(side="bottom", pady=20, padx=20, fill="x")

        self.tabview = ctk.CTkTabview(self.root, fg_color="transparent")
        self.tabview.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")
        self.tabview.add("Dashboard")
        self.tabview.add("Camera")
        self.tabview.add("Sharing")
        
        self.build_dash(self.tabview.tab("Dashboard"))
        self.build_cam(self.tabview.tab("Camera"))
        self.build_share(self.tabview.tab("Sharing"))
        self.select_tab("Dashboard")

    def select_tab(self, name):
        self.tabview.set(name)

    def build_dash(self, parent):
        self.btn_start = ctk.CTkButton(parent, text="START SERVER", height=50, font=("Segoe UI", 16, "bold"),
                                      command=self.toggle_server, fg_color="#00e676", text_color="black")
        self.btn_start.pack(pady=40, padx=40, fill="x")
        
        self.lbl_status = ctk.CTkLabel(parent, text="🔴 Offline", font=("Segoe UI", 18, "bold"))
        self.lbl_status.pack()
        
        self.ent_ip = ctk.CTkEntry(parent, placeholder_text="IP Address", justify="center")
        self.ent_ip.pack(pady=10)
        
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=20, padx=40)
        
        ctk.CTkButton(row, text="⚙ Settings", command=self.open_settings, 
                      fg_color="#333", hover_color="#444").pack(side="right", expand=True, fill="x")
        
        self.log_box = ctk.CTkTextbox(parent)
        self.log_box.pack(fill="both", expand=True, padx=20, pady=10)
        self.log_box.insert("end", "⚠️ REMINDER: Grant 'Accessibility' & 'Screen Recording' permissions in System Settings.\n")

    def build_cam(self, parent):
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        panel_left = ctk.CTkFrame(parent, width=250, corner_radius=0, fg_color="transparent")
        panel_left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(panel_left, text="Background", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 5))
        self.var_bg = ctk.StringVar(value="none")
        ctk.CTkRadioButton(panel_left, text="None", variable=self.var_bg, value="none", command=self.update_cam_settings).pack(anchor="w", pady=2)
        ctk.CTkRadioButton(panel_left, text="Blur", variable=self.var_bg, value="blur", command=self.update_cam_settings).pack(anchor="w", pady=2)
        ctk.CTkRadioButton(panel_left, text="Image", variable=self.var_bg, value="image", command=self.update_cam_settings).pack(anchor="w", pady=2)
        ctk.CTkButton(panel_left, text="Select Image...", height=24, fg_color="#333", command=self.select_bg_image).pack(fill="x", pady=5)

        ctk.CTkLabel(panel_left, text="Adjustments", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(20, 5))
        self.sw_mirror = ctk.CTkSwitch(panel_left, text="Mirror Video", command=self.update_cam_settings)
        self.sw_mirror.pack(anchor="w", pady=5)
        self.sw_flip = ctk.CTkSwitch(panel_left, text="Flip Vertical", command=self.update_cam_settings)
        self.sw_flip.pack(anchor="w", pady=5)
        
        ctk.CTkLabel(panel_left, text="Output Control", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(20, 5))
        self.sw_preview = ctk.CTkSwitch(panel_left, text="Show Preview", command=self.update_cam_settings)
        self.sw_preview.select()
        self.sw_preview.pack(anchor="w", pady=5)

        self.btn_vcam = ctk.CTkButton(panel_left, text="Start Virtual Camera (OBS)", fg_color="#3B8ED0", 
                      command=lambda: self.server.start_virtual_camera() if self.server else None)
        self.btn_vcam.pack(fill="x", pady=5)

        panel_right = ctk.CTkFrame(parent, fg_color="#000", corner_radius=10)
        panel_right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.lbl_preview = ctk.CTkLabel(panel_right, text="Waiting for connection...", text_color="gray")
        self.lbl_preview.place(relx=0.5, rely=0.5, anchor="center")

    def select_bg_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if path and self.server:
            try:
                img = cv2.imread(path)
                if img is not None:
                    self.server.bg_image_cv2 = img
                    self.var_bg.set("image")
                    self.update_cam_settings()
            except: pass

    def update_cam_settings(self):
        if self.server:
            self.server.cam_settings["mirror"] = bool(self.sw_mirror.get())
            self.server.cam_settings["flip"] = bool(self.sw_flip.get())
            self.server.cam_settings["background"] = self.var_bg.get()
            self.server.cam_settings["preview_on"] = bool(self.sw_preview.get())

    def build_share(self, parent):
        ctk.CTkLabel(parent, text="Clipboard Sync", font=("Segoe UI", 16, "bold")).pack(pady=(10, 5))
        self.txt_clip = ctk.CTkTextbox(parent, height=60)
        self.txt_clip.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(parent, text="Send Text to Phone", command=self.send_text).pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(parent, text="File Transfer", font=("Segoe UI", 16, "bold")).pack(pady=(20, 5))
        ctk.CTkButton(parent, text="📤 Send File to Phone", fg_color="#3B8ED0", command=self.send_file).pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(parent, text="Received Files:", font=("Segoe UI", 12)).pack(anchor="w", padx=15, pady=(10,0))
        self.files_frame = ctk.CTkScrollableFrame(parent, height=200, label_text="Downloads Folder")
        self.files_frame.pack(fill="both", expand=True, padx=10, pady=5)
        ctk.CTkButton(parent, text="📂 Open Received Folder", fg_color="#2d2d2d", 
                      command=lambda: subprocess.Popen(['open', str(SAVE_DIR)])).pack(fill="x", padx=10, pady=10)
        self.refresh_file_list()

    def refresh_file_list(self):
        for widget in self.files_frame.winfo_children():
            widget.destroy()
        if not SAVE_DIR.exists(): return
        try:
            files = sorted(SAVE_DIR.glob("*"), key=os.path.getmtime, reverse=True)
            for f in files:
                if f.is_file():
                    row = ctk.CTkFrame(self.files_frame, fg_color="transparent")
                    row.pack(fill="x", pady=2)
                    ctk.CTkLabel(row, text=f"📄 {f.name}", anchor="w").pack(side="left", padx=5)
                    ctk.CTkButton(row, text="Open", width=60, height=24, 
                                  command=lambda p=f: subprocess.Popen(['open', str(p)])).pack(side="right", padx=5)
        except: pass

    def send_text(self):
        if self.server: self.server.send_to_android("clipboard_text", self.txt_clip.get("1.0", "end").strip())

    def send_file(self):
        if self.server:
            paths = filedialog.askopenfilenames()
            if paths: threading.Thread(target=self.process_files, args=(paths,), daemon=True).start()

    def process_files(self, paths):
        for p in paths: self.server.send_file_to_phone_thread(p); time.sleep(0.5)

    def toggle_server(self):
        if not self.is_running:
            self.server = UnifiedRemoteServer(port=self.prefs.get("port", 8080), update_queue=self.update_queue)
            self.server.start()
            self.is_running = True
            
            self.btn_start.configure(text="STOP SERVER", fg_color="#cf6679", hover_color="#b00020")
            self.lbl_status.configure(text="🟢 Online", text_color="#00e676")
            
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
            self.btn_start.configure(text="Stopping...", state="disabled")
            def stopper():
                if self.server: self.server.stop()
                self.is_running = False
                self.btn_start.configure(text="START SERVER", state="normal", fg_color="#00e676", hover_color="#00c853")
                self.lbl_status.configure(text="🔴 Offline", text_color="gray")
                self.server = None
            threading.Thread(target=stopper, daemon=True).start()

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
                elif kind == "vcam_state":
                    is_running = data
                    if hasattr(self, 'btn_vcam'):
                        self.btn_vcam.configure(text="Stop Virtual Camera" if is_running else "Start Virtual Camera (OBS)",
                            fg_color="#cf6679" if is_running else "#3B8ED0")
                elif kind == "clip":
                    self.txt_clip.delete("1.0", "end")
                    self.txt_clip.insert("end", data)
                elif kind == "refresh_files":
                    self.refresh_file_list()
        except queue.Empty: pass
        except Exception as e: pass
        self.root.after(15, self.process_queue)

    def minimize_to_tray(self):
        """Native macOS minimize to dock (prevents Cocoa background thread crashes)"""
        self.root.iconify() 
        self.update_queue.put(("log", "⏬ App minimized to Dock."))

    def restore(self, icon=None, item=None):
        if self.tray_icon: self.tray_icon.stop()
        self.root.after(0, self.root.deiconify)

    def quit(self, icon=None, item=None):
        if self.tray_icon: self.tray_icon.stop()
        self.root.after(0, self.on_closing)

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
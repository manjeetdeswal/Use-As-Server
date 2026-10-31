"""macOS: Quartz mouse events, pynput media keys, BlackHole audio, LaunchAgent auto-start.

macOS needs Accessibility permission for input (System Settings > Privacy & Security >
Accessibility) and Screen Recording permission for the second screen.
"""

import subprocess
import time
from pathlib import Path

from .base import KeyboardGamepad, Platform

# media key name -> pynput Key attribute
MEDIA_KEYS = {
    "volumemute": "media_volume_mute", "volumedown": "media_volume_down",
    "volumeup": "media_volume_up", "playpause": "media_play_pause",
    "nexttrack": "media_next", "prevtrack": "media_previous",
}
SUPER_NAMES = {"win", "windows", "super", "meta", "cmd", "command", "win_l", "win_r"}
MODIFIERS = {"ctrl": "ctrl", "ctrl_l": "ctrl", "ctrl_r": "ctrl", "control": "ctrl",
             "shift": "shift", "shift_l": "shift", "shift_r": "shift",
             "alt": "option", "alt_l": "option", "alt_r": "option", "option": "option"}
KEY_NAMES = {"esc": "escape", "caps": "capslock", "return": "enter",
             "↑": "up", "↓": "down", "←": "left", "→": "right"}


class MacPlatform(Platform):
    name = "mac"
    display_name = "macOS"
    settings_dir = Path.home() / "Library" / "Application Support" / "UseAsServer"
    settings_filename = "settings.json"
    legacy_pref_keys = {"auto_start_app": "autostart_pc", "auto_start_server": "autostart_server"}
    features = ("unicode_keys", "clipboard_private_filter")
    scroll_notches_per_unit = 10   # same speed as the old macOS server (scrollDelta * 10)
    move_sensitivity = 1.6         # macOS pointer feels slow at 1:1

    def __init__(self):
        super().__init__()
        self._buttons_down = set()
        self._keyboard = None
        self._zoom_acc = 0

    # --- OS integration --------------------------------------------------------------
    def open_path(self, path):
        subprocess.Popen(["open", str(path)])

    def set_autostart(self, enabled):
        plist = Path.home() / "Library" / "LaunchAgents" / "com.useas.server.plist"
        try:
            if enabled:
                plist.parent.mkdir(parents=True, exist_ok=True)
                args = "\n".join(f"        <string>{p}</string>" for p in self.launch_command())
                plist.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.useas.server</string>
    <key>ProgramArguments</key>
    <array>
{args}
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
""")
            elif plist.exists():
                plist.unlink()
            return True
        except Exception as e:
            self._log(f"⚠️ Couldn't change auto-start: {e}")
            return False

    # --- clipboard -------------------------------------------------------------------
    def _pasteboard(self):
        try:
            from AppKit import NSPasteboard
            return NSPasteboard.generalPasteboard()
        except Exception:
            return None

    def clipboard_sequence(self):
        pb = self._pasteboard()
        return int(pb.changeCount()) if pb is not None else None

    def clipboard_is_private(self):
        """Password managers tag secrets with org.nspasteboard.ConcealedType."""
        pb = self._pasteboard()
        try:
            types = [str(t) for t in (pb.types() or [])] if pb is not None else []
        except Exception:
            return False
        return any(t in ("org.nspasteboard.ConcealedType", "org.nspasteboard.TransientType") for t in types)

    def set_clipboard_image(self, image):
        import io
        try:
            from AppKit import NSPasteboardTypePNG
            from Foundation import NSData
        except Exception:
            return False
        pb = self._pasteboard()
        if pb is None:
            return False
        buf = io.BytesIO()
        image.save(buf, "PNG")
        png = buf.getvalue()
        try:
            pb.clearContents()
            return bool(pb.setData_forType_(NSData.dataWithBytes_length_(png, len(png)), NSPasteboardTypePNG))
        except Exception as e:
            self._log(f"⚠️ Couldn't set the clipboard image: {e}")
            return False

    # --- input -------------------------------------------------------------------------
    def _quartz(self):
        import Quartz
        return Quartz

    def _post_mouse(self, event_type, button, pos=None):
        Q = self._quartz()
        source = Q.CGEventSourceCreate(Q.kCGEventSourceStateHIDSystemState)
        if pos is None:
            pos = Q.CGEventGetLocation(Q.CGEventCreate(None))
        event = Q.CGEventCreateMouseEvent(source, event_type, pos, button)
        # Click state 1 makes macOS treat it as a real click/drag (window dragging works).
        Q.CGEventSetIntegerValueField(event, Q.kCGMouseEventClickState, 1)
        Q.CGEventPost(Q.kCGHIDEventTap, event)

    def mouse_move_rel(self, dx, dy):
        Q = self._quartz()
        loc = Q.CGEventGetLocation(Q.CGEventCreate(None))
        pos = (loc.x + dx * self.move_sensitivity, loc.y + dy * self.move_sensitivity)
        # While a button is held this must be a "dragged" event, or drags don't work.
        if 0 in self._buttons_down:
            etype, button = Q.kCGEventLeftMouseDragged, Q.kCGMouseButtonLeft
        elif 1 in self._buttons_down:
            etype, button = Q.kCGEventRightMouseDragged, Q.kCGMouseButtonRight
        elif 2 in self._buttons_down:
            etype, button = Q.kCGEventOtherMouseDragged, Q.kCGMouseButtonCenter
        else:
            etype, button = Q.kCGEventMouseMoved, Q.kCGMouseButtonLeft
        self._post_mouse(etype, button, pos)

    def mouse_button(self, button, action, gaming_mode=False):
        """All three buttons (the old macOS server only handled the left one)."""
        Q = self._quartz()
        kinds = {
            0: (Q.kCGEventLeftMouseDown, Q.kCGEventLeftMouseUp, Q.kCGMouseButtonLeft),
            1: (Q.kCGEventRightMouseDown, Q.kCGEventRightMouseUp, Q.kCGMouseButtonRight),
            2: (Q.kCGEventOtherMouseDown, Q.kCGEventOtherMouseUp, Q.kCGMouseButtonCenter),
        }
        down, up, qbutton = kinds.get(button, kinds[0])
        if action == "down":
            self._buttons_down.add(button)
            self._post_mouse(down, qbutton)
        elif action == "up":
            self._buttons_down.discard(button)
            self._post_mouse(up, qbutton)
        else:
            self._post_mouse(down, qbutton)
            time.sleep(0.04 if gaming_mode else 0.01)
            self._post_mouse(up, qbutton)

    def mouse_zoom(self, notches):
        """Cmd + '=' / Cmd + '-' zooms in Safari, Chrome, Preview, Office…"""
        import pyautogui
        key = "=" if notches > 0 else "-"
        for _ in range(min(abs(int(notches)), 5)):
            pyautogui.hotkey("command", key)

    def _pynput(self):
        if self._keyboard is None:
            from pynput.keyboard import Controller
            self._keyboard = Controller()
        return self._keyboard

    def key_event(self, key, modifiers, action, gaming_mode=False):
        import pyautogui
        raw = key.lower().replace(" ", "").replace("_", "")

        if raw in MEDIA_KEYS:
            if action in ("press", "down"):
                from pynput.keyboard import Key
                k = getattr(Key, MEDIA_KEYS[raw])
                kb = self._pynput()
                kb.press(k)
                kb.release(k)
            return

        mods = []
        for m in (m.lower() for m in modifiers):
            mods.append("command" if m in SUPER_NAMES else MODIFIERS.get(m, m))
        final = "command" if raw in SUPER_NAMES else KEY_NAMES.get(raw, KEY_NAMES.get(key.lower(), key.lower()))
        keys = mods + [final]
        # The old server ignored down/up whenever a modifier was held; hold them properly.
        if action == "down":
            for k in keys:
                pyautogui.keyDown(k)
        elif action == "up":
            for k in reversed(keys):
                pyautogui.keyUp(k)
        else:
            pyautogui.hotkey(*keys)

    def type_unicode(self, ch, action="press"):
        if action == "up":
            return
        try:
            ch.encode("utf-16-le")
        except UnicodeEncodeError:
            return
        self._pynput().type(ch)

    def change_brightness(self, up):
        code = "144" if up else "145"
        subprocess.Popen(["osascript", "-e", f'tell application "System Events" to key code {code}'])

    # --- gamepad -----------------------------------------------------------------------
    def create_gamepad(self, on_rumble):
        # macOS has no virtual-controller driver: map to keyboard (WASD + arrows).
        return KeyboardGamepad(self)

    # --- audio -------------------------------------------------------------------------
    @staticmethod
    def _find_blackhole(pa, want_output):
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            channels = info.get("maxOutputChannels" if want_output else "maxInputChannels", 0)
            if "BlackHole" in info.get("name", "") and channels > 0:
                return info
        return None

    def mic_output(self, pa):
        info = self._find_blackhole(pa, want_output=True)
        if info is None:
            return None, 2, "Mac speakers (install BlackHole to use it as a microphone)"
        # BlackHole wants stereo; the core upmixes the phone's mono audio.
        return info["index"], 2, info.get("name", "BlackHole")

    def loopback_device(self, pa, target_rate):
        """macOS can't capture system audio by itself; BlackHole (a multi-output device
        with BlackHole + your speakers) provides it."""
        info = self._find_blackhole(pa, want_output=False)
        if info is None:
            self._log("⚠️ BlackHole not found — streaming the default microphone instead.")
            info = pa.get_default_input_device_info()
        return (info["index"], int(info.get("defaultSampleRate", 48000)),
                max(1, min(2, int(info.get("maxInputChannels", 2)))), info.get("name", "input"))

    # --- video -------------------------------------------------------------------------
    def vcam_options(self):
        return [("OBS Camera", None)]

    def resolve_vcam_device(self, requested):
        return None   # pyvirtualcam picks the OBS backend on macOS

    def vcam_help(self):
        return "Install OBS Studio and start its Virtual Camera once."

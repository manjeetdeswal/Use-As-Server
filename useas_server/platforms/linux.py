"""Linux: PyAutoGUI/xdotool input, uinput virtual pads, PulseAudio/PipeWire routing, xrandr."""

import os
import re
import subprocess
from pathlib import Path

from .base import KeyboardGamepad, Platform, VirtualGamepad, run_quiet, which

MIC_SINK = "UseAs_Mic"

# xdotool names for keys PyAutoGUI can't press on X11
XDOTOOL_SPECIAL = {
    "brightnessup": "XF86MonBrightnessUp", "brightnessdown": "XF86MonBrightnessDown",
    "volumemute": "XF86AudioMute", "volumedown": "XF86AudioLowerVolume", "volumeup": "XF86AudioRaiseVolume",
    "playpause": "XF86AudioPlay", "nexttrack": "XF86AudioNext", "prevtrack": "XF86AudioPrev",
    "stop": "XF86AudioStop", "search": "XF86Search", "home": "Home", "end": "End",
    "pageup": "Page_Up", "pagedown": "Page_Down", "insert": "Insert", "delete": "Delete",
    "printscreen": "Print", "numlock": "Num_Lock", "scrolllock": "Scroll_Lock", "pause": "Pause",
    "menu": "Menu", "capslock": "Caps_Lock",
}
NUMPAD = {
    **{f"numpad{i}": str(i) for i in range(10)}, **{f"num{i}": str(i) for i in range(10)},
    "numpadenter": "enter", "numpadadd": "+", "numpadsubtract": "-",
    "numpadmultiply": "*", "numpaddivide": "/", "numpaddecimal": ".",
}
ARROWS = {"up": "up", "down": "down", "left": "left", "right": "right",
          "↑": "up", "↓": "down", "←": "left", "→": "right"}
KEY_NAMES = {"enter": "enter", "return": "enter", "backspace": "backspace", "tab": "tab",
             "space": "space", "esc": "escape", "caps": "capslock"}
MODIFIERS = {"shift_l": "shiftleft", "shift_r": "shiftright", "ctrl_l": "ctrlleft",
             "ctrl_r": "ctrlright", "alt_l": "altleft", "alt_r": "altright",
             "shift": "shift", "ctrl": "ctrl", "alt": "alt"}
SUPER_NAMES = {"win", "windows", "super", "meta", "cmd", "win_l", "win_r"}


class UInputPad(VirtualGamepad):
    """Virtual Xbox-style controller through /dev/uinput (python-evdev)."""

    description = "Virtual controller (uinput)"

    def __init__(self):
        from evdev import AbsInfo, UInput, ecodes
        self.e = ecodes
        stick = AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=16, resolution=0)
        trig = AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0)
        self.map = {
            "a": ecodes.BTN_A, "b": ecodes.BTN_B, "x": ecodes.BTN_X, "y": ecodes.BTN_Y,
            "lb": ecodes.BTN_TL, "rb": ecodes.BTN_TR, "back": ecodes.BTN_SELECT,
            "start": ecodes.BTN_START, "guide": ecodes.BTN_MODE,
            "ls": ecodes.BTN_THUMBL, "rs": ecodes.BTN_THUMBR,
            "dpad_up": ecodes.BTN_DPAD_UP, "dpad_down": ecodes.BTN_DPAD_DOWN,
            "dpad_left": ecodes.BTN_DPAD_LEFT, "dpad_right": ecodes.BTN_DPAD_RIGHT,
        }
        caps = {
            ecodes.EV_KEY: list(self.map.values()),
            ecodes.EV_ABS: [(ecodes.ABS_X, stick), (ecodes.ABS_Y, stick),
                            (ecodes.ABS_RX, stick), (ecodes.ABS_RY, stick),
                            (ecodes.ABS_Z, trig), (ecodes.ABS_RZ, trig)],
        }
        self.dev = UInput(events=caps, name="UseAs-Virtual-Controller", version=0x1)
        self.prev = {}

    def apply(self, norm):
        e = self.e
        for name, code in self.map.items():
            pressed = bool(norm["buttons"].get(name, False))
            if pressed != self.prev.get(name, False):   # only write changes
                self.dev.write(e.EV_KEY, code, 1 if pressed else 0)
                self.prev[name] = pressed
        self.dev.write(e.EV_ABS, e.ABS_X, int(norm["lx"] * 32767))
        self.dev.write(e.EV_ABS, e.ABS_Y, int(norm["ly"] * 32767))
        self.dev.write(e.EV_ABS, e.ABS_RX, int(norm["rx"] * 32767))
        self.dev.write(e.EV_ABS, e.ABS_RY, int(norm["ry"] * 32767))
        self.dev.write(e.EV_ABS, e.ABS_Z, int(norm["lt"] * 255))
        self.dev.write(e.EV_ABS, e.ABS_RZ, int(norm["rt"] * 255))
        self.dev.syn()

    def close(self):
        try:
            self.release_all()
        except Exception:
            pass
        try:
            self.dev.close()
        except Exception:
            pass


class LinuxPlatform(Platform):
    name = "linux"
    display_name = "Linux"
    settings_dir = Path.home() / ".config" / "useas_server"
    settings_filename = "settings.json"
    default_ui_scale = 1.4
    legacy_pref_keys = {"autostart": "autostart_pc", "auto_server": "autostart_server"}
    features = ("virtual_gamepad", "unicode_keys", "virtual_display")

    def __init__(self):
        super().__init__()
        self._has_xdotool = which("xdotool") is not None
        self._uinput_warned = False

    # --- lifecycle -------------------------------------------------------------------
    def setup(self):
        """Virtual microphone: a null sink called UseAs_Mic. Apps pick its monitor
        ('Monitor of UseAs_Virtual_Microphone') as their microphone."""
        if which("pactl") is None:
            return
        check = run_quiet(["pactl", "list", "short", "sinks"])
        if check is not None and MIC_SINK not in (check.stdout or ""):
            self._log("⚙️ Creating virtual microphone…")
            run_quiet(["pactl", "load-module", "module-null-sink", f"sink_name={MIC_SINK}",
                       'sink_properties=device.description="UseAs_Virtual_Microphone"'])
            run_quiet(["pactl", "set-sink-volume", MIC_SINK, "100%"])

    # --- OS integration --------------------------------------------------------------
    def launch_command(self):
        """Inside the Flatpak the app's own path only exists in the sandbox; the auto-start
        entry and the Nautilus script run on the computer, so they go through flatpak run."""
        app_id = os.environ.get("FLATPAK_ID")
        if app_id:
            return ["flatpak", "run", app_id]
        return super().launch_command()

    def open_path(self, path):
        subprocess.Popen(["xdg-open", str(path)])

    def set_autostart(self, enabled):
        """XDG autostart entry. (The old Linux server saved this setting but never applied it.)"""
        entry = Path.home() / ".config" / "autostart" / "useas-server.desktop"
        try:
            if enabled:
                entry.parent.mkdir(parents=True, exist_ok=True)
                cmd = " ".join(f'"{p}"' for p in self.launch_command())
                entry.write_text(
                    "[Desktop Entry]\nType=Application\nName=Use As Server\n"
                    f"Exec={cmd}\nX-GNOME-Autostart-enabled=true\nTerminal=false\n")
            elif entry.exists():
                entry.unlink()
            return True
        except Exception as e:
            self._log(f"⚠️ Couldn't change auto-start: {e}")
            return False

    send_to_label = "Files (Nautilus) → Scripts → Send to phone (Use As)"

    def install_send_to(self, enabled):
        """Nautilus runs scripts from this folder with the selected files as arguments."""
        script = Path.home() / ".local" / "share" / "nautilus" / "scripts" / "Send to phone (Use As)"
        try:
            if not enabled:
                if script.exists():
                    script.unlink()
                return True
            script.parent.mkdir(parents=True, exist_ok=True)
            cmd = " ".join("'" + p.replace("'", "'\\''") + "'" for p in self.send_to_command())
            script.write_text(f'#!/bin/sh\nexec {cmd} "$@"\n')
            script.chmod(0o755)
            return True
        except Exception as e:
            self._log(f"⚠️ Couldn't update the Nautilus script: {e}")
            return False

    # --- input -------------------------------------------------------------------------
    scroll_notches_per_unit = 5   # same speed as the old Linux server (scrollDelta * 5)

    def _xdotool(self, action, keysym):
        if not self._has_xdotool:
            return False
        verb = {"down": "keydown", "up": "keyup"}.get(action, "key")
        subprocess.Popen(["xdotool", verb, keysym])
        return True

    def key_event(self, key, modifiers, action, gaming_mode=False):
        import pyautogui
        raw = key.lower().replace(" ", "").replace("_", "")
        mods = [m.lower() for m in modifiers]

        if raw in XDOTOOL_SPECIAL and self._xdotool(action, XDOTOOL_SPECIAL[raw]):
            return
        if raw in SUPER_NAMES:
            if action != "up":
                if not self._xdotool("press", "Super_L"):
                    pyautogui.press("win")
            return

        final = NUMPAD.get(raw) or ARROWS.get(raw) or KEY_NAMES.get(raw) or raw
        mapped = [MODIFIERS.get(m, m) for m in mods if m not in SUPER_NAMES]
        if any(m in SUPER_NAMES for m in mods):
            mapped.insert(0, "win")
        keys = mapped + [final]
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
            ch.encode("utf-16-le")   # a lone surrogate can't be typed on its own
        except UnicodeEncodeError:
            return
        if self._has_xdotool:
            subprocess.Popen(["xdotool", "type", "--clearmodifiers", "--", ch])
        else:
            super().type_unicode(ch, action)

    def change_brightness(self, up):
        if not self._xdotool("press", "XF86MonBrightnessUp" if up else "XF86MonBrightnessDown"):
            super().change_brightness(up)

    # --- gamepad -----------------------------------------------------------------------
    def create_gamepad(self, on_rumble):
        try:
            return UInputPad()
        except ImportError:
            reason = "python-evdev isn't installed (pip install evdev)"
        except PermissionError:
            reason = "no permission for /dev/uinput (run: sudo chmod 666 /dev/uinput, or add a udev rule)"
        except Exception as e:
            reason = str(e)
        if not self._uinput_warned:
            self._uinput_warned = True
            self._log(f"⚠️ Virtual controller unavailable — {reason}. Using keyboard mode.")
        return KeyboardGamepad(self)

    # --- audio -------------------------------------------------------------------------
    def mic_output(self, pa):
        # Route this process's playback into the virtual mic sink (read by libpulse
        # when the stream connects; works on PulseAudio and PipeWire-pulse).
        os.environ["PULSE_SINK"] = MIC_SINK
        return None, 1, f"{MIC_SINK} (pick 'Monitor of UseAs_Virtual_Microphone' as your mic)"

    def after_mic_stream_opened(self):
        """Fallback if PULSE_SINK was ignored: move OUR stream (matched by PID, not by
        guessing 'python' like the old code, which could move other apps' audio)."""
        if which("pactl") is None:
            return
        import threading

        def move():
            import time
            time.sleep(1.0)
            out = run_quiet(["pactl", "list", "sink-inputs"])
            if out is None:
                return
            current, pid = None, str(os.getpid())
            for line in out.stdout.splitlines():
                m = re.match(r"\s*Sink Input #(\d+)", line)
                if m:
                    current = m.group(1)
                elif current and "application.process.id" in line and f'"{pid}"' in line:
                    run_quiet(["pactl", "move-sink-input", current, MIC_SINK])
                    return

        threading.Thread(target=move, daemon=True).start()

    def loopback_device(self, pa, target_rate):
        """Capture the monitor of the default output (what you hear), not the microphone."""
        label = "default input"
        if which("pactl") is not None:
            sink = run_quiet(["pactl", "get-default-sink"])
            if sink is not None and sink.returncode == 0 and sink.stdout.strip():
                os.environ["PULSE_SOURCE"] = sink.stdout.strip() + ".monitor"
                label = f"{sink.stdout.strip()} (monitor)"
        info = pa.get_default_input_device_info()
        rate = int(target_rate) if target_rate else int(info.get("defaultSampleRate", 48000))
        return info["index"], rate, 2, label

    # --- video -------------------------------------------------------------------------
    def vcam_options(self):
        device = "/dev/video10" if os.path.exists("/dev/video10") else None
        return [("Virtual Camera", device)]

    def resolve_vcam_device(self, requested):
        # The phone asks for "OBS Virtual Camera"/"Unity Video Capture" (Windows names).
        return self.vcam_options()[0][1]

    def vcam_help(self):
        return "Load the driver: sudo modprobe v4l2loopback devices=1 video_nr=10 exclusive_caps=1"

    # --- clipboard: images and files -------------------------------------------------
    def _wayland(self):
        return bool(os.environ.get("WAYLAND_DISPLAY")) and which("wl-paste") is not None

    def clipboard_rich(self):
        # Copied files arrive as a text/uri-list (Nautilus, Dolphin, Thunar…).
        uris = None
        if self._wayland():
            r = run_quiet(["wl-paste", "--no-newline", "--type", "text/uri-list"], timeout=2)
            uris = r.stdout if r is not None and r.returncode == 0 else None
        elif which("xclip"):
            r = run_quiet(["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"], timeout=2)
            if r is not None and "text/uri-list" in (r.stdout or ""):
                r = run_quiet(["xclip", "-selection", "clipboard", "-t", "text/uri-list", "-o"], timeout=2)
                uris = r.stdout if r is not None and r.returncode == 0 else None
        if uris:
            from urllib.parse import unquote, urlparse
            paths = [unquote(urlparse(u.strip()).path) for u in uris.splitlines()
                     if u.strip().startswith("file://")]
            if paths:
                return ("files", paths)
        return super().clipboard_rich()

    def set_clipboard_image(self, image):
        import io
        buf = io.BytesIO()
        image.save(buf, "PNG")
        png = buf.getvalue()
        if self._wayland() and which("wl-copy"):
            cmd = ["wl-copy", "--type", "image/png"]
        elif which("xclip"):
            cmd = ["xclip", "-selection", "clipboard", "-t", "image/png", "-i"]
        else:
            self._log("⚠️ Install xclip (or wl-clipboard) to receive images on the clipboard")
            return False
        try:
            # Both tools fork into the background to keep serving the clipboard.
            subprocess.run(cmd, input=png, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            self._log(f"⚠️ Couldn't set the clipboard image: {e}")
            return False

    def ensure_display_count(self, needed, width=None, height=None, phone_size=None, exact_size=None):
        """The xrandr 'HDMI hack' from the old Linux server: add a virtual 1920x1080
        output to the right of the primary screen."""
        if which("xrandr") is None:
            return False
        try:
            out = subprocess.check_output("xrandr | grep ' primary'", shell=True).decode()
            m = re.search(r"primary (\d+)x\d+\+\d+\+\d+", out)
            primary_width = int(m.group(1)) if m else 1920
        except Exception:
            primary_width = 1920
        port, mode = "HDMI-A-0", "1920x1080_Virtual"
        self._log("⚙️ Creating a virtual display…")
        run_quiet(["xrandr", "--output", port, "--off"])
        run_quiet(["xrandr", "--delmode", port, mode])
        run_quiet(["xrandr", "--rmmode", mode])
        run_quiet(["xrandr", "--newmode", mode, "173.00", "1920", "2048", "2248", "2576",
                   "1080", "1083", "1088", "1120", "-hsync", "+vsync"])
        run_quiet(["xrandr", "--addmode", port, mode])
        result = run_quiet(["xrandr", "--output", port, "--mode", mode, "--pos", f"{primary_width}x0"])
        ok = result is not None and result.returncode == 0
        self._log("✅ Virtual display ready" if ok else f"❌ Couldn't create a virtual display on {port}")
        if ok:
            import time
            time.sleep(1.5)
        return ok

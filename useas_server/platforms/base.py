"""Platform interface. The core server only talks to the OS through these methods.

Defaults use PyAutoGUI / PyAudio so a new platform works out of the box; windows.py,
linux.py and mac.py override what they can do better natively.
"""

import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Gamepad state normalisation (shared by every backend)
# ---------------------------------------------------------------------------

#: phone button name -> canonical Xbox-style name
BUTTON_ALIASES = {
    "a": "a", "cross": "a",
    "b": "b", "circle": "b",
    "x": "x", "square": "x",
    "y": "y", "triangle": "y",
    "lb": "lb", "l1": "lb", "l_shoulder": "lb",
    "rb": "rb", "r1": "rb", "r_shoulder": "rb",
    "view": "back", "share": "back", "select": "back", "back": "back",
    "menu": "start", "options": "start", "start": "start",
    "xbox": "guide", "ps": "guide", "mode": "guide", "center": "guide",
    "ls_btn": "ls", "thumbl": "ls", "l_thumb": "ls",
    "rs_btn": "rs", "thumbr": "rs", "r_thumb": "rs",
    "dpad_up": "dpad_up", "up": "dpad_up",
    "dpad_down": "dpad_down", "down": "dpad_down",
    "dpad_left": "dpad_left", "left": "dpad_left",
    "dpad_right": "dpad_right", "right": "dpad_right",
}
LEFT_TRIGGER_BUTTONS = {"lt", "l2"}
RIGHT_TRIGGER_BUTTONS = {"rt", "r2"}
CANONICAL_BUTTONS = sorted(set(BUTTON_ALIASES.values()))


def _clamp(value, lo=-1.0, hi=1.0):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return 0.0


def normalize_gamepad_state(state):
    """Phone gamepad_state -> {"buttons": {canonical: bool}, "lx", "ly", "rx", "ry", "lt", "rt"}.

    Sticks are -1..1 with +y meaning DOWN (as the phone sends it); triggers 0..1. Buttons
    named "lt"/"l2"/"rt"/"r2" (digital-trigger mode or key bindings) press the trigger fully.
    """
    raw_buttons = state.get("buttons") or {}
    buttons = {}
    lt_pressed = rt_pressed = False
    for name, pressed in raw_buttons.items():
        key = str(name).lower()
        if key in LEFT_TRIGGER_BUTTONS:
            lt_pressed = lt_pressed or bool(pressed)
        elif key in RIGHT_TRIGGER_BUTTONS:
            rt_pressed = rt_pressed or bool(pressed)
        elif key in BUTTON_ALIASES:
            canon = BUTTON_ALIASES[key]
            buttons[canon] = buttons.get(canon, False) or bool(pressed)
    lt = 1.0 if lt_pressed else _clamp(state.get("leftTrigger", 0.0), 0.0, 1.0)
    rt = 1.0 if rt_pressed else _clamp(state.get("rightTrigger", 0.0), 0.0, 1.0)
    return {
        "buttons": buttons,
        "lx": _clamp(state.get("leftStickX", 0.0)),
        "ly": _clamp(state.get("leftStickY", 0.0)),
        "rx": _clamp(state.get("rightStickX", 0.0)),
        "ry": _clamp(state.get("rightStickY", 0.0)),
        "lt": lt,
        "rt": rt,
    }


class VirtualGamepad:
    """A controller owned by one phone. Backends implement apply()/close()."""

    #: True when the OS sends vibration back for this pad
    supports_rumble = False
    #: shown in the log
    description = "Virtual controller"

    def apply(self, norm):
        raise NotImplementedError

    def release_all(self):
        self.apply(normalize_gamepad_state({}))

    def player_number(self):
        """1-4 if the OS assigned this pad a player slot, else None."""
        return None

    def close(self):
        pass


class KeyboardGamepad(VirtualGamepad):
    """Fallback when the OS has no virtual-controller driver (macOS, or Linux without
    /dev/uinput access): left stick = WASD, right stick = arrows, A = Space, B = Enter,
    X = E, Y = Q, Start = Esc, Back = Tab, D-pad = arrows. Keys are held, not tapped,
    so walking and holding a button work."""

    description = "Keyboard fallback (WASD + arrows)"

    BUTTON_KEYS = {"a": "space", "b": "enter", "x": "e", "y": "q",
                   "start": "esc", "back": "tab",
                   "dpad_up": "up", "dpad_down": "down", "dpad_left": "left", "dpad_right": "right"}

    def __init__(self, platform):
        self.platform = platform
        self.held = set()

    def _set(self, key, down):
        if down and key not in self.held:
            self.held.add(key)
            self.platform.key_event(key, [], "down", gaming_mode=False)
        elif not down and key in self.held:
            self.held.discard(key)
            self.platform.key_event(key, [], "up", gaming_mode=False)

    def apply(self, norm):
        want = set()
        if norm["ly"] < -0.5: want.add("w")
        if norm["ly"] > 0.5: want.add("s")
        if norm["lx"] < -0.5: want.add("a")
        if norm["lx"] > 0.5: want.add("d")
        if norm["ry"] < -0.5: want.add("up")
        if norm["ry"] > 0.5: want.add("down")
        if norm["rx"] < -0.5: want.add("left")
        if norm["rx"] > 0.5: want.add("right")
        for button, key in self.BUTTON_KEYS.items():
            if norm["buttons"].get(button):
                want.add(key)
        for key in list(self.held - want):
            self._set(key, False)
        for key in want - self.held:
            self._set(key, True)

    def release_all(self):
        for key in list(self.held):
            self._set(key, False)

    def close(self):
        self.release_all()


# ---------------------------------------------------------------------------
# Platform base
# ---------------------------------------------------------------------------

class Platform:
    name = "generic"
    display_name = "this computer"
    settings_dir = Path.home() / ".config" / "useas_server"
    settings_filename = "settings.json"
    default_ui_scale = 1.0
    #: legacy pref key -> new key (for settings written by the old per-OS servers)
    legacy_pref_keys = {}
    #: features this platform adds to the hello message
    features = ()

    #: wheel notches per phone scroll unit (the phone sends ~10 units per notch on Windows)
    scroll_notches_per_unit = 0.1

    def __init__(self):
        self._log = lambda msg: None
        self._scroll_acc = 0.0

    # --- lifecycle -------------------------------------------------------------------
    def set_logger(self, log):
        self._log = log

    def early_init(self):
        """Called before any GUI is created (DPI awareness etc.)."""

    def setup(self):
        """Called once when a server starts (create virtual devices etc.)."""

    @property
    def settings_file(self):
        return self.settings_dir / self.settings_filename

    # --- OS integration --------------------------------------------------------------
    def open_path(self, path):
        subprocess.Popen(["xdg-open", str(path)])

    def launch_command(self):
        """Command that starts this app, for auto-start entries."""
        if getattr(sys, "frozen", False):
            return [sys.executable]
        return [sys.executable, os.path.abspath(sys.argv[0])]

    def set_autostart(self, enabled):
        return False

    #: label of the file-manager "send to phone" entry, or None if unsupported
    send_to_label = None

    def install_send_to(self, enabled):
        """Add/remove 'Send to phone' in the file manager's right-click menu."""
        return False

    def send_to_command(self):
        """Command the file manager runs; selected files are appended."""
        return self.launch_command() + ["--send"]

    def ensure_firewall_rule(self):
        pass

    def is_admin(self):
        return False

    def relaunch_as_admin(self, extra_args=()):
        return False

    def supports_admin(self):
        return False

    def style_window(self, root):
        pass

    # --- clipboard -------------------------------------------------------------------
    def clipboard_sequence(self):
        """A number that changes on every clipboard change, or None if unknown."""
        return None

    def clipboard_is_private(self):
        """True if the current clipboard content was marked secret (password managers)."""
        return False

    def clipboard_rich(self):
        """What's on the clipboard besides text: ("image", PIL.Image), ("files", [paths]) or None.
        Pillow reads images on every OS (Linux needs wl-paste or xclip) and copied files on Windows."""
        try:
            from PIL import ImageGrab
            data = ImageGrab.grabclipboard()
        except Exception:
            return None
        if data is None:
            return None
        if isinstance(data, (list, tuple)):
            paths = [str(p) for p in data if isinstance(p, str)]
            return ("files", paths) if paths else None
        return ("image", data)

    def set_clipboard_image(self, image):
        """Put a PIL image on the clipboard. True on success."""
        return False

    # --- input -------------------------------------------------------------------------
    def mouse_move_rel(self, dx, dy):
        import pyautogui
        pyautogui.moveRel(dx, dy, _pause=False)

    def mouse_button(self, button, action, gaming_mode=False):
        import pyautogui
        name = {0: "left", 1: "right", 2: "middle"}.get(button, "left")
        if action == "down":
            pyautogui.mouseDown(button=name, _pause=False)
        elif action == "up":
            pyautogui.mouseUp(button=name, _pause=False)
        else:
            pyautogui.click(button=name, _pause=False)

    def mouse_scroll(self, scroll_delta):
        """scroll_delta: the phone's units. Fractions are carried over, so slow two-finger
        scrolls add up instead of being rounded to nothing."""
        import pyautogui
        self._scroll_acc += scroll_delta * self.scroll_notches_per_unit
        clicks = int(self._scroll_acc)
        if clicks:
            self._scroll_acc -= clicks
            pyautogui.scroll(clicks, _pause=False)

    def mouse_zoom(self, notches):
        """Zoom in (+) / out (-) by whole wheel notches (pinch on the touchpad)."""
        import pyautogui
        pyautogui.keyDown("ctrl")
        try:
            pyautogui.scroll(notches, _pause=False)
        finally:
            pyautogui.keyUp("ctrl")

    def key_event(self, key, modifiers, action, gaming_mode=False):
        """key: phone key name; modifiers: list; action: press|down|up."""
        import pyautogui
        keys = [m.lower() for m in modifiers] + [key.lower()]
        if action == "down":
            for k in keys:
                pyautogui.keyDown(k)
        elif action == "up":
            for k in reversed(keys):
                pyautogui.keyUp(k)
        else:
            pyautogui.hotkey(*keys)

    def type_unicode(self, ch, action="press"):
        """Type a character that isn't on a US keyboard."""
        if action in ("press", "down"):
            import pyautogui
            pyautogui.write(ch)

    def change_brightness(self, up):
        try:
            import screen_brightness_control as sbc
            current = sbc.get_brightness()
            if current:
                value = current[0]
                sbc.set_brightness(min(100, value + 10) if up else max(0, value - 10))
        except Exception as e:
            self._log(f"⚠️ Brightness change not supported: {e}")

    def cursor_position(self):
        import pyautogui
        return pyautogui.position()

    # --- gamepad -----------------------------------------------------------------------
    def create_gamepad(self, on_rumble):
        """Return a VirtualGamepad for one phone. on_rumble(intensity 0..1) may be called
        from any thread by pads that support vibration."""
        return KeyboardGamepad(self)

    # --- audio -------------------------------------------------------------------------
    def mic_output(self, pa):
        """Where the phone's microphone is played: (device_index or None, channels, label).
        A 'virtual cable' here lets other apps pick it up as a microphone."""
        return None, 1, "default speakers"

    def after_mic_stream_opened(self):
        pass

    def run_loopback_capture(self, target_rate, still_wanted, restart_event, emit):
        """Capture what the PC is playing and call emit(pcm_bytes, rate, channels) until
        still_wanted() is False. restart_event is set when the requested rate changes."""
        import pyaudio
        pa = pyaudio.PyAudio()
        try:
            device_index, rate, channels, label = self.loopback_device(pa, target_rate)
            self._log(f"🔊 Capturing {label} @ {rate}Hz, {channels} ch")
            stream = pa.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True,
                             input_device_index=device_index, frames_per_buffer=2048)
            try:
                while still_wanted() and not restart_event.is_set():
                    data = stream.read(2048, exception_on_overflow=False)
                    emit(data, rate, channels)
            finally:
                stream.stop_stream()
                stream.close()
        finally:
            pa.terminate()

    def loopback_device(self, pa, target_rate):
        """(device_index, rate, channels, label) for run_loopback_capture's default loop."""
        info = pa.get_default_input_device_info()
        return (info["index"], int(info.get("defaultSampleRate", 48000)),
                max(1, min(2, int(info.get("maxInputChannels", 2)))), info.get("name", "default input"))

    # --- video ---------------------------------------------------------------------------
    def vcam_options(self):
        """[(button label, device name for pyvirtualcam)]"""
        return [("Virtual Camera", None)]

    def resolve_vcam_device(self, requested):
        """Map a device name the phone asked for to one that exists on this OS."""
        names = [d for _, d in self.vcam_options()]
        return requested if requested in names else (names[0] if names else None)

    def vcam_help(self):
        return "Install OBS Studio and start its Virtual Camera once."

    def ensure_display_count(self, needed, width=None, height=None):
        """Try to provide at least `needed` monitors (virtual display). True if it did.
        width/height: the size the phone streams at, a hint for the new monitor's resolution."""
        return False

    def remove_virtual_displays(self):
        """Remove the virtual monitors ensure_display_count() added (phone stopped viewing)."""

    @property
    def virtual_displays_added(self):
        return 0


def run_quiet(cmd, timeout=5, **kw):
    """subprocess.run that never raises; returns CompletedProcess or None."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    except Exception:
        return None


def which(program):
    from shutil import which as _which
    return _which(program)



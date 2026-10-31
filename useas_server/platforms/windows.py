"""Windows: native SendInput/keybd_event input, ViGEm virtual Xbox pads, WASAPI loopback."""

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

from .base import Platform, VirtualGamepad, run_quiet

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP = 0x0020, 0x0040
MOUSEEVENTF_WHEEL = 0x0800
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

_BUTTON_FLAGS = {
    0: (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    1: (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    2: (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
}

# Virtual-key codes, including left/right modifier variants
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
    'meta': 0x5B, 'win': 0x5B, 'win_l': 0x5B, 'win_r': 0x5C,
}
PYAUTOGUI_MODIFIERS = {
    'shift_l': 'shiftleft', 'shift_r': 'shiftright',
    'ctrl_l': 'ctrlleft', 'ctrl_r': 'ctrlright',
    'alt_l': 'altleft', 'alt_r': 'altright',
    'meta': 'win', 'win_l': 'winleft', 'win_r': 'winright',
}


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.c_size_t)]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.c_size_t)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("u", _INPUTUNION)]


class XboxPad(VirtualGamepad):
    """Virtual Xbox 360 controller through the ViGEmBus driver."""

    supports_rumble = True
    description = "Virtual Xbox 360 controller"

    def __init__(self, on_rumble):
        import vgamepad as vg
        self.vg = vg
        self.pad = vg.VX360Gamepad()
        B = vg.XUSB_BUTTON
        self.map = {
            "a": B.XUSB_GAMEPAD_A, "b": B.XUSB_GAMEPAD_B, "x": B.XUSB_GAMEPAD_X, "y": B.XUSB_GAMEPAD_Y,
            "lb": B.XUSB_GAMEPAD_LEFT_SHOULDER, "rb": B.XUSB_GAMEPAD_RIGHT_SHOULDER,
            "back": B.XUSB_GAMEPAD_BACK, "start": B.XUSB_GAMEPAD_START, "guide": B.XUSB_GAMEPAD_GUIDE,
            "ls": B.XUSB_GAMEPAD_LEFT_THUMB, "rs": B.XUSB_GAMEPAD_RIGHT_THUMB,
            "dpad_up": B.XUSB_GAMEPAD_DPAD_UP, "dpad_down": B.XUSB_GAMEPAD_DPAD_DOWN,
            "dpad_left": B.XUSB_GAMEPAD_DPAD_LEFT, "dpad_right": B.XUSB_GAMEPAD_DPAD_RIGHT,
        }

        def callback(client, target, large_motor, small_motor, led_number, user_data):
            try:
                on_rumble(round(max(large_motor, small_motor) / 255.0, 2))
            except Exception:
                pass

        self._callback = callback
        self.pad.register_notification(callback_function=callback)

    def apply(self, norm):
        for name, pressed in norm["buttons"].items():
            code = self.map.get(name)
            if code is None:
                continue
            if pressed:
                self.pad.press_button(code)
            else:
                self.pad.release_button(code)
        # The phone sends +y = down; XInput wants +y = up.
        self.pad.left_joystick_float(x_value_float=norm["lx"], y_value_float=-norm["ly"])
        self.pad.right_joystick_float(x_value_float=norm["rx"], y_value_float=-norm["ry"])
        self.pad.left_trigger_float(value_float=norm["lt"])
        self.pad.right_trigger_float(value_float=norm["rt"])
        self.pad.update()

    def release_all(self):
        self.pad.reset()
        self.pad.update()

    def player_number(self):
        """XInput user index (P1-P4) from ViGEm; vgamepad doesn't wrap this call itself."""
        try:
            from vgamepad.win import vigem_client as vcli
            index = ctypes.c_ulong(0)
            err = vcli.vigem_target_x360_get_user_index(self.pad._busp, self.pad._devicep, ctypes.byref(index))
            if err == 0x20000000 and index.value < 4:   # VIGEM_ERROR_NONE
                return index.value + 1
        except Exception:
            pass
        return None

    def close(self):
        try:
            self.pad.unregister_notification()
        except Exception:
            pass
        try:
            self.release_all()
        except Exception:
            pass
        self.pad = None   # dropping the last reference unplugs it from ViGEmBus


class WindowsPlatform(Platform):
    name = "windows"
    display_name = "Windows"
    settings_dir = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")) / "UseAsServer"
    settings_filename = "server_settings.json"
    features = ("virtual_gamepad", "gamepad_rumble", "unicode_keys", "clipboard_private_filter",
                "virtual_display")

    def __init__(self):
        super().__init__()
        self.user32 = ctypes.windll.user32
        self._virtual_added = None        # loaded lazily from virtual_display.json
        self._idd_needs_admin = False

    # --- lifecycle -------------------------------------------------------------------
    def early_init(self):
        # Per-monitor DPI awareness: correct multi-monitor capture coordinates.
        try:
            self.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except Exception:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    self.user32.SetProcessDPIAware()
                except Exception:
                    pass
        # Never flash a console window for netsh/xdg helpers etc.
        if getattr(subprocess.Popen, "_useas_no_window", False):
            return
        orig = subprocess.Popen

        class _NoWindowPopen(orig):
            _useas_no_window = True

            def __init__(self, *args, **kwargs):
                kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
                super().__init__(*args, **kwargs)

        subprocess.Popen = _NoWindowPopen
        # PyAutoGUI sleeps 0.1 s after every call by default.
        try:
            import pyautogui
            pyautogui.PAUSE = 0
            pyautogui.FAILSAFE = False
        except Exception:
            pass

    # --- OS integration --------------------------------------------------------------
    def open_path(self, path):
        os.startfile(str(path))

    def set_autostart(self, enabled):
        import winreg
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                 r"Software\Microsoft\Windows\CurrentVersion\Run", 0, winreg.KEY_SET_VALUE)
            try:
                if enabled:
                    cmd = " ".join(f'"{p}"' for p in self.launch_command())
                    winreg.SetValueEx(key, "UseAsServer", 0, winreg.REG_SZ, cmd)
                else:
                    try:
                        winreg.DeleteValue(key, "UseAsServer")
                    except OSError:
                        pass
            finally:
                winreg.CloseKey(key)
            return True
        except Exception as e:
            self._log(f"⚠️ Couldn't change auto-start: {e}")
            return False

    send_to_label = "Explorer → Send to → Use As (phone)"

    def _send_to_link(self):
        appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(appdata) / "Microsoft" / "Windows" / "SendTo" / "Use As (phone).lnk"

    def install_send_to(self, enabled):
        """Explorer's right-click 'Send to' menu is a folder of shortcuts; Explorer appends
        the selected files to the shortcut's command line."""
        link = self._send_to_link()
        try:
            if not enabled:
                if link.exists():
                    link.unlink()
                return True
            cmd = self.send_to_command()
            target, args = cmd[0], " ".join(f'"{a}"' if " " in a else a for a in cmd[1:])
            icon = sys.executable if getattr(sys, "frozen", False) else ""
            ps = ("$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:USEAS_LNK);"
                  "$s.TargetPath=$env:USEAS_TARGET;$s.Arguments=$env:USEAS_ARGS;"
                  "$s.Description='Send to your phone with Use As';"
                  "if($env:USEAS_ICON){$s.IconLocation=$env:USEAS_ICON};$s.Save()")
            env = dict(os.environ, USEAS_LNK=str(link), USEAS_TARGET=target, USEAS_ARGS=args, USEAS_ICON=icon)
            link.parent.mkdir(parents=True, exist_ok=True)
            result = run_quiet(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], timeout=15, env=env)
            return result is not None and result.returncode == 0
        except Exception as e:
            self._log(f"⚠️ Couldn't update the Send to menu: {e}")
            return False

    def ensure_firewall_rule(self):
        """Fallback for elevated runs; the installer normally adds these rules."""
        if not self.is_admin():
            return
        # Firewall rules name the program that opens the sockets: the exe when built,
        # python.exe when run from source (a .py path never matches anything).
        exe_path = sys.executable
        suffix = "" if getattr(sys, "frozen", False) else "_Python"
        for proto, base in (("TCP", "UseAsServer_TCP"), ("UDP", "UseAsServer_UDP")):
            name = base + suffix
            existing = run_quiet(["netsh", "advfirewall", "firewall", "show", "rule", f"name={name}"])
            if existing is not None and existing.returncode == 0:
                continue
            run_quiet(["netsh", "advfirewall", "firewall", "add", "rule", f"name={name}", "dir=in",
                       "action=allow", f"program={exe_path}", "enable=yes", "profile=any", f"protocol={proto}"])

    def supports_admin(self):
        return True

    def is_admin(self):
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False

    def relaunch_as_admin(self, extra_args=()):
        """Start an elevated copy of this app. Handles the exe, `python UseAsServer.py`
        and `python -m useas_server` (the old code relaunched __main__.py directly, which
        can't import its own package and died instantly)."""
        extra = " ".join(f'"{a}"' if " " in a else a for a in extra_args)   # file paths may have spaces
        if getattr(sys, "frozen", False):
            exe, params, workdir = sys.executable, extra, os.path.dirname(sys.executable)
        else:
            script = os.path.abspath(sys.argv[0])
            if os.path.basename(script) == "__main__.py":
                package_dir = os.path.dirname(script)
                params = f"-m {os.path.basename(package_dir)} {extra}"
                workdir = os.path.dirname(package_dir)
            else:
                params = f'"{script}" {extra}'
                workdir = os.path.dirname(script)
            exe = sys.executable
        result = ctypes.windll.shell32.ShellExecuteW(None, "runas", exe, params.strip(), workdir, 1)
        return result > 32

    def style_window(self, root):
        """Dark title bar matching the theme (Windows 11)."""
        try:
            root.update_idletasks()
            hwnd = self.user32.GetParent(root.winfo_id())
            for attr, colour in ((35, 0x000d0d0d), (36, 0x0076e600)):
                value = ctypes.c_int(colour)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(value), ctypes.sizeof(value))
        except Exception:
            pass

    # --- clipboard -------------------------------------------------------------------
    def clipboard_sequence(self):
        try:
            return self.user32.GetClipboardSequenceNumber()
        except Exception:
            return None

    def clipboard_is_private(self):
        """KeePass/Bitwarden/1Password mark copied secrets with these formats."""
        try:
            for name in ("ExcludeClipboardContentFromMonitorProcessing", "Clipboard Viewer Ignore"):
                fmt = self.user32.RegisterClipboardFormatW(name)
                if fmt and self.user32.IsClipboardFormatAvailable(fmt):
                    return True
        except Exception:
            pass
        return False

    def set_clipboard_image(self, image):
        """Image as CF_DIB (every app) plus "PNG" (keeps transparency in apps that read it)."""
        import io
        try:
            import win32clipboard
        except ImportError:
            self._log("⚠️ pywin32 is needed to put images on the clipboard (pip install pywin32)")
            return False
        bmp = io.BytesIO()
        image.convert("RGB").save(bmp, "BMP")
        dib = bmp.getvalue()[14:]                 # CF_DIB = BMP without its file header
        png = io.BytesIO()
        image.save(png, "PNG")
        for _ in range(10):                       # another app may be holding the clipboard
            try:
                win32clipboard.OpenClipboard()
            except Exception:
                time.sleep(0.05)
                continue
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, dib)
                win32clipboard.SetClipboardData(win32clipboard.RegisterClipboardFormat("PNG"), png.getvalue())
                return True
            except Exception as e:
                self._log(f"⚠️ Couldn't set the clipboard image: {e}")
                return False
            finally:
                try:
                    win32clipboard.CloseClipboard()
                except Exception:
                    pass
        return False

    # --- virtual display (Amyuni usbmmidd) --------------------------------------------
    # The phone picks "Monitor 2" on a PC with one screen -> a real extra monitor is added
    # (it shows up in Display settings; windows can be dragged onto it). Removed again when
    # the phone stops viewing. Driver: https://www.amyuni.com (free, signed, up to 4 monitors).
    _IDD_SERVICE_KEY = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\WUDF\Services\usbmmIdd"
    _IDD_MAX = 4

    def _monitor_count(self):
        try:
            return int(self.user32.GetSystemMetrics(80))      # SM_CMONITORS
        except Exception:
            return 1

    def _idd_dir(self):
        """Folder with deviceinstaller64.exe + usbmmidd.inf: next to the installed exe, in the
        PyInstaller bundle, or (source runs) Final_Build. A lone zip is unpacked once."""
        if getattr(sys, "frozen", False):
            base = Path(sys.executable).parent
        else:
            base = Path(os.path.abspath(sys.argv[0])).parent
        candidates = [base / "drivers" / "usbmmidd", base / "Final_Build" / "usbmmidd_v2",
                      self.settings_dir / "usbmmidd" / "usbmmidd_v2"]
        bundle = getattr(sys, "_MEIPASS", None)
        if bundle:
            candidates.insert(1, Path(bundle) / "drivers" / "usbmmidd")
        for c in candidates:
            if (c / "usbmmIdd.inf").exists() and (c / "deviceinstaller64.exe").exists():
                return c
        for z in (base / "Final_Build" / "usbmmidd_v2.zip", base / "drivers" / "usbmmidd_v2.zip"):
            if z.exists():
                try:
                    import zipfile
                    target = self.settings_dir / "usbmmidd"
                    with zipfile.ZipFile(z) as zf:
                        zf.extractall(target)
                    found = next(target.rglob("usbmmIdd.inf"), None)
                    if found is not None:
                        return found.parent
                except Exception as e:
                    self._log(f"⚠️ Couldn't unpack {z.name}: {e}")
        return None

    def _devinst(self):
        folder = self._idd_dir()
        if folder is None:
            return None, None
        arch = (os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE") or "").upper()
        if arch == "ARM64":
            return None, None                      # the driver has no ARM build
        exe = folder / ("deviceinstaller.exe" if arch == "X86" else "deviceinstaller64.exe")
        return (exe, folder) if exe.exists() else (None, None)

    def _idd_installed(self):
        import winreg
        try:
            winreg.CloseKey(winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, self._IDD_SERVICE_KEY))
            return True
        except OSError:
            return False

    def _run_elevated(self, exe, params, workdir, timeout=90):
        """ShellExecuteEx "runas" (UAC prompt), hidden, waiting for it. Exit code, or None if
        the user said no."""
        from ctypes import wintypes

        class SHELLEXECUTEINFOW(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("fMask", wintypes.ULONG), ("hwnd", wintypes.HWND),
                        ("lpVerb", wintypes.LPCWSTR), ("lpFile", wintypes.LPCWSTR),
                        ("lpParameters", wintypes.LPCWSTR), ("lpDirectory", wintypes.LPCWSTR),
                        ("nShow", ctypes.c_int), ("hInstApp", wintypes.HINSTANCE), ("lpIDList", ctypes.c_void_p),
                        ("lpClass", wintypes.LPCWSTR), ("hkeyClass", wintypes.HKEY), ("dwHotKey", wintypes.DWORD),
                        ("hIconOrMonitor", wintypes.HANDLE), ("hProcess", wintypes.HANDLE)]

        info = SHELLEXECUTEINFOW()
        info.cbSize = ctypes.sizeof(info)
        info.fMask = 0x00000040 | 0x00008000      # NOCLOSEPROCESS | NO_CONSOLE
        info.lpVerb = "runas"
        info.lpFile = str(exe)
        info.lpParameters = params
        info.lpDirectory = str(workdir)
        info.nShow = 0                            # SW_HIDE
        if not ctypes.windll.shell32.ShellExecuteExW(ctypes.byref(info)):
            return None
        if not info.hProcess:
            return 0
        kernel32 = ctypes.windll.kernel32
        kernel32.WaitForSingleObject(info.hProcess, int(timeout * 1000))
        code = wintypes.DWORD(0)
        kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(code))
        kernel32.CloseHandle(info.hProcess)
        return int(code.value)

    def _devinst_run(self, args_list, elevated):
        """Run deviceinstaller once per argument list. Elevated runs are batched into one
        UAC prompt. Returns False only if elevation was refused or the tool is missing."""
        exe, folder = self._devinst()
        if exe is None:
            return False
        if elevated and not self.is_admin():
            if len(args_list) == 1:
                params, target = " ".join(args_list[0]), exe
            else:
                target = Path(os.environ.get("ComSpec", r"C:\Windows\System32\cmd.exe"))
                chain = " & ".join(f'"{exe}" ' + " ".join(a) for a in args_list)
                params = f'/c "{chain}"'
            return self._run_elevated(target, params, folder) is not None
        for args in args_list:
            run_quiet([str(exe)] + list(args), timeout=60, cwd=str(folder))
        return True

    def _wait_monitors(self, count, timeout):
        end = time.time() + timeout
        while time.time() < end:
            if self._monitor_count() >= count:
                return True
            time.sleep(0.25)
        return self._monitor_count() >= count

    # bookkeeping survives crashes, so leftover virtual monitors get removed next time
    def _state_file(self):
        return self.settings_dir / "virtual_display.json"

    @property
    def virtual_displays_added(self):
        if self._virtual_added is None:
            try:
                import json
                data = json.loads(self._state_file().read_text())
                self._virtual_added = max(0, min(self._IDD_MAX, int(data.get("added", 0))))
                self._idd_needs_admin = bool(data.get("needs_admin", False))
            except Exception:
                self._virtual_added = 0
        return self._virtual_added

    def _save_state(self, added):
        self._virtual_added = added
        try:
            import json
            self.settings_dir.mkdir(parents=True, exist_ok=True)
            self._state_file().write_text(json.dumps({"added": added, "needs_admin": self._idd_needs_admin}))
        except Exception:
            pass

    # The driver's 10th resolution slot ("9") is ours: set to the phone's shape (e.g. 2400x1080),
    # so the picture fills the phone instead of a 16:9 box with bars at the sides.
    _IDD_MONITORS_KEY = _IDD_SERVICE_KEY + r"\Parameters\Monitors"

    @staticmethod
    def phone_shaped_size(phone_size, height):
        """(width, height) of a monitor with the phone's aspect ratio at the stream's height,
        or None when the phone is already ~16:9 (the standard sizes fit it)."""
        if not phone_size or not height:
            return None
        pw, ph = max(phone_size), min(phone_size)
        if ph <= 0:
            return None
        aspect = pw / ph
        if abs(aspect - 16 / 9) < 0.06 or not 1.2 <= aspect <= 3.0:
            return None
        h = int(height)
        w = int(round(h * aspect / 8.0)) * 8
        return (w, h)

    def _ensure_custom_mode(self, size):
        """Write slot 9 = "W,H" if it isn't already. True if it changed (needs a re-add)."""
        import winreg
        value = f"{size[0]},{size[1]}"
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, self._IDD_MONITORS_KEY) as k:
                if winreg.QueryValueEx(k, "9")[0] == value:
                    return False
        except OSError:
            pass
        if self.is_admin():
            try:
                with winreg.CreateKeyEx(winreg.HKEY_LOCAL_MACHINE, self._IDD_MONITORS_KEY, 0,
                                        winreg.KEY_SET_VALUE) as k:
                    winreg.SetValueEx(k, "9", 0, winreg.REG_SZ, value)
                return True
            except OSError as e:
                self._log(f"⚠️ Couldn't add the phone-shaped resolution: {e}")
                return False
        self._log(f"🔐 Adding a {value.replace(',', 'x')} resolution for the virtual monitor — "
                  f"allow it in the Windows prompt")
        params = (f'add "HKLM\\{self._IDD_MONITORS_KEY}" /v 9 /t REG_SZ /d "{value}" /f')
        code = self._run_elevated(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "reg.exe"),
                                  params, os.environ.get("SystemRoot", r"C:\Windows"))
        return code == 0

    def ensure_display_count(self, needed, width=None, height=None, phone_size=None, exact_size=None):
        """exact_size: a resolution picked on the phone; otherwise the phone's shape."""
        have = self._monitor_count()
        if exact_size and self._idd_installed():
            custom = (int(exact_size[0]), int(exact_size[1]))
        else:
            custom = self.phone_shaped_size(phone_size, height) if self._idd_installed() else None
        if custom is not None and self._ensure_custom_mode(custom) and self.virtual_displays_added and have >= needed:
            # The driver reads its resolution list when a monitor is added: re-add ours.
            self._log("🖥️ Re-adding the virtual monitor with the new resolution…")
            self.remove_virtual_displays()
            have = self._monitor_count()
        if have >= needed:
            if self.virtual_displays_added and width and height:
                self._set_virtual_mode(*(custom or (width, height)))
            return True
        exe, _ = self._devinst()
        if exe is None:
            self._log("ℹ️ No virtual display driver found. Reinstall Use As Server with "
                      "\"Virtual monitor\" ticked to use the phone as an extra screen.")
            return False
        if not self._idd_installed():
            self._log("⚙️ Installing the virtual display driver (one time) — allow it in the Windows prompt…")
            if not self._devinst_run([["install", "usbmmidd.inf", "usbmmidd"]], elevated=True):
                self._log("❌ Virtual display driver not installed (permission refused)")
                return False
            if not self._idd_installed():
                self._log("❌ Virtual display driver install failed")
                return False

        added = self.virtual_displays_added
        missing = min(needed - have, self._IDD_MAX - added)
        if missing <= 0:
            self._log(f"⚠️ Already {added} virtual monitors (the driver's maximum is {self._IDD_MAX})")
            return False
        self._log(f"🖥️ Adding {missing} virtual monitor(s) for the phone…")
        target = have + missing
        ok = refused = False
        if not self._idd_needs_admin or self.is_admin():
            self._devinst_run([["enableidd", "1"]] * missing, elevated=False)
            ok = self._wait_monitors(target, 6)
        if not ok and not self.is_admin():
            # Some systems only allow it elevated: one UAC prompt, remembered for next time.
            self._idd_needs_admin = True
            self._log("🔐 Windows needs permission to add the virtual monitor…")
            still = target - self._monitor_count()
            if still > 0:
                if self._devinst_run([["enableidd", "1"]] * still, elevated=True):
                    ok = self._wait_monitors(target, 8)
                else:
                    refused = True
        now = self._monitor_count()
        self._save_state(added + max(0, now - have))
        if not ok:
            self._log("❌ No virtual monitor: Windows permission was refused" if refused
                      else "❌ The virtual monitor didn't appear")
            return False
        if width and height:
            self._set_virtual_mode(*(custom or (width, height)))
        time.sleep(1.0)    # let the desktop settle before capturing it
        self._log("✅ Virtual monitor ready — it's in Settings › Display like a real screen; "
                  "drag windows onto it")
        return True

    def remove_virtual_displays(self):
        added = self.virtual_displays_added
        if not added:
            return
        before = self._monitor_count()
        ok = self._devinst_run([["enableidd", "0"]] * added,
                               elevated=self._idd_needs_admin and not self.is_admin())
        if ok:
            self._wait_monitors_below(before - added + 1, 5)
            self._save_state(0)
            self._log(f"🖥️ Removed {added} virtual monitor(s)")
        else:
            self._log("⚠️ Virtual monitor left in place (permission refused)")

    def _wait_monitors_below(self, count, timeout):
        end = time.time() + timeout
        while time.time() < end and self._monitor_count() >= count:
            time.sleep(0.25)

    @staticmethod
    def _pick_mode(modes, width, height):
        """Best driver mode for a phone streaming at width x height: that exact size, else the
        smallest mode with the same shape (16:9) at least that big, else the largest one with
        that shape. A 4:3 monitor (Windows sometimes picks 1024x768) would be letterboxed
        into a much smaller picture on the phone."""
        if (width, height) in modes:
            return width, height
        aspect = width / height
        same = [m for m in modes if abs(m[0] / m[1] - aspect) < 0.03]
        bigger = sorted((m for m in same if m[0] >= width), key=lambda m: m[0] * m[1])
        if bigger:
            return bigger[0]
        if same:
            return max(same, key=lambda m: m[0] * m[1])
        return (1920, 1080) if (1920, 1080) in modes else None

    def _set_virtual_mode(self, width, height):
        """Give the virtual monitor the phone stream's shape (see _pick_mode)."""
        try:
            import pywintypes
            import win32api
            import win32con
        except ImportError:
            self._log("⚠️ Install pywin32 (pip install pywin32) so the virtual monitor's "
                      "resolution can match the phone")
            return
        found = False
        i = 0
        while True:
            try:
                dev = win32api.EnumDisplayDevices(None, i)
            except pywintypes.error:
                break
            i += 1
            ident = f"{dev.DeviceID} {dev.DeviceString}".lower()
            if "usbmm" not in ident and "mobile monitor" not in ident:
                continue
            found = True
            modes = set()
            j = 0
            while True:
                try:
                    m = win32api.EnumDisplaySettings(dev.DeviceName, j)
                except pywintypes.error:
                    break
                modes.add((int(m.PelsWidth), int(m.PelsHeight)))
                j += 1
            best = self._pick_mode(modes, int(width), int(height))
            if best is None:
                continue
            try:
                cur = win32api.EnumDisplaySettings(dev.DeviceName, win32con.ENUM_CURRENT_SETTINGS)
                if (int(cur.PelsWidth), int(cur.PelsHeight)) == best:
                    continue
                cur.PelsWidth, cur.PelsHeight = best
                cur.Fields = win32con.DM_PELSWIDTH | win32con.DM_PELSHEIGHT
                result = win32api.ChangeDisplaySettingsEx(dev.DeviceName, cur, win32con.CDS_UPDATEREGISTRY)
                if result == win32con.DISP_CHANGE_SUCCESSFUL:
                    self._log(f"🖥️ Virtual monitor set to {best[0]}x{best[1]}")
                    time.sleep(0.8)          # let Windows apply it before the capture starts
                else:
                    self._log(f"⚠️ Couldn't change the virtual monitor's resolution (code {result})")
            except Exception as e:
                self._log(f"⚠️ Couldn't change the virtual monitor's resolution: {e}")
        if not found:
            self._log("⚠️ Virtual monitor not found among the display adapters — "
                      "set its resolution in Settings › Display")

    # --- input -------------------------------------------------------------------------
    def mouse_move_rel(self, dx, dy):
        # Relative mouse_event feels smoother than SetCursorPos and works in games.
        self.user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)

    def mouse_button(self, button, action, gaming_mode=False):
        down, up = _BUTTON_FLAGS.get(button, _BUTTON_FLAGS[0])
        if action == "down":
            self.user32.mouse_event(down, 0, 0, 0, 0)
        elif action == "up":
            self.user32.mouse_event(up, 0, 0, 0, 0)
        else:
            self.user32.mouse_event(down, 0, 0, 0, 0)
            if gaming_mode:
                time.sleep(0.04)   # games miss 0 ms clicks
            self.user32.mouse_event(up, 0, 0, 0, 0)

    def mouse_scroll(self, scroll_delta):
        wheel = int(scroll_delta * 12)   # WHEEL_DELTA = 120 = 10 phone units
        if wheel:
            self.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, wheel, 0)

    def mouse_zoom(self, notches):
        VK_CONTROL = 0x11
        self.user32.keybd_event(VK_CONTROL, 0, 0, 0)
        try:
            self.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(notches * 120), 0)
        finally:
            self.user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)

    def key_event(self, key, modifiers, action, gaming_mode=False):
        key_lower = key.lower()
        vk_code = VK_MAP.get(key_lower, 0)
        if vk_code:
            scan_code = self.user32.MapVirtualKeyW(vk_code, 0)

            def mods(is_down):
                flags = 0 if is_down else KEYEVENTF_KEYUP
                for mod in modifiers:
                    mod_vk = VK_MAP.get(mod.lower(), 0)
                    if mod_vk:
                        self.user32.keybd_event(mod_vk, self.user32.MapVirtualKeyW(mod_vk, 0), flags, 0)

            if action == "down":
                mods(True)
                self.user32.keybd_event(vk_code, scan_code, 0, 0)
            elif action == "up":
                self.user32.keybd_event(vk_code, scan_code, KEYEVENTF_KEYUP, 0)
                mods(False)
            else:
                mods(True)
                self.user32.keybd_event(vk_code, scan_code, 0, 0)
                if gaming_mode:
                    time.sleep(0.03)
                self.user32.keybd_event(vk_code, scan_code, KEYEVENTF_KEYUP, 0)
                mods(False)
            return

        # Everything else (F-keys, arrows by name, media keys, punctuation): PyAutoGUI.
        import pyautogui
        if key_lower in ("caps", "caps lock"):
            key_lower = "capslock"
        keys = [PYAUTOGUI_MODIFIERS.get(m.lower(), m.lower()) for m in modifiers] + [key_lower]
        if action == "down":
            for k in keys:
                pyautogui.keyDown(k)
        elif action == "up":
            for k in reversed(keys):
                pyautogui.keyUp(k)
        else:
            pyautogui.hotkey(*keys)

    def type_unicode(self, ch, action="press"):
        """SendInput + KEYEVENTF_UNICODE: any character (é, emoji, हिन्दी)."""
        # surrogatepass: the phone sends emoji one UTF-16 half at a time.
        units = ch.encode("utf-16-le", "surrogatepass")
        codes = [int.from_bytes(units[i:i + 2], "little") for i in range(0, len(units), 2)]
        events = []
        for code in codes:
            if action in ("press", "click", "down"):
                events.append((code, KEYEVENTF_UNICODE))
            if action in ("press", "click", "up"):
                events.append((code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
        if not events:
            return
        arr = (_INPUT * len(events))()
        for i, (code, flags) in enumerate(events):
            arr[i].type = 1   # INPUT_KEYBOARD
            arr[i].u.ki = _KEYBDINPUT(0, code, flags, 0, 0)
        self.user32.SendInput(len(events), arr, ctypes.sizeof(_INPUT))

    # --- gamepad -----------------------------------------------------------------------
    def create_gamepad(self, on_rumble):
        try:
            return XboxPad(on_rumble)
        except Exception as e:
            self._log(f"❌ ViGEmBus Error (Is it installed?): {e}")
            return None

    # --- audio -------------------------------------------------------------------------
    def mic_output(self, pa):
        """VB-CABLE's 'CABLE Input' makes the phone mic available to Zoom/Discord/etc."""
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                name = info.get("name", "")
                if info.get("maxOutputChannels", 0) > 0 and ("CABLE Input" in name or "VB-Audio" in name):
                    return info["index"], 1, name
        except Exception:
            pass
        return None, 1, "default speakers (install VB-CABLE to use it as a microphone)"

    @staticmethod
    def _default_device_ids():
        """Windows' default playback device IDs; a change means the user switched output."""
        ids = {}
        try:
            import winreg
            base = r"Software\Microsoft\Multimedia\Audio\DefaultRole"
            for role in ("0", "1", "2"):
                try:
                    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, f"{base}\\{role}") as key:
                        ids[role] = winreg.QueryValueEx(key, "Device")[0]
                except OSError:
                    ids[role] = "unknown"
        except Exception:
            pass
        return ids

    @staticmethod
    def _clean_name(n):
        s = (n or "").lower()
        for junk in ("[loopback]", "loopback", "(default)", "(default output)"):
            s = s.replace(junk, "")
        return " ".join(s.replace("(", "").replace(")", "").replace("-", " ").replace("_", " ").split())

    def _find_loopback(self, pa, default_name):
        try:
            loopbacks = list(pa.get_loopback_device_info_generator())
        except Exception:
            return None
        if not loopbacks:
            return None
        want = self._clean_name(default_name)
        for match in (lambda n: n == want, lambda n: want in n, lambda n: n in want):
            for lb in loopbacks:
                if match(self._clean_name(lb["name"])):
                    return lb
        return loopbacks[0]

    def run_loopback_capture(self, target_rate, still_wanted, restart_event, emit):
        """WASAPI loopback of the default output; returns when the output device changes,
        the rate should change (restart_event) or streaming stops."""
        import pyaudiowpatch as pyaudio
        ids_at_start = self._default_device_ids()
        pa = pyaudio.PyAudio()
        stream = None
        try:
            try:
                default_name = pa.get_default_output_device_info().get("name", "Unknown")
            except Exception:
                default_name = "Unknown"
            loopback = self._find_loopback(pa, default_name)
            if loopback is None:
                self._log("❌ No loopback device found — retrying…")
                time.sleep(1.0)
                return
            channels = int(min(loopback.get("maxInputChannels", 2), 2)) or 2
            rate = int(target_rate)

            def callback(in_data, frame_count, time_info, status):
                try:
                    emit(in_data, rate, channels)
                    return None, pyaudio.paContinue
                except Exception:
                    return None, pyaudio.paAbort

            try:
                stream = pa.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True,
                                 input_device_index=loopback["index"], frames_per_buffer=2048,
                                 stream_callback=callback)
            except OSError:
                rate = int(loopback.get("defaultSampleRate", 44100))
                self._log(f"⚠️ {target_rate}Hz not supported by this device — using {rate}Hz")
                stream = pa.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True,
                                 input_device_index=loopback["index"], frames_per_buffer=2048,
                                 stream_callback=callback)
            self._log(f"🔊 Capturing {default_name} @ {rate}Hz")
            stream.start_stream()
            while still_wanted() and stream.is_active():
                if restart_event.wait(timeout=1.0):
                    return
                if self._default_device_ids() != ids_at_start:
                    self._log("🔄 Output device changed — switching")
                    return
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            pa.terminate()

    # --- video -------------------------------------------------------------------------
    def vcam_options(self):
        return [("OBS Camera", "OBS Virtual Camera"), ("Unity Camera", "Unity Video Capture")]

    def resolve_vcam_device(self, requested):
        return requested or "OBS Virtual Camera"

    def vcam_help(self):
        return "Install OBS Studio (or Unity Capture) to get a virtual camera driver."

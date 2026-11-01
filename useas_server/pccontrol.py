"""Remote PC controls for the phone's PC tab: power, volume, open windows and app launching.

Every function is safe to call on any OS: what isn't available returns None / [] / False
(the phone only shows what the server's hello says it supports).

Windows volume needs ``pip install pycaw``; Linux uses pactl / loginctl / systemctl / wmctrl;
macOS uses osascript / pmset.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

WINDOWS = sys.platform == "win32"
MAC = sys.platform == "darwin"
LINUX = not WINDOWS and not MAC

SHUTDOWN_DELAY = 15     # seconds: the phone shows a Cancel button meanwhile


def _run(cmd, timeout=5):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None


def _which(cmd):
    from shutil import which
    return which(cmd) is not None


# ------------------------------------------------------------------------------------------
# Power
# ------------------------------------------------------------------------------------------

POWER_ACTIONS = ("lock", "sleep", "hibernate", "screen_off", "restart", "shutdown", "cancel_shutdown",
                 "sign_out")


def power(action):
    """Returns (ok, message)."""
    if action not in POWER_ACTIONS:
        return False, f"unknown action {action}"
    try:
        if WINDOWS:
            return _power_windows(action)
        if MAC:
            return _power_mac(action)
        return _power_linux(action)
    except Exception as e:
        return False, str(e)


def _power_windows(action):
    import ctypes
    user32 = ctypes.windll.user32
    if action == "lock":
        return bool(user32.LockWorkStation()), "Locked"
    if action == "screen_off":
        # PostMessage, not SendMessage: a hung window can't block us. 2 = monitor off.
        user32.PostMessageW(0xFFFF, 0x0112, 0xF170, 2)
        return True, "Screen off"
    if action == "sleep":
        # SetSuspendState(hibernate=False, force=False, disableWake=False)
        threading_sleep(lambda: ctypes.windll.powrprof.SetSuspendState(0, 0, 0))
        return True, "Going to sleep"
    if action == "hibernate":
        if not hibernate_supported():
            return False, "Hibernate is turned off on this PC (turn it on with: powercfg /hibernate on)"
        threading_sleep(lambda: _run(["shutdown", "/h"]))
        return True, "Hibernating"
    if action == "shutdown":
        r = _run(["shutdown", "/s", "/t", str(SHUTDOWN_DELAY), "/c", "Shut down from your phone (Use As)"])
        return r is not None and r.returncode == 0, f"Shutting down in {SHUTDOWN_DELAY} s"
    if action == "restart":
        r = _run(["shutdown", "/r", "/t", str(SHUTDOWN_DELAY), "/c", "Restart from your phone (Use As)"])
        return r is not None and r.returncode == 0, f"Restarting in {SHUTDOWN_DELAY} s"
    if action == "cancel_shutdown":
        r = _run(["shutdown", "/a"])
        return r is not None and r.returncode == 0, "Cancelled"
    if action == "sign_out":
        threading_sleep(lambda: _run(["shutdown", "/l"]))
        return True, "Signing out"
    return False, "unsupported"


def _power_linux(action):
    if action == "lock":
        for cmd in (["loginctl", "lock-session"], ["xdg-screensaver", "lock"], ["gnome-screensaver-command", "-l"]):
            if _which(cmd[0]):
                r = _run(cmd)
                if r is not None and r.returncode == 0:
                    return True, "Locked"
        return False, "No lock command found"
    if action == "screen_off":
        r = _run(["xset", "dpms", "force", "off"])
        return r is not None and r.returncode == 0, "Screen off"
    if action == "sleep":
        threading_sleep(lambda: _run(["systemctl", "suspend"]))
        return True, "Going to sleep"
    if action == "hibernate":
        threading_sleep(lambda: _run(["systemctl", "hibernate"]))
        return True, "Hibernating"
    if action == "shutdown":
        r = _run(["shutdown", "-h", f"+{max(1, SHUTDOWN_DELAY // 60)}"])
        return r is not None and r.returncode == 0, "Shutting down in 1 minute"
    if action == "restart":
        r = _run(["shutdown", "-r", f"+{max(1, SHUTDOWN_DELAY // 60)}"])
        return r is not None and r.returncode == 0, "Restarting in 1 minute"
    if action == "cancel_shutdown":
        r = _run(["shutdown", "-c"])
        return r is not None and r.returncode == 0, "Cancelled"
    if action == "sign_out":
        r = _run(["loginctl", "terminate-user", os.environ.get("USER", "")])
        return r is not None and r.returncode == 0, "Signing out"
    return False, "unsupported"


_mac_pending = {"timer": None}


def _power_mac(action):
    import threading
    if action == "lock":
        r = _run(["pmset", "displaysleepnow"])
        return r is not None and r.returncode == 0, "Locked"
    if action == "screen_off":
        r = _run(["pmset", "displaysleepnow"])
        return r is not None and r.returncode == 0, "Screen off"
    if action == "sleep":
        threading_sleep(lambda: _run(["pmset", "sleepnow"]))
        return True, "Going to sleep"
    if action == "hibernate":
        return False, "Hibernate isn't available on macOS"
    if action in ("shutdown", "restart", "sign_out"):
        verb = {"shutdown": "shut down", "restart": "restart", "sign_out": "log out"}[action]
        t = threading.Timer(SHUTDOWN_DELAY, lambda: _run(
            ["osascript", "-e", f'tell application "System Events" to {verb}']))
        t.daemon = True
        t.start()
        _mac_pending["timer"] = t
        return True, f"{verb.capitalize()} in {SHUTDOWN_DELAY} s"
    if action == "cancel_shutdown":
        t = _mac_pending.get("timer")
        if t is not None:
            t.cancel()
            _mac_pending["timer"] = None
            return True, "Cancelled"
        return False, "Nothing to cancel"
    return False, "unsupported"


def hibernate_supported():
    """Windows: hibernation is switched on; Linux: systemd is there to do it."""
    if WINDOWS:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Power") as k:
                return bool(winreg.QueryValueEx(k, "HibernateEnabled")[0])
        except OSError:
            return False
    if MAC:
        return False
    return _which("systemctl")


# --- timed power actions ("shut down in 1 hour") -------------------------------------------
TIMER_ACTIONS = ("shutdown", "restart", "sleep", "hibernate", "lock", "screen_off", "sign_out")
_timer = {"t": None, "action": None, "at": 0.0}
_timer_lock = threading.Lock()


def schedule_power(action, minutes, on_fire=None):
    """Run power(action) after `minutes`. Replaces an earlier timer. Returns seconds."""
    if action not in TIMER_ACTIONS:
        return 0
    seconds = max(60, min(int(float(minutes) * 60), 24 * 3600))

    def fire():
        with _timer_lock:
            if _timer["t"] is not t:
                return
            _timer.update(t=None, action=None, at=0.0)
        result = power(action)
        if on_fire:
            on_fire(action, result)

    with _timer_lock:
        if _timer["t"] is not None:
            _timer["t"].cancel()
        t = threading.Timer(seconds, fire)
        t.daemon = True
        _timer.update(t=t, action=action, at=time.time() + seconds)
        t.start()
    return seconds


def scheduled_power():
    """{"action", "seconds"} of the pending timer, or None."""
    with _timer_lock:
        if _timer["t"] is None:
            return None
        return {"action": _timer["action"], "seconds": max(0, int(_timer["at"] - time.time()))}


def cancel_scheduled_power():
    with _timer_lock:
        t = _timer["t"]
        _timer.update(t=None, action=None, at=0.0)
    if t is not None:
        t.cancel()
        return True
    return False


def threading_sleep(fn, delay=1.0):
    """Run fn a moment later on its own thread, so the phone gets its answer first."""
    import threading

    def run():
        time.sleep(delay)
        fn()
    threading.Thread(target=run, daemon=True).start()


# ------------------------------------------------------------------------------------------
# Volume (0-100) and mute
# ------------------------------------------------------------------------------------------

def _win_endpoint():
    from pycaw.pycaw import AudioUtilities
    speakers = AudioUtilities.GetSpeakers()
    ep = getattr(speakers, "EndpointVolume", None)       # pycaw >= 20240316
    if ep is not None:
        return ep
    from ctypes import POINTER, cast
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import IAudioEndpointVolume
    interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def _win_call(fn):
    """pycaw is COM: initialise COM on this (worker) thread for the call."""
    import comtypes
    try:
        comtypes.CoInitialize()
    except Exception:
        pass
    try:
        return fn(_win_endpoint())
    finally:
        try:
            comtypes.CoUninitialize()
        except Exception:
            pass


def volume_supported():
    if WINDOWS:
        try:
            import pycaw.pycaw  # noqa: F401
            return True
        except Exception:
            return False
    if MAC:
        return True
    return _which("pactl")


def get_volume():
    """(level 0-100, muted) or None."""
    try:
        if WINDOWS:
            return _win_call(lambda ep: (round(ep.GetMasterVolumeLevelScalar() * 100), bool(ep.GetMute())))
        if MAC:
            r = _run(["osascript", "-e", "set s to get volume settings\n"
                                         "return (output volume of s as text) & \",\" & (output muted of s as text)"])
            if r and r.returncode == 0 and "," in r.stdout:
                level, muted = r.stdout.strip().split(",", 1)
                return int(level), muted.strip() == "true"
            return None
        r = _run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
        m = _run(["pactl", "get-sink-mute", "@DEFAULT_SINK@"])
        if r and r.returncode == 0:
            import re
            pct = re.search(r"(\d+)%", r.stdout)
            return (int(pct.group(1)) if pct else 0,
                    bool(m and "yes" in m.stdout.lower()))
    except Exception:
        return None
    return None


def set_volume(level):
    level = max(0, min(100, int(level)))
    try:
        if WINDOWS:
            def apply(ep):
                ep.SetMasterVolumeLevelScalar(level / 100.0, None)
                if level > 0 and ep.GetMute():
                    ep.SetMute(0, None)
            _win_call(apply)
            return True
        if MAC:
            return (_run(["osascript", "-e", f"set volume output volume {level}"]) or None) is not None
        return (_run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"]) or None) is not None
    except Exception:
        return False


def toggle_mute():
    try:
        if WINDOWS:
            _win_call(lambda ep: ep.SetMute(0 if ep.GetMute() else 1, None))
            return True
        if MAC:
            state = get_volume()
            muted = state[1] if state else False
            return (_run(["osascript", "-e", f"set volume output muted {'false' if muted else 'true'}"])
                    or None) is not None
        return (_run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"]) or None) is not None
    except Exception:
        return False


# ------------------------------------------------------------------------------------------
# Per-app volume (Windows volume mixer / PulseAudio sink inputs)
# ------------------------------------------------------------------------------------------

def mixer_supported():
    if WINDOWS:
        return volume_supported()
    if MAC:
        return False
    return _which("pactl")


def _win_sessions():
    """{id: {"name", "exe", "sessions": [ISimpleAudioVolume], "active"}} grouped by program."""
    from pycaw.pycaw import AudioUtilities
    groups = {}
    for s in AudioUtilities.GetAllSessions():
        try:
            vol = s.SimpleAudioVolume
            proc = s.Process
            if proc is None:
                sid, name, exe = "system", "System sounds", ""
            else:
                exe_name = proc.name()
                sid = exe_name.lower()
                name = exe_name[:-4] if exe_name.lower().endswith(".exe") else exe_name
                try:
                    exe = proc.exe()
                except Exception:
                    exe = ""
            g = groups.setdefault(sid, {"name": name, "exe": exe, "sessions": [], "active": False})
            g["sessions"].append(vol)
            g["active"] = g["active"] or getattr(s, "State", 0) == 1
        except Exception:
            continue
    return groups


def _with_com(fn):
    import comtypes
    try:
        comtypes.CoInitialize()
    except Exception:
        pass
    try:
        return fn()
    finally:
        try:
            comtypes.CoUninitialize()
        except Exception:
            pass


def _pactl_inputs():
    r = _run(["pactl", "-f", "json", "list", "sink-inputs"])
    if not r or r.returncode != 0 or not r.stdout.strip().startswith("["):
        return []
    try:
        return json.loads(r.stdout)
    except ValueError:
        return []


def _pactl_level(inp):
    try:
        vols = inp.get("volume", {}).values()
        pcts = [int(str(v.get("value_percent", "0")).rstrip("%")) for v in vols]
        return round(sum(pcts) / len(pcts)) if pcts else 0
    except Exception:
        return 0


def list_sessions():
    """[{"id", "name", "volume", "muted", "active"}] of programs playing (or able to play) sound."""
    try:
        if WINDOWS:
            def read():
                out = []
                for sid, g in _win_sessions().items():
                    first = g["sessions"][0]
                    out.append({"id": sid, "name": g["name"], "volume": round(first.GetMasterVolume() * 100),
                                "muted": bool(first.GetMute()), "active": g["active"]})
                    if g["exe"]:
                        _program_sources["session:" + sid] = g["exe"]
                return out
            items = _with_com(read)
        elif MAC:
            return []
        else:
            groups = {}
            for inp in _pactl_inputs():
                props = inp.get("properties", {})
                name = props.get("application.name") or props.get("application.process.binary") or "Unknown"
                sid = name.lower()
                g = groups.setdefault(sid, {"id": sid, "name": name, "volume": _pactl_level(inp),
                                            "muted": bool(inp.get("mute")), "active": not inp.get("corked", False)})
                icon = props.get("application.icon_name") or props.get("application.process.binary")
                if icon:
                    _program_sources["session:" + sid] = ("icon", icon)
            items = list(groups.values())
        items.sort(key=lambda x: (not x["active"], x["id"] == "system", x["name"].lower()))
        return items
    except Exception:
        return []


def set_session_volume(sid, level):
    level = max(0, min(100, int(level)))
    try:
        if WINDOWS:
            def apply():
                g = _win_sessions().get(sid)
                for vol in (g or {}).get("sessions", []):
                    vol.SetMasterVolume(level / 100.0, None)
                    if level > 0 and vol.GetMute():
                        vol.SetMute(0, None)
                return bool(g)
            return _with_com(apply)
        ok = False
        for inp in _pactl_inputs():
            props = inp.get("properties", {})
            name = props.get("application.name") or props.get("application.process.binary") or "Unknown"
            if name.lower() == sid:
                _run(["pactl", "set-sink-input-volume", str(inp.get("index")), f"{level}%"])
                ok = True
        return ok
    except Exception:
        return False


def toggle_session_mute(sid):
    try:
        if WINDOWS:
            def apply():
                g = _win_sessions().get(sid)
                if not g:
                    return False
                new = 0 if g["sessions"][0].GetMute() else 1
                for vol in g["sessions"]:
                    vol.SetMute(new, None)
                return True
            return _with_com(apply)
        inputs = [i for i in _pactl_inputs()
                  if (i.get("properties", {}).get("application.name")
                      or i.get("properties", {}).get("application.process.binary") or "Unknown").lower() == sid]
        if not inputs:
            return False
        new = "0" if inputs[0].get("mute") else "1"
        for inp in inputs:
            _run(["pactl", "set-sink-input-mute", str(inp.get("index")), new])
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------------------
# Screen brightness (laptop panels, and monitors that support DDC/CI)
# ------------------------------------------------------------------------------------------

def brightness_supported():
    if MAC:
        return False
    try:
        import screen_brightness_control  # noqa: F401
        return True
    except Exception:
        return False


def get_brightness():
    """0-100 of the first screen, or None when it can't be read."""
    try:
        import screen_brightness_control as sbc
        value = sbc.get_brightness(display=0)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        return None if value is None else int(value)
    except Exception:
        return None


def set_brightness(level):
    try:
        import screen_brightness_control as sbc
        sbc.set_brightness(max(0, min(100, int(level))))
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------------------
# Open windows
# ------------------------------------------------------------------------------------------

def windows_supported():
    return WINDOWS or MAC or _which("wmctrl")


# window id -> where its icon comes from (an .exe / .app path, or ("icon", name) on Linux)
_window_sources = {}
# "session:<id>" -> the same, for the volume mixer
_program_sources = {}


def window_source(win_id):
    return _window_sources.get(str(win_id))


def program_source(key):
    return _program_sources.get(str(key))


def list_windows():
    """[{"id": str, "title": str, "app": str}] of the user's normal windows, front-most first."""
    try:
        if WINDOWS:
            return _list_windows_win()
        if MAC:
            return _list_windows_mac()
        return _list_windows_linux()
    except Exception:
        return []


def _list_windows_win():
    import ctypes
    from ctypes import wintypes
    import psutil
    user32 = ctypes.windll.user32
    dwm = ctypes.windll.dwmapi
    out = []
    own_pid = os.getpid()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.GetWindow(hwnd, 4):   # GW_OWNER: skip dialogs
            return True
        ex = user32.GetWindowLongW(hwnd, -20)                               # GWL_EXSTYLE
        if ex & 0x80:                                                      # WS_EX_TOOLWINDOW
            return True
        cloaked = ctypes.c_int(0)
        dwm.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))  # DWMWA_CLOAKED
        if cloaked.value:
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == own_pid:
            return True
        exe = ""
        try:
            proc = psutil.Process(pid.value)
            app = proc.name()
            app = app[:-4] if app.lower().endswith(".exe") else app
            try:
                exe = proc.exe()
            except Exception:
                exe = ""
        except Exception:
            app = ""
        if title == "Program Manager":
            return True
        wid = str(int(hwnd) if hwnd else 0)
        if exe:
            _window_sources[wid] = exe
        out.append({"id": wid, "title": title, "app": app,
                    "minimized": bool(user32.IsIconic(hwnd)), "maximized": bool(user32.IsZoomed(hwnd))})
        return True

    user32.EnumWindows(callback, 0)     # Z-order: front-most first
    return out[:60]


def _list_windows_linux():
    r = _run(["wmctrl", "-lx"])
    if not r or r.returncode != 0:
        return []
    out = []
    for line in r.stdout.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5 or parts[1] == "-1":       # -1 = sticky panels/desktop
            continue
        wid, _, wmclass, _host, title = parts
        _window_sources[wid] = ("icon", wmclass.split(".")[0])
        out.append({"id": wid, "title": title, "app": wmclass.split(".")[-1], "minimized": False})
    return out[:60]


def _list_windows_mac():
    r = _run(["osascript", "-e",
              'tell application "System Events" to get name of every application process '
              'whose background only is false'])
    if not r or r.returncode != 0:
        return []
    names = [n.strip() for n in r.stdout.strip().split(",") if n.strip()]
    for n in names:
        if n not in _window_sources:
            from . import thumbs
            _window_sources[n] = thumbs.mac_app_path(n) or ""
    return [{"id": n, "title": n, "app": n, "minimized": False} for n in names if n != "Use As Server"]


def focus_window(win_id):
    try:
        if WINDOWS:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = int(win_id)
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)                  # SW_RESTORE
            # Windows only lets the foreground app hand over focus; a tap of Alt counts as input.
            user32.keybd_event(0x12, 0, 0, 0)
            user32.keybd_event(0x12, 0, 2, 0)
            return bool(user32.SetForegroundWindow(hwnd))
        if MAC:
            r = _run(["osascript", "-e", f'tell application "{win_id}" to activate'])
            return r is not None and r.returncode == 0
        r = _run(["wmctrl", "-i", "-a", str(win_id)])
        return r is not None and r.returncode == 0
    except Exception:
        return False


def window_actions_supported():
    return WINDOWS or (LINUX and _which("wmctrl"))


def set_window_state(win_id, state):
    """state: minimize / maximize / restore."""
    try:
        if WINDOWS:
            import ctypes
            cmd = {"minimize": 6, "maximize": 3, "restore": 9}.get(state)     # SW_*
            if cmd is None:
                return False
            ctypes.windll.user32.ShowWindow(int(win_id), cmd)
            return True
        if MAC:
            return False
        if state == "minimize":
            if _which("xdotool"):
                r = _run(["xdotool", "windowminimize", str(int(str(win_id), 16))])
                return r is not None and r.returncode == 0
            return False
        flag = "add" if state == "maximize" else "remove"
        r = _run(["wmctrl", "-i", "-r", str(win_id), "-b", f"{flag},maximized_vert,maximized_horz"])
        return r is not None and r.returncode == 0
    except Exception:
        return False


def close_window(win_id):
    try:
        if WINDOWS:
            import ctypes
            return bool(ctypes.windll.user32.PostMessageW(int(win_id), 0x0010, 0, 0))   # WM_CLOSE
        if MAC:
            r = _run(["osascript", "-e", f'tell application "{win_id}" to quit'])
            return r is not None and r.returncode == 0
        r = _run(["wmctrl", "-i", "-c", str(win_id)])
        return r is not None and r.returncode == 0
    except Exception:
        return False


# ------------------------------------------------------------------------------------------
# App launcher
# ------------------------------------------------------------------------------------------

_SKIP_WORDS = ("uninstall", "readme", "read me", "help", "documentation", "license", "website",
               "release notes", "changelog", "manual", "support")
_launcher_cache = {"at": 0.0, "items": []}


def list_launchers():
    """[{"id": str, "name": str}] sorted by name. Cached for a minute."""
    if time.time() - _launcher_cache["at"] < 300 and _launcher_cache["items"]:
        return _launcher_cache["items"]
    try:
        if WINDOWS:
            items = _launchers_win()
        elif MAC:
            items = _launchers_mac()
        else:
            items = _launchers_linux()
    except Exception:
        items = []
    seen = set()
    unique = []
    for it in sorted(items, key=lambda x: x["name"].lower()):
        key = it["name"].lower()
        if key in seen or any(w in key for w in _SKIP_WORDS):
            continue
        seen.add(key)
        unique.append(it)
    _launcher_cache.update(at=time.time(), items=unique[:800])
    return _launcher_cache["items"]


def _launchers_win():
    """Everything in the Start menu, Microsoft Store apps included (Get-StartApps); the plain
    shortcut scan is the fallback."""
    items = _start_apps_win()
    return items if items else _shortcut_launchers_win()


def _start_apps_win():
    script = ("[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
              "Get-StartApps | Select-Object Name,AppID | ConvertTo-Json -Compress")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                           capture_output=True, timeout=25, creationflags=0x08000000)   # no console window
        data = json.loads(r.stdout.decode("utf-8", "replace").strip() or "[]")
    except Exception:
        return []
    if isinstance(data, dict):
        data = [data]
    items = []
    for d in data:
        name, app_id = str(d.get("Name") or "").strip(), str(d.get("AppID") or "").strip()
        if name and app_id:
            items.append({"id": "shell:AppsFolder\\" + app_id, "name": name})
    return items


def _shortcut_launchers_win():
    roots = [Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "Microsoft/Windows/Start Menu/Programs",
             Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs"]
    items = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() in (".lnk", ".url", ".appref-ms"):
                items.append({"id": str(p), "name": p.stem})
    return items


def _launchers_linux():
    items = []
    dirs = [Path("/usr/share/applications"), Path("/var/lib/flatpak/exports/share/applications"),
            Path.home() / ".local/share/applications"]
    if os.environ.get("FLATPAK_ID"):
        # Inside the Flatpak, /usr is the sandbox's; the computer's apps are under /run/host
        dirs[0] = Path("/run/host/usr/share/applications")
    for d in dirs:
        if not d.exists():
            continue
        for f in d.glob("*.desktop"):
            try:
                text = f.read_text(errors="ignore")
            except Exception:
                continue
            if "NoDisplay=true" in text or "Type=Application" not in text:
                continue
            name = next((l.split("=", 1)[1] for l in text.splitlines() if l.startswith("Name=")), f.stem)
            items.append({"id": str(f), "name": name})
    return items


def _launchers_mac():
    items = []
    for d in (Path("/Applications"), Path("/System/Applications"), Path.home() / "Applications"):
        if d.exists():
            for app in d.glob("*.app"):
                items.append({"id": str(app), "name": app.stem})
    return items


def is_launcher(item_id):
    return any(it["id"] == item_id for it in list_launchers())


def launch(item_id):
    """Only ids from list_launchers() are accepted, so the phone can't run arbitrary paths."""
    if not any(it["id"] == item_id for it in list_launchers()):
        return False
    try:
        if WINDOWS:
            if item_id.startswith("shell:AppsFolder\\"):
                subprocess.Popen(["explorer.exe", item_id], creationflags=0x08000000)
            else:
                os.startfile(item_id)
        elif MAC:
            subprocess.Popen(["open", item_id])
        else:
            if os.environ.get("FLATPAK_ID"):
                # gtk-launch runs on the computer (see packaging/flatpak/host-command.sh)
                subprocess.Popen(["gtk-launch", Path(item_id).stem])
            elif _which("gio"):
                subprocess.Popen(["gio", "launch", item_id])
            else:
                subprocess.Popen(["gtk-launch", Path(item_id).stem])
        return True
    except Exception:
        return False

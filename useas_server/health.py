"""Setup check: is everything each feature needs installed? (the "Setup" tab)

run_checks() returns a list of Check rows: what it is, which feature needs it, whether it's
there, and — when the app can do something about it — a fix (install a package, add the
firewall rule, open the download page, install a bundled driver).
"""

import importlib
import os
import subprocess
import sys
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

OK, WARN, MISSING, INFO = "ok", "warn", "missing", "info"

FROZEN = getattr(sys, "frozen", False)


@dataclass
class Check:
    title: str
    feature: str                   # which feature needs it (shown under the title)
    status: str                    # OK / WARN / MISSING / INFO
    detail: str = ""
    fix_label: str = ""
    fix: Optional[Callable[[Callable[[str], None]], None]] = field(default=None, repr=False)
    optional: bool = False         # "nice to have": missing shows as a warning, not an error


# --- helpers -------------------------------------------------------------------------------

def _has_module(name):
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _app_dir():
    if FROZEN:
        return Path(sys.executable).parent
    return Path(os.path.abspath(sys.argv[0])).parent


def _find_file(*relatives):
    for base in (_app_dir(), Path(getattr(sys, "_MEIPASS", _app_dir()))):
        for rel in relatives:
            p = base / rel
            if p.exists():
                return p
    return None


def pip_fix(*packages):
    """A fix that installs packages into the Python running the server (source runs only;
    the built app already bundles everything it can)."""
    if FROZEN:
        return None

    def run(log):
        log(f"📦 Installing {' '.join(packages)}…")
        try:
            r = subprocess.run([sys.executable, "-m", "pip", "install", *packages],
                               capture_output=True, text=True, timeout=900)
        except Exception as e:
            log(f"❌ pip failed: {e}")
            return
        if r.returncode == 0:
            log(f"✅ Installed {' '.join(packages)} — restart the server to use it")
        else:
            tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
            log("❌ pip: " + " | ".join(tail))
    return run


def url_fix(url, what):
    def run(log):
        log(f"🌐 Opening the {what} download page…")
        webbrowser.open(url)
    return run


def _pip_check(module, title, feature, packages, detail_ok="Installed", optional=False):
    if _has_module(module):
        return Check(title, feature, OK, detail_ok)
    return Check(title, feature, WARN if optional else MISSING,
                 f"Not installed (pip install {' '.join(packages)})" +
                 ("" if not FROZEN else " — update the app"),
                 "Install", pip_fix(*packages), optional=optional)


# --- checks shared by every OS -------------------------------------------------------------

def _common_checks(server):
    rows = []
    # Network
    try:
        from .core import DiscoveryServer
        found = DiscoveryServer(port=0).get_active_ips()
        ips = [f"{d['ip']} ({d.get('label') or d.get('iface', '')})" for d in found]
    except Exception:
        ips = []
    if not ips and server is not None:
        try:
            ips = [server.get_local_ip()]
        except Exception:
            ips = []
    ips = [ip for ip in ips if ip and not ip.startswith("127.")]
    if ips:
        rows.append(Check("Network", "Phones find this PC on Wi-Fi", OK,
                          "Reachable at " + ", ".join(ips[:4]) + " — the phone must be on the same network"))
    else:
        rows.append(Check("Network", "Phones find this PC on Wi-Fi", MISSING,
                          "No network address found. Connect this PC to Wi-Fi or Ethernet."))

    # Server running
    if server is None:
        rows.append(Check("Server", "Everything", WARN, "Not running — press START on the Dashboard"))
    else:
        rows.append(Check("Server", "Everything", OK, f"Running on port {server.port} "
                                                      f"(UDP {server.port + 1} for the mouse)"))

    # Codecs
    from . import media
    if media.available():
        rows.append(Check("Video & audio codecs (PyAV)", "Smooth second screen (H.264), compressed sound (Opus)",
                          OK, "Installed. The graphics card's video encoder is used when there is one "
                              "(the log names it when the second screen starts)."))
    else:
        rows.append(Check("Video & audio codecs (PyAV)", "Smooth second screen (H.264), compressed sound (Opus)",
                          WARN, "Not installed: second screen uses JPEG, sound is uncompressed",
                          "Install", pip_fix("av"), optional=True))

    rows.append(_pip_check("qrcode", "QR codes", "Pair a phone by scanning a QR code", ["qrcode"], optional=True))
    rows.append(_pip_check("tkinterdnd2", "Drag and drop", "Drop files on this window to send them",
                           ["tkinterdnd2"], optional=True))
    mp = _pip_check("mediapipe", "Background effects (MediaPipe)", "Blur or replace the webcam background",
                    ["mediapipe"], optional=True)
    if mp.status == OK:
        from .common import resource_path
        if not os.path.exists(resource_path("selfie_segmenter.tflite")):
            mp = Check(mp.title, mp.feature, WARN, "selfie_segmenter.tflite is missing next to the app",
                       optional=True)
    rows.append(mp)

    from . import nowplaying
    backend, reason = nowplaying.make_backend()
    if backend is not None:
        rows.append(Check("Song info", "Title, artist and cover of what's playing, on the phone", OK, "Available"))
    else:
        packages = reason.replace("pip install ", "").split() if reason.startswith("pip install") else []
        rows.append(Check("Song info", "Title, artist and cover of what's playing, on the phone", WARN,
                          f"Not available ({reason}). Play/pause/next still work.",
                          "Install" if packages else "", pip_fix(*packages) if packages else None, optional=True))
    return rows


# --- Windows ---------------------------------------------------------------------------------

def _reg_exists(path, root="HKLM"):
    import winreg
    hive = winreg.HKEY_LOCAL_MACHINE if root == "HKLM" else winreg.HKEY_CURRENT_USER
    for view in (0, getattr(winreg, "KEY_WOW64_64KEY", 0)):
        try:
            winreg.CloseKey(winreg.OpenKey(hive, path, 0, winreg.KEY_READ | view))
            return True
        except OSError:
            continue
    return False


def _windows_checks(platform, server):
    rows = []
    # Firewall
    suffix = "" if FROZEN else "_Python"
    names = [f"UseAsServer_TCP{suffix}", f"UseAsServer_UDP{suffix}"]

    def rule_exists(name):
        try:
            r = subprocess.run(["netsh", "advfirewall", "firewall", "show", "rule", f"name={name}"],
                               capture_output=True, text=True, timeout=8)
            return r.returncode == 0
        except Exception:
            return None

    states = [rule_exists(n) for n in names]

    def add_rules(log):
        exe = sys.executable
        chain = " & ".join(
            f'netsh advfirewall firewall add rule name={n} dir=in action=allow program="{exe}" '
            f'enable=yes profile=any protocol={proto}'
            for n, proto, have in zip(names, ("TCP", "UDP"), states) if not have)
        log("🔐 Adding the firewall rules — allow it in the Windows prompt…")
        code = platform._run_elevated(os.environ.get("ComSpec", "cmd.exe"), f'/c "{chain}"', _app_dir())
        log("✅ Firewall rules added" if code is not None else "❌ Permission refused")

    if all(states):
        rows.append(Check("Firewall", "Phones can connect", OK, "Allowed through Windows Firewall"))
    elif any(s is None for s in states):
        rows.append(Check("Firewall", "Phones can connect", INFO, "Couldn't read the firewall rules"))
    else:
        rows.append(Check("Firewall", "Phones can connect", MISSING,
                          "No rule for this program: Windows may block phones (especially on Public networks)",
                          "Allow", add_rules))

    # Gamepad
    vigem = _reg_exists(r"SYSTEM\CurrentControlSet\Services\ViGEmBus")
    if vigem and _has_module("vgamepad"):
        rows.append(Check("Gamepad driver (ViGEmBus)", "Phone as an Xbox controller", OK, "Installed"))
    elif not vigem:
        bundled = _find_file("ViGEmBus_Setup_x64.exe", "Final_Build/drivers/ViGEmBus_Setup_x64.exe")

        def install_vigem(log, path=bundled):
            if path is not None:
                log("⚙️ Starting the ViGEmBus installer…")
                platform._run_elevated(path, "", path.parent, timeout=600)
                log("ℹ️ Restart the server when the installer has finished")
            else:
                url_fix("https://github.com/nefarius/ViGEmBus/releases/latest", "ViGEmBus")(log)

        rows.append(Check("Gamepad driver (ViGEmBus)", "Phone as an Xbox controller", MISSING,
                          "Not installed: games won't see the controller (keyboard keys are used instead)",
                          "Install", install_vigem))
    else:
        rows.append(_pip_check("vgamepad", "Gamepad driver (ViGEmBus)", "Phone as an Xbox controller",
                               ["vgamepad"]))

    # Audio: loopback (PC sound -> phone) and VB-CABLE (phone mic -> PC)
    rows.append(_pip_check("pyaudiowpatch", "PC sound capture", "Hear the PC's sound on the phone",
                           ["pyaudiowpatch"]))
    cable = None
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        try:
            import pyaudio
        except ImportError:
            pyaudio = None
    if pyaudio is not None:
        try:
            pa = pyaudio.PyAudio()
            try:
                cable = any("CABLE Input" in pa.get_device_info_by_index(i).get("name", "")
                            for i in range(pa.get_device_count()))
            finally:
                pa.terminate()
        except Exception:
            cable = None
    if cable:
        rows.append(Check("Virtual microphone (VB-CABLE)", "Phone as a microphone for Zoom, Discord, OBS…", OK,
                          "Installed — pick \"CABLE Output\" as the microphone in your apps"))
    else:
        setup = _find_file("Final_Build/drivers/VBCABLE/VBCABLE_Setup_x64.exe", "VBCABLE_Setup_x64.exe")

        def install_cable(log, path=setup):
            if path is not None:
                log("⚙️ Starting the VB-CABLE installer — click \"Install Driver\", then restart the PC")
                platform._run_elevated(path, "", path.parent, timeout=600)
            else:
                url_fix("https://vb-audio.com/Cable/", "VB-CABLE")(log)

        rows.append(Check("Virtual microphone (VB-CABLE)", "Phone as a microphone for Zoom, Discord, OBS…",
                          MISSING if cable is False else INFO,
                          "Not installed: the phone mic plays on the speakers instead" if cable is False
                          else "Couldn't check the audio devices",
                          "Install", install_cable))

    # Virtual camera
    obs_dll = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / \
        "obs-studio/data/obs-plugins/win-dshow/obs-virtualcam-module64.dll"
    obs = obs_dll.exists() or _reg_exists(r"SOFTWARE\Classes\CLSID\{A3FCE0F5-3493-419F-958A-ABA1250EC20B}")
    cam_ok = obs and _has_module("pyvirtualcam")
    if cam_ok:
        rows.append(Check("Virtual camera (OBS)", "Phone as a webcam", OK,
                          "OBS is installed — the camera is called \"OBS Virtual Camera\" in your apps"))
    elif not _has_module("pyvirtualcam"):
        rows.append(_pip_check("pyvirtualcam", "Virtual camera (OBS)", "Phone as a webcam", ["pyvirtualcam"]))
    else:
        rows.append(Check("Virtual camera (OBS)", "Phone as a webcam", MISSING,
                          "Install OBS Studio, then click \"Start Virtual Camera\" in it once",
                          "Download", url_fix("https://obsproject.com/download", "OBS Studio")))

    # Virtual monitor
    exe, _ = platform._devinst()
    if platform._idd_installed():
        rows.append(Check("Virtual monitor driver", "Phone as an extra screen", OK,
                          "Installed — pick a monitor this PC doesn't have in the app"))
    elif exe is not None:
        def install_idd(log):
            log("⚙️ Installing the virtual monitor driver — allow it in the Windows prompt…")
            ok = platform._devinst_run([["install", "usbmmidd.inf", "usbmmidd"]], elevated=True)
            log("✅ Driver installed" if ok and platform._idd_installed() else "❌ Not installed")
        rows.append(Check("Virtual monitor driver", "Phone as an extra screen", WARN,
                          "Not installed yet (it installs itself the first time it's needed)",
                          "Install", install_idd, optional=True))
    else:
        rows.append(Check("Virtual monitor driver", "Phone as an extra screen", WARN,
                          "Driver files not found — reinstall with \"Virtual monitor driver\" ticked",
                          optional=True))

    rows.append(_pip_check("win32clipboard", "pywin32", "Images on the clipboard, virtual monitor resolution",
                           ["pywin32"]))
    rows.append(_pip_check("pycaw", "pycaw", "PC volume slider on the phone", ["pycaw"], optional=True))

    admin = platform.is_admin()
    rows.append(Check("Admin rights", "Mouse/keyboard in apps running as administrator (some games, Task Manager)",
                      OK if admin else INFO,
                      "Running as administrator" if admin else
                      "Not admin: input into admin windows is blocked. Settings › Request admin rights"))
    return rows


# --- Linux / macOS -------------------------------------------------------------------------

def _which(cmd):
    from shutil import which
    return which(cmd) is not None


def _linux_checks(platform, server):
    rows = []
    rows.append(Check("xdotool", "Mouse and keyboard", OK if _which("xdotool") else MISSING,
                      "Installed" if _which("xdotool") else "sudo apt install xdotool"))
    uinput = os.access("/dev/uinput", os.W_OK)
    rows.append(Check("Virtual gamepad (uinput)", "Phone as a game controller", OK if uinput else WARN,
                      "Available" if uinput else
                      "No access to /dev/uinput: sudo usermod -aG input $USER (log out and in); keyboard keys are used meanwhile",
                      optional=True))
    rows.append(Check("PulseAudio / PipeWire tools", "PC sound to the phone, phone as a microphone",
                      OK if _which("pactl") else MISSING,
                      "Installed" if _which("pactl") else "sudo apt install pulseaudio-utils"))
    cam = os.path.exists("/dev/video10")
    rows.append(Check("Virtual camera (v4l2loopback)", "Phone as a webcam", OK if cam else WARN,
                      "/dev/video10 ready" if cam else
                      "sudo apt install v4l2loopback-dkms && sudo modprobe v4l2loopback devices=1 video_nr=10 exclusive_caps=1",
                      optional=True))
    clip = _which("xclip") or _which("wl-copy")
    rows.append(Check("Clipboard tools", "Images and copied files on the clipboard", OK if clip else WARN,
                      "Installed" if clip else "sudo apt install xclip (or wl-clipboard on Wayland)", optional=True))
    rows.append(Check("wmctrl", "Switch between windows from the phone", OK if _which("wmctrl") else WARN,
                      "Installed" if _which("wmctrl") else "sudo apt install wmctrl", optional=True))
    rows.append(Check("xrandr", "Phone as an extra screen", OK if _which("xrandr") else WARN,
                      "Installed" if _which("xrandr") else "sudo apt install x11-xserver-utils", optional=True))
    return rows


def _mac_checks(platform, server):
    rows = [_pip_check("Quartz", "Quartz (pyobjc)", "Mouse control", ["pyobjc-framework-Quartz"]),
            _pip_check("pynput", "pynput", "Keyboard control", ["pynput"])]
    blackhole = False
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        try:
            blackhole = any("BlackHole" in pa.get_device_info_by_index(i).get("name", "")
                            for i in range(pa.get_device_count()))
        finally:
            pa.terminate()
    except Exception:
        pass
    rows.append(Check("BlackHole", "PC sound to the phone, phone as a microphone", OK if blackhole else MISSING,
                      "Installed" if blackhole else "brew install blackhole-2ch",
                      "Download", url_fix("https://existential.audio/blackhole/", "BlackHole")))
    rows.append(Check("Permissions", "Mouse, keyboard and second screen", INFO,
                      "Allow the app under System Settings › Privacy & Security › Accessibility and Screen Recording"))
    return rows


def run_checks(platform, server=None):
    """All checks for this OS. Slow-ish (a second or two): call it off the GUI thread."""
    rows = []
    try:
        rows += _common_checks(server)
    except Exception as e:
        rows.append(Check("Basics", "", INFO, f"Check failed: {e}"))
    try:
        if platform.name == "windows":
            rows += _windows_checks(platform, server)
        elif platform.name == "linux":
            rows += _linux_checks(platform, server)
        elif platform.name == "mac":
            rows += _mac_checks(platform, server)
    except Exception as e:
        rows.append(Check(f"{platform.display_name} checks", "", INFO, f"Check failed: {e}"))
    order = {MISSING: 0, WARN: 1, INFO: 2, OK: 3}
    return sorted(rows, key=lambda c: order.get(c.status, 9))

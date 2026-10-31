"""Shared constants and small helpers."""

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

APP_VERSION = "2.0"
GITHUB_REPO = "manjeetdeswal/Use-As-Server"
GITHUB_URL = f"https://github.com/{GITHUB_REPO}"

SAVE_DIR = Path.home() / "Downloads" / "UseAs_Received"


def resource_path(relative_path):
    """Path to a bundled resource, for dev runs and for PyInstaller (_MEIPASS)."""
    base = getattr(sys, "_MEIPASS", None)
    if base is None:
        # dev run: resources sit next to the launcher script (UseAsServer.py etc.)
        base = os.path.dirname(os.path.abspath(sys.argv[0] or __file__))
    return os.path.join(base, relative_path)


_INVALID_NAME_CHARS = re.compile(r'[\x00-\x1f<>:"/\\|?*]')
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL",
                   *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_filename(raw, fallback=None):
    """Reduce a client-supplied name to a bare, safe file name (no directories, drive
    letters, reserved characters or device names), so it can never leave SAVE_DIR."""
    name = str(raw or "").replace("\\", "/").split("/")[-1]
    name = _INVALID_NAME_CHARS.sub("_", name).strip().strip(".").strip()
    if name.split(".")[0].upper() in _RESERVED_NAMES:
        name = "_" + name
    name = name[:150]
    return name or (fallback or f"file_{int(time.time())}.dat")


def unique_path(directory: Path, filename: str) -> Path:
    """directory/filename, or 'name (1).ext', 'name (2).ext' … if it already exists."""
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, ext = os.path.splitext(filename)
    i = 1
    while True:
        candidate = directory / f"{stem} ({i}){ext}"
        if not candidate.exists():
            return candidate
        i += 1


DEFAULT_PREFS = {
    "autostart_pc": False,
    "autostart_server": True,
    "run_as_admin": False,
    "gaming_mode": True,
    "port": 8080,
    "password": "useas",
    "password_enabled": False,
    "clipboard_autosync": False,
    "virtual_display": True,
    "allow_file_browsing": True,
    "allow_file_changes": True,
    "allow_deck_commands": True,   # the phone's custom buttons may run programs and commands
    "call_popups": True,           # incoming phone calls pop up on the PC    # the phone may rename files and move them to the Recycle Bin
    "browse_folders": [],          # folders the phone may browse; empty = the whole PC
    "allow_power_control": True,
    "notification_popups": True,
}


def load_prefs(platform):
    """Defaults + saved settings. Also reads (and converts) files written by the older
    per-OS servers, so nobody loses their settings."""
    prefs = dict(DEFAULT_PREFS)
    prefs["ui_scale"] = platform.default_ui_scale
    candidates = [platform.settings_file]
    if platform.name == "windows":
        # v1.64 and older kept it in the received-files folder
        candidates.append(Path.home() / "Downloads" / "UseAs_Received" / "server_settings.json")
    for path in candidates:
        try:
            if path.exists():
                with open(path, "r") as f:
                    saved = json.load(f)
                if isinstance(saved, dict):
                    for old, new in platform.legacy_pref_keys.items():
                        if old in saved and new not in saved:
                            saved[new] = saved.pop(old)
                    if "scale" in saved and "ui_scale" not in saved:
                        saved["ui_scale"] = saved.pop("scale")
                    prefs.update(saved)
                    if path != platform.settings_file:
                        save_prefs(platform, prefs)   # migrate to the new location
                    break
        except Exception as e:
            print(f"Settings load skipped ({path}): {e}")
    return prefs


def save_prefs(platform, prefs):
    try:
        platform.settings_dir.mkdir(parents=True, exist_ok=True)
        tmp = platform.settings_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(prefs, f, indent=4)
        shutil.move(str(tmp), str(platform.settings_file))   # never leave a half-written file
        return True
    except Exception as e:
        print(f"Settings save failed: {e}")
        return False

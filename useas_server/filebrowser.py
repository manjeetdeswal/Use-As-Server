"""Browse this PC's files from the phone (read-only listing; downloads use the normal
file transfer). Hidden/system entries are left out.

`allowed` (a list of folders) limits browsing to those folders and everything inside them;
an empty list means the whole PC. It's set in the PC app's Settings.
"""

import os
import stat
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

MAX_ENTRIES = 1500
RECENT = "::recent"         # the "Recent files" pseudo-folder
RECENT_MAX = 60


def _norm(path):
    p = os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))
    return os.path.normcase(p)


def is_allowed(path, allowed):
    """True when `path` is inside one of the allowed folders (or no limit is set)."""
    if not allowed:
        return True
    try:
        target = _norm(path)
    except (OSError, ValueError):
        return False
    for folder in allowed:
        try:
            root = _norm(folder)
        except (OSError, ValueError):
            continue
        if target == root or target.startswith(root.rstrip("\\/") + os.sep):
            return True
    return False


def containing_root(path, allowed):
    """The shared folder that holds `path` (the deepest one), or "" when there's no limit."""
    if not allowed:
        return ""
    best = ""
    try:
        target = _norm(path)
    except (OSError, ValueError):
        return ""
    for folder in allowed:
        try:
            root = _norm(folder)
        except (OSError, ValueError):
            continue
        if (target == root or target.startswith(root.rstrip("\\/") + os.sep)) and len(root) > len(_norm(best) if best else ""):
            best = str(folder)
    return best


def roots(allowed=None):
    """Starting points: the usual user folders, then every drive / mount. With a limit:
    just the allowed folders. "Recent files" comes second either way."""
    recent = {"name": "Recent files", "path": RECENT, "kind": "recent"}
    if allowed:
        out = [{"name": os.path.basename(str(a).rstrip("\\/")) or str(a), "path": str(a), "kind": "folder"}
               for a in allowed if os.path.isdir(a)]
        return out[:1] + [recent] + out[1:] if out else [recent]
    home = Path.home()
    out = []
    for name in ("Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos"):
        p = home / name
        if p.is_dir():
            out.append({"name": name, "path": str(p), "kind": "folder"})
    out.insert(0, {"name": "Home", "path": str(home), "kind": "home"})
    out.insert(1, recent)
    try:
        import psutil
        seen = set()
        for part in psutil.disk_partitions(all=False):
            mp = part.mountpoint
            if mp in seen or (sys.platform != "win32" and not (mp == "/" or mp.startswith(("/media", "/mnt", "/Volumes", "/run/media")))):
                continue
            if sys.platform == "win32" and "cdrom" in (part.opts or "").lower():
                continue
            seen.add(mp)
            label = mp.rstrip("\\/") or "/"
            out.append({"name": label if sys.platform == "win32" else ("Computer" if mp == "/" else os.path.basename(mp)),
                        "path": mp, "kind": "drive"})
    except Exception:
        pass
    return out


def _hidden(entry, st):
    if entry.name.startswith("."):
        return True
    if sys.platform == "win32":
        attrs = getattr(st, "st_file_attributes", 0)
        return bool(attrs & (stat.FILE_ATTRIBUTE_HIDDEN | stat.FILE_ATTRIBUTE_SYSTEM))
    return False


def list_dir(path, allowed=None):
    """{"path", "parent", "entries": [{"name", "path", "dir", "size", "mtime"}], "truncated"}.
    Raises OSError (e.g. permission denied) for the caller to report."""
    if path == RECENT:
        return recent_files(allowed)
    p = Path(path).expanduser()
    if not is_allowed(p, allowed):
        raise PermissionError("This folder isn't shared with the phone")
    if not p.is_dir():
        raise NotADirectoryError(str(p))
    entries = []
    truncated = False
    with os.scandir(p) as it:
        for entry in it:
            try:
                st = entry.stat(follow_symlinks=True)
            except OSError:
                continue
            if _hidden(entry, st):
                continue
            is_dir = stat.S_ISDIR(st.st_mode)
            entries.append({"name": entry.name, "path": entry.path, "dir": is_dir,
                            "size": 0 if is_dir else st.st_size, "mtime": int(st.st_mtime)})
            if len(entries) >= MAX_ENTRIES:
                truncated = True
                break
    entries.sort(key=lambda e: (not e["dir"], e["name"].lower()))
    parent = str(p.parent) if p.parent != p else ""
    if parent and not is_allowed(parent, allowed):
        parent = ""                     # the top of a shared folder: back goes to the start
    # "base": where the phone's path bar starts (the shared folder, when browsing is limited)
    return {"path": str(p), "parent": parent, "entries": entries, "truncated": truncated,
            "base": containing_root(p, allowed)}


# ------------------------------------------------------------------------------------------
# Actions on files from the phone: open / show on the PC, rename, move to the Recycle Bin
# ------------------------------------------------------------------------------------------
def _is_top(path, allowed):
    """Drives, "/" and the shared folders themselves can't be renamed or deleted."""
    p = os.path.abspath(path)
    if os.path.dirname(p.rstrip("\\/")) == p.rstrip("\\/") or p in ("/",) or len(p.rstrip("\\/")) <= 3:
        return True
    home = os.path.normcase(os.path.abspath(str(Path.home())))
    if os.path.normcase(p) == home:
        return True
    return any(os.path.normcase(_norm(a)) == os.path.normcase(_norm(p)) for a in (allowed or []))


def open_on_pc(path):
    """Open a file (its usual app) or a folder (the file manager) on the PC."""
    import subprocess
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def reveal_on_pc(path):
    """Show the file selected in the PC's file manager."""
    import subprocess
    if sys.platform == "win32":
        subprocess.Popen(f'explorer /select,"{os.path.normpath(path)}"')
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
    else:
        subprocess.Popen(["xdg-open", os.path.dirname(path) or "/"])


_BAD_NAME = set('<>:"/\\|?*') | {chr(c) for c in range(32)}


def rename(path, new_name, allowed=None):
    """Rename inside the same folder. Returns the new path. Raises ValueError with a reason."""
    new_name = (new_name or "").strip().rstrip(".")
    if not new_name or new_name in (".", "..") or any(ch in _BAD_NAME for ch in new_name):
        raise ValueError("That name can't be used")
    if _is_top(path, allowed):
        raise ValueError("This folder can't be renamed from the phone")
    target = os.path.join(os.path.dirname(os.path.abspath(path)), new_name)
    if os.path.exists(target) and os.path.normcase(target) != os.path.normcase(os.path.abspath(path)):
        raise ValueError("Something with that name is already there")
    os.rename(path, target)
    return target


def move_to_trash(paths, allowed=None):
    """Recycle Bin / Trash only — never a permanent delete. Returns how many were moved."""
    paths = [os.path.abspath(p) for p in paths if p and os.path.exists(p)]
    paths = [p for p in paths if not _is_top(p, allowed)]
    if not paths:
        return 0
    if sys.platform == "win32":
        return _trash_windows(paths)
    import subprocess
    moved = 0
    for p in paths:
        try:
            if sys.platform == "darwin":
                esc = p.replace("\\", "\\\\").replace('"', '\\"')
                r = subprocess.run(["osascript", "-e", f'tell application "Finder" to delete POSIX file "{esc}"'],
                                   capture_output=True, timeout=20)
            else:
                r = subprocess.run(["gio", "trash", p], capture_output=True, timeout=20)
            moved += r.returncode == 0
        except Exception:
            continue
    return moved


def _trash_windows(paths):
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", ctypes.c_void_p),
                    ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

    FO_DELETE = 3
    FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = 0x4, 0x10, 0x40, 0x400
    op = SHFILEOPSTRUCTW()
    op.wFunc = FO_DELETE
    # A double-NUL-terminated list; a raw buffer, as ctypes strings can't hold embedded NULs.
    names = "\0".join(paths) + "\0\0"
    buf = (ctypes.c_wchar * len(names))()
    for i, ch in enumerate(names):
        buf[i] = ch
    op.pFrom = ctypes.addressof(buf)
    op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if result != 0 or op.fAnyOperationsAborted:
        return sum(not os.path.exists(p) for p in paths)
    return len(paths)


# ------------------------------------------------------------------------------------------
# Recent files
# ------------------------------------------------------------------------------------------
def recent_files(allowed=None):
    paths = []
    try:
        if sys.platform == "win32":
            paths = _recent_windows()
        elif sys.platform != "darwin":
            paths = _recent_linux()
    except Exception:
        paths = []
    # Topped up with the newest files in the usual (or shared) folders.
    paths = list(paths) + _recent_by_date(allowed)
    entries, seen = [], set()
    for path in paths:
        key = os.path.normcase(path)
        if key in seen or not is_allowed(path, allowed):
            continue
        seen.add(key)
        try:
            st = os.stat(path)
        except OSError:
            continue
        if not stat.S_ISREG(st.st_mode):
            continue
        entries.append({"name": os.path.basename(path), "path": path, "dir": False,
                        "size": st.st_size, "mtime": int(st.st_mtime)})
        if len(entries) >= RECENT_MAX:
            break
    return {"path": RECENT, "parent": "", "entries": entries, "truncated": False, "title": "Recent files"}


def _recent_windows():
    """Targets of the shortcuts in %APPDATA%\\Microsoft\\Windows\\Recent, newest first."""
    folder = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Recent"
    if not folder.is_dir():
        return []
    links = sorted(folder.glob("*.lnk"), key=lambda p: p.stat().st_mtime, reverse=True)[:200]
    try:
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
    except Exception:
        return []
    out = []
    try:
        shell = win32com.client.Dispatch("WScript.Shell")
        for link in links:
            try:
                target = shell.CreateShortcut(str(link)).TargetPath
            except Exception:
                continue
            if target and os.path.isfile(target):
                out.append(target)
    finally:
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass
    return out


def _recent_linux():
    """~/.local/share/recently-used.xbel (GTK's list), newest first."""
    import xml.etree.ElementTree as ET
    xbel = Path.home() / ".local" / "share" / "recently-used.xbel"
    if not xbel.is_file():
        return []
    items = []
    for bm in ET.parse(xbel).getroot().iter("bookmark"):
        href = bm.get("href", "")
        if not href.startswith("file://"):
            continue
        items.append((bm.get("modified") or bm.get("visited") or "", unquote(urlparse(href).path)))
    items.sort(reverse=True)
    return [p for _, p in items]


def _recent_by_date(allowed=None):
    """The newest files in the usual folders (or the shared ones), one level deep."""
    home = Path.home()
    found = []
    cutoff = time.time() - 60 * 86400
    bases = [Path(a) for a in allowed] if allowed else \
        [home / n for n in ("Desktop", "Documents", "Downloads", "Pictures", "Videos", "Music")]
    for base in bases:
        if not base.is_dir():
            continue
        try:
            subdirs = [d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")][:40]
        except OSError:
            subdirs = []
        for depth_dir in [base] + subdirs:
            try:
                with os.scandir(depth_dir) as it:
                    for e in it:
                        if e.name.startswith(".") or not e.is_file():
                            continue
                        m = e.stat().st_mtime
                        if m >= cutoff:
                            found.append((m, e.path))
            except OSError:
                continue
    found.sort(reverse=True)
    return [p for _, p in found[:RECENT_MAX * 2]]

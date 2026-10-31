"""Small thumbnails / icons for the phone's file browser, app list and window list.

Windows asks the shell (IShellItemImageFactory), which gives real thumbnails for photos,
videos, PDFs and documents and the proper icon for everything else (apps, shortcuts,
folders). Elsewhere, pictures are thumbnailed with Pillow, apps use their .icns (macOS)
or icon-theme PNG (Linux), and other files fall back to the phone's own icons.

Every call returns (mime, bytes) or None. Results are cached in memory.
"""
import io
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path

WINDOWS = sys.platform == "win32"
MAC = sys.platform == "darwin"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".ico", ".heic"}
MAX_IMAGE_BYTES = 60 * 1024 * 1024          # don't decode huge files just for a thumbnail
_CACHE_MAX = 600

_cache = OrderedDict()
_cache_lock = threading.Lock()


def thread_init():
    """Initializer for the worker threads: the Windows shell API needs COM on each thread."""
    if WINDOWS:
        try:
            import ctypes
            ctypes.windll.ole32.CoInitializeEx(None, 0x2)     # COINIT_APARTMENTTHREADED
        except Exception:
            pass


def _cached(key, fn):
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    try:
        value = fn()
    except Exception:
        value = None
    with _cache_lock:
        _cache[key] = value
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return value


# ----------------------------------------------------------------------------------------
# public entry points
# ----------------------------------------------------------------------------------------
def for_file(path, size=96):
    """Thumbnail of a file or folder on this PC."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = ("file", path, int(st.st_mtime), st.st_size, size)

    def make():
        if WINDOWS:
            img = _shell_image(path, size)
            if img is not None:
                return _encode(img, size)
        if Path(path).suffix.lower() in IMAGE_EXTS and st.st_size <= MAX_IMAGE_BYTES:
            return _encode(_pil_open(path), size)
        return None
    return _cached(key, make)


def for_app(path, size=96):
    """Icon of a Start-menu shortcut, .desktop file or .app bundle."""
    key = ("app", path, size)

    def make():
        if WINDOWS:
            img = _shell_image(path, size)
            return _encode(img, size) if img is not None else None
        if MAC:
            return _encode(_mac_app_icon(path), size)
        return _encode(_linux_desktop_icon(path), size)
    return _cached(key, make)


def for_program(source, size=96):
    """Icon of a running program: an .exe / .app path, or ("icon", name) on Linux."""
    if not source:
        return None
    if isinstance(source, tuple):
        key = ("icon", source[1], size)
        return _cached(key, lambda: _encode(_linux_theme_icon(source[1]), size))
    return for_app(source, size)


def window_preview(win_id, width=480):
    """A small live picture of an open window (Windows only; not cached). JPEG or None."""
    if not WINDOWS:
        return None
    try:
        img = _capture_window(int(win_id))
    except Exception:
        img = None
    if img is None:
        return None
    from PIL import Image
    img.thumbnail((width, width), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=72)
    return "image/jpeg", buf.getvalue()


def _capture_window(hwnd):
    import ctypes
    from ctypes import wintypes
    from PIL import Image
    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    # Handles are pointers: declare types so they aren't cut to 32 bits on 64-bit Windows.
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetDC.restype = wintypes.HDC
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                                ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]

    if not user32.IsWindow(hwnd) or user32.IsIconic(hwnd):
        return None                     # minimised windows have nothing to draw
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w <= 0 or h <= 0 or w > 10000 or h > 10000:
        return None

    class BMIH(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long), ("biHeight", ctypes.c_long),
                    ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD),
                    ("biCompression", wintypes.DWORD), ("biSizeImage", wintypes.DWORD),
                    ("biXPelsPerMeter", ctypes.c_long), ("biYPelsPerMeter", ctypes.c_long),
                    ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]

    screen = user32.GetDC(None)
    mem = gdi32.CreateCompatibleDC(screen)
    bmp = gdi32.CreateCompatibleBitmap(screen, w, h)
    try:
        old = gdi32.SelectObject(mem, bmp)
        ok = user32.PrintWindow(hwnd, mem, 2)          # PW_RENDERFULLCONTENT: works for GPU apps too
        gdi32.SelectObject(mem, old)
        if not ok:
            return None
        hdr = BMIH()
        hdr.biSize = ctypes.sizeof(hdr)
        hdr.biWidth, hdr.biHeight = w, -h
        hdr.biPlanes, hdr.biBitCount, hdr.biCompression = 1, 32, 0
        buf = ctypes.create_string_buffer(w * h * 4)
        if gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(hdr), 0) != h:
            return None
        img = Image.frombuffer("RGB", (w, h), buf.raw, "raw", "BGRX", 0, 1)
        if img.getextrema() in (((0, 0), (0, 0), (0, 0)),):
            return None                 # all black: the app refused to draw
        return img.copy()
    finally:
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem)
        user32.ReleaseDC(None, screen)


# ----------------------------------------------------------------------------------------
# encoding
# ----------------------------------------------------------------------------------------
def _encode(img, size):
    """PNG when the picture has transparency (icons), JPEG otherwise (photos)."""
    if img is None:
        return None
    from PIL import Image
    img.thumbnail((size, size), Image.LANCZOS)
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        rgba = img.convert("RGBA")
        lo, _hi = rgba.getchannel("A").getextrema()
        if lo < 250:
            buf = io.BytesIO()
            rgba.save(buf, "PNG", optimize=True)
            return "image/png", buf.getvalue()
        img = rgba
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=78)
    return "image/jpeg", buf.getvalue()


def _pil_open(path):
    from PIL import Image, ImageOps
    img = Image.open(path)
    img.draft("RGB", (256, 256))            # fast JPEG decode at reduced size
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img.load()
    return img


# ----------------------------------------------------------------------------------------
# Windows: IShellItemImageFactory
# ----------------------------------------------------------------------------------------
_shell = None
_shell_lock = threading.Lock()


def _shell_api():
    """Lazily build the ctypes / comtypes bits (only on Windows, only once)."""
    with _shell_lock:
        if _shell is None:
            _build_shell_api()
    return _shell or None


def _build_shell_api():
    global _shell
    try:
        import ctypes
        from ctypes import wintypes
        import comtypes
        from comtypes import GUID, IUnknown, COMMETHOD, HRESULT

        class SIZE(ctypes.Structure):
            _fields_ = [("cx", ctypes.c_long), ("cy", ctypes.c_long)]

        class IShellItemImageFactory(IUnknown):
            _iid_ = GUID("{bcc18b79-ba16-442f-80c4-8a59c30c463b}")
            _methods_ = [
                COMMETHOD([], HRESULT, "GetImage",
                          (["in"], SIZE, "size"),
                          (["in"], ctypes.c_int, "flags"),
                          (["out"], ctypes.POINTER(wintypes.HBITMAP), "phbm")),
            ]

        shell32 = ctypes.windll.shell32
        shell32.SHCreateItemFromParsingName.argtypes = [
            wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.POINTER(IShellItemImageFactory))]
        shell32.SHCreateItemFromParsingName.restype = ctypes.c_long     # plain int: no exception on failure

        class BITMAP(ctypes.Structure):
            _fields_ = [("bmType", ctypes.c_long), ("bmWidth", ctypes.c_long), ("bmHeight", ctypes.c_long),
                        ("bmWidthBytes", ctypes.c_long), ("bmPlanes", wintypes.WORD),
                        ("bmBitsPixel", wintypes.WORD), ("bmBits", ctypes.c_void_p)]

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long), ("biHeight", ctypes.c_long),
                        ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD),
                        ("biCompression", wintypes.DWORD), ("biSizeImage", wintypes.DWORD),
                        ("biXPelsPerMeter", ctypes.c_long), ("biYPelsPerMeter", ctypes.c_long),
                        ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]

        gdi32 = ctypes.windll.gdi32
        gdi32.GetObjectW.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p]
        gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                                    ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
        gdi32.DeleteObject.argtypes = [wintypes.HANDLE]
        user32 = ctypes.windll.user32
        user32.GetDC.restype = wintypes.HDC
        user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]

        _shell = dict(ctypes=ctypes, comtypes=comtypes, SIZE=SIZE, IFactory=IShellItemImageFactory,
                      shell32=shell32, gdi32=gdi32, user32=user32, BITMAP=BITMAP, BMIH=BITMAPINFOHEADER)
    except Exception:
        _shell = False


def _shell_image(path, size):
    api = _shell_api()
    if api is None:
        return None
    ctypes = api["ctypes"]
    from PIL import Image
    # comtypes owns this pointer and releases it when it goes away.
    factory = ctypes.POINTER(api["IFactory"])()
    iid = api["IFactory"]._iid_
    hr = api["shell32"].SHCreateItemFromParsingName(str(path), None, ctypes.byref(iid), ctypes.byref(factory))
    if hr != 0 or not factory:
        return None
    try:
        # flags 0: a real thumbnail when Windows has one, the file's icon otherwise
        hbmp = factory.GetImage(api["SIZE"](size, size), 0)
    except Exception:
        return None
    finally:
        del factory
    if not hbmp:
        return None
    gdi32 = api["gdi32"]
    try:
        bm = api["BITMAP"]()
        if not gdi32.GetObjectW(hbmp, ctypes.sizeof(bm), ctypes.byref(bm)):
            return None
        w, h = bm.bmWidth, abs(bm.bmHeight)
        if w <= 0 or h <= 0 or w > 1024 or h > 1024:
            return None
        hdr = api["BMIH"]()
        hdr.biSize = ctypes.sizeof(hdr)
        hdr.biWidth, hdr.biHeight = w, -h          # negative = top-down rows
        hdr.biPlanes, hdr.biBitCount, hdr.biCompression = 1, 32, 0
        buf = ctypes.create_string_buffer(w * h * 4)
        hdc = api["user32"].GetDC(None)
        try:
            lines = gdi32.GetDIBits(hdc, hbmp, 0, h, buf, ctypes.byref(hdr), 0)
        finally:
            api["user32"].ReleaseDC(None, hdc)
        if lines != h:
            return None
        img = Image.frombuffer("RGBA", (w, h), buf.raw, "raw", "BGRA", 0, 1)
        _alpha_lo, alpha_hi = img.getchannel("A").getextrema()
        if alpha_hi == 0:                       # thumbnails often come with an empty alpha channel
            img = img.convert("RGB")
        else:
            img = _unpremultiply(img)
        return img
    finally:
        gdi32.DeleteObject(hbmp)


def _unpremultiply(img):
    """Shell icons use premultiplied alpha; undo it so edges don't look dark."""
    try:
        if img.getchannel("A").getextrema()[0] == 255:
            return img
        px = img.load()
        w, h = img.size
        for y in range(h):          # ~9k pixels at 96x96: fast enough in plain Python
            for x in range(w):
                pr, pg, pb, pa = px[x, y]
                if 0 < pa < 255:
                    px[x, y] = (min(255, pr * 255 // pa), min(255, pg * 255 // pa),
                                min(255, pb * 255 // pa), pa)
        return img
    except Exception:
        return img


# ----------------------------------------------------------------------------------------
# macOS: .app bundle icons
# ----------------------------------------------------------------------------------------
def _mac_app_icon(app_path):
    import plistlib
    from PIL import Image
    app = Path(app_path)
    if app.suffix != ".app":
        return None
    resources = app / "Contents" / "Resources"
    name = None
    try:
        with open(app / "Contents" / "Info.plist", "rb") as f:
            info = plistlib.load(f)
        name = info.get("CFBundleIconFile") or info.get("CFBundleIconName")
    except Exception:
        pass
    candidates = []
    if name:
        candidates.append(resources / (name if name.endswith(".icns") else name + ".icns"))
    candidates += sorted(resources.glob("*.icns")) if resources.exists() else []
    for c in candidates:
        if c.exists():
            img = Image.open(c)
            img.load()
            return img
    return None


def mac_app_path(app_name):
    for d in ("/Applications", "/System/Applications", str(Path.home() / "Applications"),
              "/System/Applications/Utilities", "/Applications/Utilities"):
        p = Path(d) / f"{app_name}.app"
        if p.exists():
            return str(p)
    return None


# ----------------------------------------------------------------------------------------
# Linux: icon theme lookup
# ----------------------------------------------------------------------------------------
_ICON_DIRS = [
    "/usr/share/icons/hicolor/128x128/apps", "/usr/share/icons/hicolor/96x96/apps",
    "/usr/share/icons/hicolor/64x64/apps", "/usr/share/icons/hicolor/48x48/apps",
    "/usr/share/icons/hicolor/256x256/apps", "/usr/share/pixmaps",
    "/var/lib/flatpak/exports/share/icons/hicolor/128x128/apps",
    str(Path.home() / ".local/share/icons/hicolor/128x128/apps"),
]


def _linux_desktop_icon(desktop_path):
    try:
        text = Path(desktop_path).read_text(errors="ignore")
    except Exception:
        return None
    icon = next((l.split("=", 1)[1].strip() for l in text.splitlines() if l.startswith("Icon=")), "")
    return _linux_theme_icon(icon)


def _linux_theme_icon(name):
    from PIL import Image
    if not name:
        return None
    if os.path.isabs(name):
        if os.path.exists(name) and not name.endswith(".svg"):
            return Image.open(name)
        return None
    for variant in (name, name.lower()):
        for d in _ICON_DIRS:
            for ext in (".png", ".xpm"):
                p = os.path.join(d, variant + ext)
                if os.path.exists(p):
                    try:
                        img = Image.open(p)
                        img.load()
                        return img
                    except Exception:
                        continue
    return None

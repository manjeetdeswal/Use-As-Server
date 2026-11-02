"""Makes every app icon from icon.png (the 2048 px artwork). Run from the repository root
after changing icon.png:   python packaging/icons/make_icons.py

  icon.ico                   Windows: program, taskbar, installer, tray. Has every size Windows
                             asks for; 16–48 px use a simple bold phone that stays readable.
  packaging/icons/icon-mac.png        macOS app icon (Apple's grid: rounded, with margin)
  packaging/icons/window-*.png        window/tray icon on Linux and macOS
  packaging/icons/hicolor/<n>/use-as-server.png   Linux menu icons (.deb, .rpm, Flatpak)
"""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "packaging" / "icons"
BG = (13, 13, 13, 255)
GREEN = (0, 230, 118, 255)
SMALL = (16, 20, 24, 32, 40, 48)      # simple design
LARGE = (64, 96, 128, 256, 512)       # full artwork


def rounded_mask(size, radius):
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius, fill=255)
    return mask


def art(size):
    """The artwork on a rounded square (transparent corners)."""
    src = Image.open(ROOT / "icon.png").convert("RGBA")
    img = src.resize((size, size), Image.LANCZOS)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img, (0, 0), rounded_mask(size, round(size * 0.2237)))
    return out


def simple(size):
    """Bold phone on the same dark rounded square: readable at 16–48 px. Drawn 8x and shrunk."""
    k = 8
    s = size * k
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, s - 1, s - 1), round(s * 0.2237), fill=BG)
    stroke = max(round(s * 0.085), round(1.6 * k))          # ~1.6 px at 16 px
    w, h = s * 0.46, s * 0.74
    x0, y0 = (s - w) / 2, (s - h) / 2
    d.rounded_rectangle((x0, y0, x0 + w, y0 + h), round(w * 0.22), outline=GREEN, width=stroke)
    bar_w = w * 0.34
    by = y0 + h - stroke * 2.1
    d.line((s / 2 - bar_w / 2, by, s / 2 + bar_w / 2, by), fill=GREEN, width=max(stroke * 2 // 3, k))
    return img.resize((size, size), Image.LANCZOS)


def mac_icon():
    """Apple's macOS grid: 1024 canvas, 824 px rounded square in the middle."""
    canvas = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    canvas.paste(art(824), (100, 100), art(824))
    return canvas


def main():
    by_size = {n: simple(n) for n in SMALL} | {n: art(n) for n in LARGE}

    ico_sizes = [n for n in sorted(by_size) if n <= 256]
    biggest = by_size[256]
    biggest.save(ROOT / "icon.ico", format="ICO", sizes=[(n, n) for n in ico_sizes],
                 append_images=[by_size[n] for n in ico_sizes if n != 256])

    mac_icon().save(OUT / "icon-mac.png")
    for n in (128, 64, 32):
        by_size[n].save(OUT / f"window-{n}.png")
    for n in (16, 24, 32, 48, 64, 128, 256, 512):
        d = OUT / "hicolor" / f"{n}x{n}"
        d.mkdir(parents=True, exist_ok=True)
        by_size[n].save(d / "use-as-server.png")
    print("icons written")


if __name__ == "__main__":
    main()

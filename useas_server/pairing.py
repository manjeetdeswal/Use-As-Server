"""QR pairing: the code holds the PC's addresses, port and a secret token.

Scanning it (in the app, or with the phone's camera app via the useas:// link) connects
without typing an address and pairs even when a password is required.
"""

import secrets
import socket
from urllib.parse import quote

PRIORITY = ["USB Tethering", "Mobile Hotspot", "Wi-Fi", "Ethernet / LAN"]


def new_token():
    return secrets.token_urlsafe(12)   # ~96 bits


def ordered_ips(endpoints):
    """Most likely reachable first (USB/hotspot, then Wi-Fi, then wired)."""
    eps = sorted(endpoints or [], key=lambda e: PRIORITY.index(e.get("label", "").rstrip(" ★"))
                 if e.get("label", "").rstrip(" ★") in PRIORITY else 99)
    seen, ips = set(), []
    for e in eps:
        ip = e.get("ip")
        if ip and ip not in seen:
            seen.add(ip)
            ips.append(ip)
    return ips


def qr_payload(ips, port, token, name=None):
    name = name or socket.gethostname()
    return (f"useas://pair?h={','.join(ips)}&p={int(port)}&t={quote(token, safe='')}"
            f"&n={quote(name[:40], safe='')}")


def qr_image(data, box_size=8):
    """PIL image of the QR code, or None if the 'qrcode' package isn't installed."""
    try:
        import qrcode
    except ImportError:
        return None
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=box_size, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img = img.get_image() if hasattr(img, "get_image") else img
    return img.convert("RGB")

"""Use As wire protocol, version 2.

Every text message is a JSON object ``{"type": <str>, "payload": <value>}``.

Version 2 adds a handshake and fixes how payloads are read:

* **Handshake.** The phone sends ``hello`` first (before or right after pairing). The server
  answers with its own ``hello``. Both carry a protocol number and a list of features, so
  each side knows what the other supports instead of guessing, and the app can tell the
  user when the PC server is too old.

* **Payloads are decoded per message type**, never by sniffing (the old server tried
  ``json.loads`` on anything starting with "{" and even swapped single quotes for double
  quotes). Each type below has one expected shape. Version-1 apps double-encode objects as
  JSON strings; that legacy form is still accepted so old apps keep working.

* **Binary frames** start with one type byte (``BIN_*`` below).
"""

import json

PROTOCOL_VERSION = 2
# Oldest phone protocol the server still talks to. 1 = app versions without "hello".
MIN_CLIENT_PROTOCOL = 1

# --- Binary frame types (first byte of every binary WebSocket message) ------------------
BIN_FILE_CHUNK = 0x01      # server -> phone: file data
BIN_FILE_END = 0x02        # server -> phone: end of file
BIN_SCREEN_JPEG = 0x03     # server -> phone: one JPEG screen frame
BIN_SCREEN_H264 = 0x04     # server -> phone: [flags: bit0 keyframe][u16 width][u16 height][Annex-B]
BIN_CLIPBOARD_IMAGE = 0x05 # server -> phone: PNG copied on the PC (phones with "clipboard_image")
BIN_NOW_PLAYING_ART = 0x06 # server -> phone: JPEG cover of what the PC is playing
BIN_PC_AUDIO_PCM = 0x10    # server -> phone: [u8 channels][u32 rate][s16le PCM]
BIN_PC_AUDIO_OPUS = 0x11   # server -> phone: [u8 channels][Opus packet, 48 kHz]
BIN_CAMERA_JPEG = 0x20     # phone -> server: camera JPEG
BIN_MIC_PCM = 0x21         # phone -> server: mic PCM 16-bit mono at the mic_start rate
BIN_MIC_OPUS = 0x22        # phone -> server: mic Opus packet (mono, 48 kHz)
BIN_PHONE_CLIPBOARD_IMAGE = 0x23  # phone -> server: image copied on the phone (PNG/JPEG/WebP)

# Codecs a phone can ask for ("codec" in display_request / audio_control / mic_start).
SCREEN_CODECS = ("h264", "jpeg")
AUDIO_CODECS = ("opus", "pcm", "json")
MIC_CODECS = ("opus", "pcm")

# --- Payload shapes -----------------------------------------------------------------------
OBJECT = "object"   # JSON object (v1 apps send it as a JSON string; both accepted)
TEXT = "text"       # plain string

PAYLOAD_SHAPES = {
    "hello": OBJECT,
    "pair": TEXT,
    "device_info": OBJECT,
    "heartbeat": TEXT,
    "ack": TEXT,
    "mouse_move": OBJECT,
    "mouse_click": OBJECT,
    "mouse_scroll": OBJECT,
    "mouse_zoom": OBJECT,
    "key_press": OBJECT,
    "gamepad_state": OBJECT,
    "display_request": OBJECT,
    "system_control": OBJECT,
    "file_transfer": OBJECT,
    "video_frame": OBJECT,
    "audio_frame": TEXT,        # base64 PCM
    "mic_start": OBJECT,        # {"rate": 16000}; v1 sends "16000"
    "mic_stop": TEXT,
    "audio_control": OBJECT,    # {"action": "start", "rate": 48000}; v1 sends "start|{...}"/"stop"
    "clipboard_text": OBJECT,   # {"text": "..."}; a bare string is accepted too
    "media_subscribe": OBJECT,  # {"on": true} -> "now_playing" messages + 0x06 cover art
    "media_control": OBJECT,    # {"action": "play_pause"|"next"|"previous"|"seek", "position": s}
}


class ProtocolError(ValueError):
    pass


def decode(raw):
    """Parse one text WebSocket message into ``(type, payload)``.

    OBJECT payloads always come back as a dict, TEXT payloads as a str. Raises
    ProtocolError for messages that aren't JSON objects with a string "type".
    """
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as e:
        raise ProtocolError(f"not JSON: {e}") from None
    if not isinstance(data, dict) or not isinstance(data.get("type"), str):
        raise ProtocolError("message must be an object with a string 'type'")

    msg_type = data["type"]
    payload = data.get("payload", "")
    shape = PAYLOAD_SHAPES.get(msg_type)

    if msg_type == "audio_control":
        return msg_type, _decode_audio_control(payload)
    if msg_type == "mic_start":
        return msg_type, _decode_mic_start(payload)
    if msg_type == "clipboard_text":
        return msg_type, _decode_clipboard(payload)

    if shape == OBJECT:
        return msg_type, _as_object(payload)
    if shape == TEXT:
        return msg_type, payload if isinstance(payload, str) else json.dumps(payload)
    return msg_type, payload   # unknown type: pass through untouched


def encode(msg_type, payload=""):
    """Server -> phone. Objects are sent as real JSON objects for v2 phones."""
    return json.dumps({"type": msg_type, "payload": payload})


def encode_for(client_protocol, msg_type, payload):
    """Like encode(), but double-encodes objects for v1 phones, which expect a JSON string."""
    if isinstance(payload, (dict, list)) and client_protocol < 2:
        payload = json.dumps(payload)
    return encode(msg_type, payload)


# --- helpers ------------------------------------------------------------------------------

def _as_object(payload):
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        s = payload.strip()
        if not s:
            return {}
        try:
            value = json.loads(s)
        except ValueError:
            raise ProtocolError("expected a JSON object payload") from None
        if isinstance(value, dict):
            return value
    raise ProtocolError("expected a JSON object payload")


def _decode_clipboard(payload):
    if isinstance(payload, dict):
        return {"text": str(payload.get("text", ""))}
    text = payload if isinstance(payload, str) else str(payload)
    if text.lstrip().startswith("{"):
        try:
            value = json.loads(text)
            if isinstance(value, dict) and "text" in value:
                return {"text": str(value["text"])}
        except ValueError:
            pass   # it's just text that happens to start with "{"
    return {"text": text}


def _decode_mic_start(payload):
    if isinstance(payload, dict):
        obj = payload
    elif isinstance(payload, (int, float)):
        obj = {"rate": payload}
    else:
        s = str(payload).strip()
        if s.isdigit():
            obj = {"rate": int(s)}
        else:
            try:
                obj = _as_object(s)
            except ProtocolError:
                obj = {}
    try:
        rate = int(obj.get("rate", 48000))
    except (TypeError, ValueError):
        rate = 48000
    codec = str(obj.get("codec", "pcm")).lower()
    return {"rate": rate, "codec": codec if codec in MIC_CODECS else "pcm"}


def _decode_audio_control(payload):
    """v2: {"action": "start"|"stop", "rate": 48000}. v1: "start", "start|{json}", "stop"."""
    codec = ""
    if isinstance(payload, dict):
        action = str(payload.get("action", "")).lower()
        rate = payload.get("rate", 48000)
        codec = payload.get("codec", "")
    else:
        s = str(payload).strip()
        if s.startswith("{"):
            obj = _as_object(s)
            action, rate = str(obj.get("action", "")).lower(), obj.get("rate", 48000)
            codec = obj.get("codec", "")
        else:
            action, _, config = s.partition("|")
            action = action.strip().lower()
            rate = 48000
            if config:
                try:
                    cfg = json.loads(config)
                    rate = cfg.get("rate", 48000)
                    codec = cfg.get("codec", "")
                except (ValueError, AttributeError):
                    pass
    try:
        rate = int(rate)
    except (TypeError, ValueError):
        rate = 48000
    if action not in ("start", "stop"):
        raise ProtocolError(f"unknown audio_control action: {action!r}")
    codec = str(codec or "").lower()
    # No codec = an app from before phase 3: base64 JSON frames, as before.
    return {"action": action, "rate": rate, "codec": codec if codec in AUDIO_CODECS else "json"}


def client_hello_info(payload):
    """Normalise a phone's hello payload."""
    try:
        protocol = int(payload.get("protocol", 1))
    except (TypeError, ValueError):
        protocol = 1
    features = payload.get("features") or []
    return {
        "protocol": protocol,
        "app_version": str(payload.get("app_version", "")),
        "device": str(payload.get("device", ""))[:80],
        "features": [str(f) for f in features if isinstance(f, (str, int))][:64],
    }

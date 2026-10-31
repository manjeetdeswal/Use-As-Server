"""Video and audio codecs for the phone streams (protocol v2, phase 3).

Everything here uses PyAV (``pip install av``), which bundles FFmpeg with x264 and libopus.
When PyAV is missing the server keeps working exactly as before: JPEG second screen and
plain PCM audio. Nothing in this module talks to the network.

* :class:`H264Encoder` – low-latency H.264 for the second screen. Tries the GPU encoders
  first (NVENC / Quick Sync / AMF / Media Foundation on Windows, VideoToolbox on macOS,
  NVENC on Linux) and falls back to x264. Output is Annex-B with SPS/PPS in front of
  every keyframe, so the phone can start or recover at any keyframe.
* :class:`OpusEncoder` – PC audio -> phone (any rate / mono or stereo in, 48 kHz Opus out).
* :class:`OpusDecoder` – phone microphone Opus -> 16-bit PCM at the rate the mic output wants.
"""

import sys
import time
from fractions import Fraction

try:
    import av                     # noqa: F401
    import numpy as np
    _AV_ERROR = None
except Exception as _e:           # ImportError, or a broken FFmpeg DLL
    av = None
    np = None
    _AV_ERROR = _e


def available():
    return av is not None


def unavailable_reason():
    return "" if av is not None else f"PyAV not installed ({_AV_ERROR})"


def _has_codec(name, mode):
    try:
        av.codec.Codec(name, mode)
        return True
    except Exception:
        return False


def h264_supported():
    return av is not None and any(_has_codec(n, "w") for n in _encoder_candidates())


def opus_supported():
    return av is not None and _has_codec("libopus", "w") and _has_codec("libopus", "r")


# ------------------------------------------------------------------------------------------
# H.264
# ------------------------------------------------------------------------------------------

def _encoder_candidates():
    if sys.platform == "win32":
        return ["h264_nvenc", "h264_qsv", "h264_amf", "h264_mf", "libx264"]
    if sys.platform == "darwin":
        return ["h264_videotoolbox", "libx264"]
    return ["h264_nvenc", "libx264"]


def _encoder_options(name, fps, bitrate):
    """Per-encoder settings for the lowest latency: no B-frames, no lookahead, no frame
    reordering, keyframes only when asked for (plus one every 10 s as a safety net)."""
    rate = str(int(bitrate))
    buf = str(int(bitrate / max(1, fps) * 2))          # ~2 frames of VBV buffer
    if name == "libx264":
        return {"preset": "ultrafast", "tune": "zerolatency", "profile": "baseline",
                "forced-idr": "1", "maxrate": rate, "bufsize": buf,
                "x264-params": "repeat-headers=1:scenecut=0:sliced-threads=1"}
    if name == "h264_nvenc":
        return {"preset": "p1", "tune": "ull", "zerolatency": "1", "delay": "0", "rc": "cbr",
                "forced-idr": "1", "profile": "main", "maxrate": rate, "bufsize": buf}
    if name == "h264_qsv":
        return {"preset": "veryfast", "forced_idr": "1", "async_depth": "1", "look_ahead": "0",
                "profile": "main", "maxrate": rate, "bufsize": buf}
    if name == "h264_amf":
        return {"usage": "ultralowlatency", "quality": "speed", "rc": "cbr",
                "header_insertion_mode": "idr", "profile": "main", "maxrate": rate, "bufsize": buf}
    if name == "h264_mf":
        return {"rate_control": "cbr", "scenario": "display_remoting", "hw_encoding": "1"}
    if name == "h264_videotoolbox":
        return {"realtime": "1", "prio_speed": "1", "profile": "main", "maxrate": rate}
    return {}


def _pix_fmt(name):
    return "nv12" if name in ("h264_qsv", "h264_amf", "h264_mf", "h264_videotoolbox") else "yuv420p"


def bitrate_for(width, height, fps, quality):
    """Phone's quality slider (roughly 20-70, the old JPEG quality) -> bits per second."""
    q = max(10, min(95, int(quality)))
    bits_per_pixel = 0.02 + q / 100.0 * 0.12
    return int(max(500_000, min(25_000_000, width * height * max(1, fps) * bits_per_pixel)))


def split_nals(data):
    """Annex-B byte stream -> list of NAL units (without start codes)."""
    nals = []
    n = len(data)
    i = 0
    start = -1
    while i + 2 < n:
        if data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 1:
            if start >= 0:
                end = i
                if end > start and data[end - 1] == 0:   # 4-byte start code
                    end -= 1
                nals.append(data[start:end])
            i += 3
            start = i
        else:
            i += 1
    if start >= 0 and start < n:
        nals.append(data[start:])
    return [x for x in nals if x]


def _nal_type(nal):
    return nal[0] & 0x1F


START = b"\x00\x00\x00\x01"


class H264Encoder:
    """Encodes BGR numpy frames into H.264 access units (Annex-B bytes)."""

    def __init__(self, log=print, candidates=None):
        self._log = log
        self._candidates = list(candidates or _encoder_candidates())
        self._bad = set()
        self.ctx = None
        self.name = None
        self.width = self.height = 0
        self._fps = 30
        self._bitrate = 0
        self._t0 = None
        self._last_pts = -1
        self._frames_out = 0
        self._sps = None
        self._pps = None
        self._want_key = True

    # --- lifecycle ------------------------------------------------------------------
    def _open(self, width, height, fps, bitrate):
        skipped = []
        for name in self._candidates:
            if name in self._bad or not _has_codec(name, "w"):
                continue
            try:
                ctx = av.CodecContext.create(name, "w")
                ctx.width, ctx.height = width, height
                ctx.pix_fmt = _pix_fmt(name)
                ctx.time_base = Fraction(1, 1000)
                ctx.framerate = Fraction(int(fps), 1)
                ctx.bit_rate = int(bitrate)
                ctx.gop_size = int(fps) * 10
                ctx.max_b_frames = 0
                ctx.thread_count = 0 if name == "libx264" else 1
                ctx.options = _encoder_options(name, fps, bitrate)
                ctx.open()
            except Exception:
                # Normal on PCs without that GPU vendor; summarised in one log line below.
                self._bad.add(name)
                skipped.append(name.replace("h264_", ""))
                continue
            self.ctx, self.name = ctx, name
            self.width, self.height, self._fps, self._bitrate = width, height, fps, bitrate
            self._t0 = time.monotonic()
            self._last_pts = -1
            self._frames_out = 0
            self._sps = self._pps = None
            self._want_key = True
            kind = "CPU" if name == "libx264" else "GPU"
            note = f" (not available here: {', '.join(skipped)})" if skipped else ""
            self._log(f"🎞️ H.264 via {name} [{kind}] {width}x{height}@{fps}, "
                      f"{bitrate / 1e6:.1f} Mbps{note}")
            return True
        if skipped:
            self._log(f"🎞️ No H.264 encoder could start (tried {', '.join(skipped)})")
        return False

    def close(self):
        self.ctx = None
        self.name = None

    def request_keyframe(self):
        self._want_key = True

    @property
    def ready(self):
        return self.ctx is not None

    # --- encoding ---------------------------------------------------------------------
    def encode(self, bgr, fps=30, quality=40, force_key=False):
        """Encode one frame. Returns a list of (annexb_bytes, is_keyframe). The frame is cropped
        to even dimensions (H.264 needs them). Raises RuntimeError if no encoder works."""
        h, w = bgr.shape[:2]
        w -= w % 2
        h -= h % 2
        if w < 16 or h < 16:
            return []
        if bgr.shape[0] != h or bgr.shape[1] != w:
            bgr = bgr[:h, :w]
        bitrate = bitrate_for(w, h, fps, quality)
        if (self.ctx is None or w != self.width or h != self.height or fps != self._fps
                or abs(bitrate - self._bitrate) > self._bitrate * 0.15):
            self.close()
            if not self._open(w, h, fps, bitrate):
                raise RuntimeError("no working H.264 encoder")

        while True:
            try:
                return self._encode_once(bgr, force_key)
            except Exception as e:
                if self._frames_out > 0:
                    raise
                # Some GPU encoders open fine and only fail on the first real frame.
                self._log(f"🎞️ {self.name} failed on first frame: {str(e)[:120]}")
                self._bad.add(self.name)
                self.close()
                if not self._open(w, h, fps, bitrate):
                    raise RuntimeError("no working H.264 encoder") from e

    def _encode_once(self, bgr, force_key):
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(bgr), format="bgr24")
        frame = frame.reformat(format=self.ctx.pix_fmt)
        pts = int((time.monotonic() - self._t0) * 1000)
        if pts <= self._last_pts:
            pts = self._last_pts + 1
        self._last_pts = pts
        frame.pts = pts
        frame.time_base = self.ctx.time_base
        if force_key or self._want_key:
            self._want_key = False
            _set_keyframe(frame)
        out = []
        for packet in self.ctx.encode(frame):
            data = bytes(packet)
            if not data:
                continue
            out.append(self._with_headers(data, bool(packet.is_keyframe)))
        if out:
            self._frames_out += len(out)
        return out

    def _with_headers(self, data, is_key):
        """Make sure every keyframe carries SPS/PPS (some hardware encoders only emit them
        once, or only in extradata)."""
        nals = split_nals(data)
        types = {_nal_type(n) for n in nals}
        for n in nals:
            t = _nal_type(n)
            if t == 7:
                self._sps = n
            elif t == 8:
                self._pps = n
        if 5 in types:
            is_key = True
        if (self._sps is None or self._pps is None) and self.ctx is not None:
            extra = self.ctx.extradata
            if extra and extra[:3] in (b"\x00\x00\x01", b"\x00\x00\x00"):
                for n in split_nals(bytes(extra)):
                    if _nal_type(n) == 7:
                        self._sps = n
                    elif _nal_type(n) == 8:
                        self._pps = n
        if is_key and (7 not in types or 8 not in types) and self._sps and self._pps:
            data = START + self._sps + START + self._pps + (data if data[:3] == b"\x00\x00\x01"
                                                            or data[:4] == START else START + data)
        return data, is_key


def _set_keyframe(frame):
    try:
        frame.pict_type = av.video.frame.PictureType.I
    except Exception:
        try:
            frame.pict_type = "I"
        except Exception:
            pass


def screen_packet(annexb, is_key, width, height):
    """0x04 frame: [0x04][flags][u16 width][u16 height][Annex-B access unit]."""
    flags = 0x01 if is_key else 0x00
    return bytes((0x04, flags, (width >> 8) & 0xFF, width & 0xFF, (height >> 8) & 0xFF, height & 0xFF)) + annexb


# ------------------------------------------------------------------------------------------
# Opus
# ------------------------------------------------------------------------------------------

OPUS_RATE = 48000


def _layout(channels):
    return "mono" if channels == 1 else "stereo"


class OpusEncoder:
    """Interleaved s16 PCM (any rate, 1 or 2 channels) -> 10 ms Opus packets at 48 kHz."""

    def __init__(self, bitrate_per_channel=64000):
        self._bpc = bitrate_per_channel
        self.ctx = None
        self.channels = 0
        self._in_rate = 0
        self._resampler = None
        self._fifo = None
        self.frame_size = 480

    def _setup(self, rate, channels):
        ctx = av.CodecContext.create("libopus", "w")
        ctx.sample_rate = OPUS_RATE
        ctx.layout = _layout(channels)
        ctx.format = "s16"
        ctx.bit_rate = self._bpc * channels
        ctx.time_base = Fraction(1, OPUS_RATE)
        ctx.options = {"application": "lowdelay", "frame_duration": "10", "vbr": "on"}
        ctx.open()
        self.ctx = ctx
        self.frame_size = ctx.frame_size or 480
        self.channels = channels
        self._in_rate = rate
        self._resampler = None if rate == OPUS_RATE else av.AudioResampler(
            format="s16", layout=_layout(channels), rate=OPUS_RATE)
        self._fifo = av.AudioFifo()

    def encode(self, pcm, rate, channels):
        """Returns a list of Opus packets (bytes); empty until 10 ms has been collected."""
        channels = 1 if channels == 1 else 2
        if self.ctx is None or channels != self.channels or rate != self._in_rate:
            self._setup(rate, channels)
        samples = len(pcm) // (2 * channels)
        if samples == 0:
            return []
        arr = np.frombuffer(pcm, dtype=np.int16, count=samples * channels).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(arr, format="s16", layout=_layout(channels))
        frame.sample_rate = rate
        frames = self._resampler.resample(frame) if self._resampler else [frame]
        for f in frames:
            f.pts = None
            self._fifo.write(f)
        out = []
        while self._fifo.samples >= self.frame_size:
            chunk = self._fifo.read(self.frame_size)
            for packet in self.ctx.encode(chunk):
                b = bytes(packet)
                if b:
                    out.append(b)
        return out


class OpusDecoder:
    """Opus packets -> interleaved s16 PCM at ``out_rate`` with ``out_channels``."""

    def __init__(self, channels=1):
        self.channels = 1 if channels == 1 else 2
        self.ctx = av.CodecContext.create("libopus", "r")
        self.ctx.sample_rate = OPUS_RATE
        self.ctx.layout = _layout(self.channels)
        self._resampler = None
        self._target = None

    def decode(self, packet_bytes, out_rate=48000, out_channels=1):
        target = (int(out_rate), 1 if out_channels == 1 else 2)
        if self._resampler is None or self._target != target:
            self._resampler = av.AudioResampler(format="s16", layout=_layout(target[1]), rate=target[0])
            self._target = target
        chunks = []
        for frame in self.ctx.decode(av.Packet(packet_bytes)):
            frame.pts = None
            for f in self._resampler.resample(frame):
                chunks.append(f.to_ndarray().tobytes())
        return b"".join(chunks)

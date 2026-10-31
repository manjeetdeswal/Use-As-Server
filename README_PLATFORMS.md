# Use As Server — Windows, Linux and macOS

All three servers now share one code base in `useas_server/`:

| File | What it is |
|---|---|
| `useas_server/core.py` | Networking, pairing, protocol, files, clipboard, screen/audio/camera streams |
| `useas_server/protocol.py` | Wire protocol v2 (handshake, payload formats, binary frame types) |
| `useas_server/gui.py` | The desktop window (same on every OS) |
| `useas_server/platforms/windows.py` | SendInput, ViGEm Xbox pads, WASAPI loopback, VB-CABLE, registry auto-start |
| `useas_server/platforms/linux.py` | PyAutoGUI/xdotool, uinput pads, PulseAudio/PipeWire, xrandr, XDG auto-start |
| `useas_server/platforms/mac.py` | Quartz mouse, pynput media keys, BlackHole audio, LaunchAgent auto-start |

`UseAsServer.py`, `UseAsServerLinux.py` and `UseAsServerMac.py` are now 10-line launchers
(any of them works on any OS), so existing shortcuts and build scripts keep working.

## Run from source

```
pip install customtkinter pillow pystray websockets pyautogui pyperclip mss numpy opencv-python psutil pyaudio pyvirtualcam mediapipe qrcode tkinterdnd2 av
```

Plus, per OS:

* **Windows:** `pip install pywin32 vgamepad pyaudiowpatch screen-brightness-control winrt-Windows.Media.Control winrt-Windows.Storage.Streams winrt-Windows.Foundation pycaw`
  (ViGEmBus driver for the gamepad, VB-CABLE for the phone mic)
* **Linux:** `pip install evdev screen-brightness-control`, and `sudo apt install xdotool pulseaudio-utils playerctl xclip wmctrl`.
  Gamepad: `sudo chmod 666 /dev/uinput` (or a udev rule).
  Virtual camera: `sudo modprobe v4l2loopback devices=1 video_nr=10 exclusive_caps=1`.
* **macOS:** `pip install pynput pyobjc-framework-Quartz pyobjc-framework-Cocoa`, and install
  BlackHole (system audio + phone mic) and OBS (virtual camera). Allow the app under
  System Settings → Privacy & Security → **Accessibility** (input) and **Screen Recording** (second screen).

Then: `python UseAsServer.py` (or `python -m useas_server`).

For the phase 3 codecs install PyAV on every OS: `pip install av` (it bundles FFmpeg).

## Build

PyInstaller can't cross-compile: build each version on that system (a VM works, or GitHub
Actions). Use a clean Python 3.10–3.12 with the packages for that OS (see Install above), then:

    pip install pyinstaller
    pyinstaller UseAsServer.spec

The one spec handles all three systems.

* **Windows** → `dist\UseAsServer.exe`. For the installer, open `my.iss` in Inno Setup 6 and
  press Compile → `Output\UseAs_Setup_v….exe` (bundles ViGEmBus and the virtual monitor
  driver from `Final_Build`; adds the firewall rules). Bump `MyAppVersion` in `my.iss` and
  `APP_VERSION` in `useas_server/common.py` together.
* **Linux** → `dist/UseAsServer` (one file; tested). Users still need the system packages:
  `sudo apt install xdotool pulseaudio-utils playerctl xclip` (+ v4l2loopback for the webcam).
  `chmod +x UseAsServer` and run it, or wrap it as an AppImage/.deb.
* **macOS** → `dist/Use As Server.app` (built for the Mac type you build on). For a disk image:
  `hdiutil create -volname "Use As Server" -srcfolder "dist/Use As Server.app" -ov UseAsServer.dmg`.
  Unsigned apps: first launch with right-click › Open. To sign:
  `codesign --deep --force --options runtime --sign "Developer ID Application: …" "dist/Use As Server.app"`,
  then notarize with `xcrun notarytool`. Users install BlackHole and allow Accessibility and
  Screen Recording.

## Phase 2 features

* **Pair with QR** (Dashboard): scan with the app's *Scan QR code from PC* button or the phone camera.
  The code holds every PC address plus a pairing token that works like the password.
  *New code* in that window revokes the old one. Needs `qrcode`.
* **Send to phone** from the file manager: Windows Explorer → *Send to → Use As (phone)*,
  Linux Nautilus → *Scripts → Send to phone (Use As)*. Turn it off in Settings.
  Files wait in a queue until a phone is connected. `UseAsServer.py --send FILE…` does the same.
* **Drag and drop** files onto the window to send them (needs `tkinterdnd2`).
* **Progress bar** for transfers in both directions (Sharing tab).
* **Gamepad player number**: the phone shows P1–P4 for its virtual controller (Windows).
* Opening the app a second time brings the running window to the front instead of
  starting a second server.

## Phase 3 features (needs `pip install av`)

* **H.264 second screen.** Uses the GPU encoder when there is one (NVIDIA NVENC, Intel Quick
  Sync, AMD AMF, Media Foundation on Windows; VideoToolbox on macOS) and x264 otherwise. Far
  less data than JPEG at the same quality, so 1080p60 works on ordinary Wi-Fi. The log shows
  which encoder is in use. If it fails, the phone is switched to JPEG automatically.
* **Unchanged frames are skipped** (JPEG and H.264): a still desktop sends one frame a second.
* **Congestion control:** when Wi-Fi can't keep up, the bitrate steps down and recovers later.
* **Opus audio** both ways: PC sound to the phone at ~128 kbps instead of ~1.5 Mbps, and the
  phone mic to the PC (Android 10+; older phones send PCM).
* **Binary camera frames** (no base64), ~25 % less data.
* Older app versions keep working with JPEG / PCM.

## Phase 4 features

* **The phone as a real extra monitor (Windows).** Pick *Monitor 2* in the app's second
  screen on a PC with one screen (or *Monitor 3* with two…) and a virtual monitor is added:
  it appears in Settings › Display, you can drag windows onto it, and the phone shows it.
  It is removed again ~20 s after the phone stops viewing (and when the server stops).
  Uses Amyuni's free signed **usbmmidd** driver (Final_Build\usbmmidd_v2, see its
  License.txt). The installer has a *Virtual monitor driver* option; source runs install it
  on first use from Final_Build (Windows asks for permission). Turn it off in Settings ›
  Second screen. Supported sizes include 1920x1080 and 2560x1440 (matched to the phone's
  stream resolution). Linux keeps using the xrandr virtual output.
* **Images on the clipboard, both ways.** A screenshot or "Copy image" on the PC lands on
  the phone's clipboard (paste it in any app), and an image copied on the phone goes to the
  PC clipboard, when clipboard auto-sync is on. Linux needs `xclip` or `wl-clipboard`.
* **Copied files** (Ctrl+C in Explorer / Finder / Nautilus) are sent to the phone as files:
  automatically with auto-sync (up to 10 files / 50 MB), or any size with the
  *Send copied image / files* button.

## Also new

* **Now playing on the phone:** title, artist and cover of what the PC plays, with
  previous / play-pause / next / seek in the Speaker tab and as a media notification.
  Windows: `pip install winrt-Windows.Media.Control winrt-Windows.Storage.Streams winrt-Windows.Foundation`;
  Linux: `sudo apt install playerctl`; macOS: Music and Spotify. Without these the buttons
  still work (media keys).
* **Present tab on the phone:** big Next/Previous, volume keys turn slides, laser pointer
  (gyro), talk timer.
* **Three-finger touchpad gestures:** tap = middle click, up = Task view, down = desktop,
  left/right = switch apps.
* **Setup check tab** on the PC (what's installed, with fix buttons) and a **How to use** tab.

## Settings

Kept where each OS server already stored them, and read automatically:
Windows `%APPDATA%\UseAsServer\server_settings.json`, Linux `~/.config/useas_server/settings.json`,
macOS `~/Library/Application Support/UseAsServer/settings.json`.

## Protocol v2 (summary)

* The server sends `hello` as soon as a phone connects: `{"protocol": 2, "server_version", "os", "features", "auth_required"}`.
* New apps answer with their own `hello`: `{"protocol": 2, "app_version", "device", "features"}`.
* Payloads are read per message type (see `PAYLOAD_SHAPES` in `protocol.py`); older apps'
  double-encoded payloads are still accepted.
* Binary frames start with a type byte: 0x01/0x02 file data/end, 0x03 screen JPEG,
  0x04 screen H.264 (`[flags][u16 w][u16 h][Annex-B]`, flags bit 0 = keyframe),
  0x10 PC audio PCM (`[channels][u32 rate][s16le]`), 0x11 PC audio Opus (`[channels][packet]`);
  0x05 clipboard image (PNG) to the phone;
  from the phone 0x20 camera JPEG, 0x21 mic PCM, 0x22 mic Opus (mono 48 kHz), 0x23 clipboard image.
* Codecs are chosen per phone: `"format": "h264"|"jpeg"` in `display_request`,
  `"codec": "opus"|"pcm"` in `audio_control` and `mic_start`. The server answers with
  `display_config` / `audio_config` saying what the phone actually gets.

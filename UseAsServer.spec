# -*- mode: python ; coding: utf-8 -*-
# One spec for all three systems. Build on the system you want the app for
# (PyInstaller can't cross-compile):
#   Windows -> dist\UseAsServer.exe          (then compile my.iss with Inno Setup)
#   Linux   -> dist/UseAsServer              (one file)
#   macOS   -> dist/Use As Server.app
#   pyinstaller UseAsServer.spec
import os
import sys

from PyInstaller.utils.hooks import collect_all

WINDOWS = sys.platform == "win32"
MAC = sys.platform == "darwin"

datas = [('icon.png', '.'),
         # Background blur / replace model: was not bundled, so the effect silently did nothing.
         ('selfie_segmenter.tflite', '.')]
if os.path.exists('icon.ico'):
    datas.append(('icon.ico', '.'))
binaries = []
# The app lives in the useas_server package; the OS backend is picked at runtime.
hiddenimports = ['useas_server.platforms.windows' if WINDOWS else
                 'useas_server.platforms.mac' if MAC else 'useas_server.platforms.linux']

# Packages with native libraries or data files the default analysis misses. Missing optional
# ones are skipped (the app then runs without that feature; see the Setup tab).
#   mediapipe: background effects   tkinterdnd2: drag and drop   qrcode: pairing QR
#   av (PyAV): H.264 second screen + Opus audio   pyvirtualcam: webcam
#   winrt / winsdk: song title and cover for the phone (Windows)
packages = ['customtkinter', 'mediapipe', 'pyvirtualcam', 'tkinterdnd2', 'qrcode', 'av']
if WINDOWS:
    packages += ['vgamepad', 'pyaudiowpatch', 'winrt', 'winsdk', 'pycaw', 'comtypes']
elif MAC:
    packages += ['pynput']
else:
    packages += ['evdev']

no_upx = []
for pkg in packages:
    try:
        tmp_ret = collect_all(pkg)
    except Exception as e:   # optional package not installed: build without that feature
        print(f"WARNING: skipping {pkg}: {e}")
        continue
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
    if pkg == 'av':
        # UPX can break FFmpeg's DLLs (they use control-flow guard); keep them as they are.
        no_upx += [os.path.basename(b[0]) for b in tmp_ret[1]]


a = Analysis(
    ['UseAsServer.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
if not WINDOWS and not MAC:
    # Linux: use the computer's own font libraries, not the build machine's. The GitHub build
    # runs on Ubuntu 22.04, whose libXft (2.3.4) kills the app with "X Error ... BadLength"
    # as soon as Tk draws a colour emoji (fixed in libXft 2.3.5). fontconfig/freetype go with
    # it so the system libXft gets the versions it was built against. Every Linux desktop has
    # these. (Pillow/OpenCV keep their own renamed copies, e.g. libfreetype-1a2b3c.so, untouched.)
    _system_libs = ('libXft.so', 'libfontconfig.so', 'libfreetype.so')
    a.binaries = [b for b in a.binaries if not os.path.basename(b[0]).startswith(_system_libs)]

pyz = PYZ(a.pure)

if MAC:
    # A normal .app (one folder inside the bundle starts much faster than one file).
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='UseAsServer',
        debug=False,
        strip=False,
        upx=False,
        console=False,
        argv_emulation=False,
        target_arch=None,          # builds for the Mac type it runs on (Apple silicon or Intel)
        codesign_identity=None,
        entitlements_file=None,
        icon='icon.png',           # PyInstaller converts it to .icns (needs Pillow)
    )
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='UseAsServer')
    app = BUNDLE(
        coll,
        name='Use As Server.app',
        icon='icon.png',
        bundle_identifier='com.jeetstudio.useasserver',
        info_plist={
            'CFBundleShortVersionString': os.environ.get('VERSION', '2.0'),   # set by the workflow
            'NSHighResolutionCapable': True,
            # Asked the first time each is used:
            'NSMicrophoneUsageDescription':
                "Use As Server records the BlackHole device to stream your Mac's sound to your phone.",
            'NSAppleEventsUsageDescription':
                'Use As Server reads what Music or Spotify is playing to show it on your phone.',
        },
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='UseAsServer',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=WINDOWS,
        upx_exclude=no_upx,
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=['icon.ico'] if WINDOWS else None,
    )

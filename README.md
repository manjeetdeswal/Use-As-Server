Use As Server 🖥️📱
Use As turns your Android device into a suite of powerful PC peripherals. Control your computer, stream video, extend your display, and use your phone as a microphone, all with low latency over WiFi or USB.

🌐 Official Website & Guide: https://manjeetdeswal.github.io/Use-As-Server/

✨ Features
🖱️ Smart Touchpad: Smooth mouse control with acceleration (ballistics) and multi-monitor support.

⌨️ Full Keyboard: 83-key layout with modifier support (Ctrl, Alt, Shift, Win) and live text preview.

🎮 Virtual Gamepad: Emulates an Xbox 360 controller for playing PC games (supports analog triggers & sticks).

📹 Virtual Webcam:

Unity Mode: Direct, high-performance webcam driver (Unity Capture).

OBS Mode: Compatible with OBS Virtual Camera.

🎤 Wireless Microphone: Routes phone audio to your PC as a microphone input with noise suppression.

🖥️ Second Display: Extend your Windows desktop to your phone screen (requires usbmmidd).

📂 File Transfer: Drag & drop files PC-to-Phone, or send Phone-to-PC instantly.

⚡ Zero Latency: Optimized for USB Tethering and 5GHz WiFi.

📥 Installation
You can run the server using the Windows Installer (Recommended) or from Python Source.

Option A: Windows Installer (Recommended)
Download the latest UseAs_Setup.exe from the Releases Page.

Run the installer to set up the server application.

⚠️ Install Required Drivers Manually:

Microphone: VB-CABLE (Required for Audio).

Webcam: Unity Capture (Required for Camera).

Display: usbmmidd_v2 (Required for Second Monitor).

Launch "Use As Server" from your desktop.

Option B: Run from Source (Advanced)
If you prefer to run the raw Python script, follow these steps.

1. Prerequisites
Python 3.10 or newer.

2. Install Drivers
Gamepad: Download & Install ViGEmBus.

Microphone: Download & Install VB-CABLE.

Webcam: Download Unity Capture → Run Install.bat as Administrator.

Display: Download usbmmidd_v2.

Extract zip.

Open CMD as Admin in that folder.

Run: deviceinstaller64 install usbmmidd.inf usbmmidd

3. Install Dependencies
Run the included batch file:

DOS

requirements.bat
4. Run
Launch the GUI:

DOS

python unified_remote_gui.py

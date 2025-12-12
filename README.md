# Use As Server 🖥️📱

**Use As** turns your Android device into a suite of powerful PC peripherals. Control your computer, stream video, extend your display, and use your phone as a microphone, all with low latency over WiFi or USB.

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg) ![Python](https://img.shields.io/badge/python-3.10%2B-yellow.svg)

<bold>Website Link</bold> - "https://manjeetdeswal.github.io/Use-As-Server/"

---

## ✨ Features

* **🖱️ Smart Touchpad:** Smooth mouse control with acceleration (ballistics) and multi-monitor support.
* **⌨️ Full Keyboard:** 83-key layout with modifier support (Ctrl, Alt, Shift, Win) and live text preview.
* **🎮 Virtual Gamepad:** Emulates an Xbox 360 controller for playing PC games (supports analog triggers & sticks).
* **📹 High-Quality Webcam:** Use your phone camera as a webcam in Zoom, Discord, OBS, etc. (Supports rotation & flash).
* **🎤 Wireless Microphone:** Routes phone audio to your PC as a microphone input with noise suppression.
* **🖥️ Second Display:** Extend your Windows desktop to your phone screen.
* **⚡ Zero Latency:** Optimized for USB Tethering and 5GHz WiFi.

---

## 📥 Installation

You have two options to run the server: using the **All-in-One Installer** (Recommended) or running from **Python Source**.

### Option A: All-in-One Installer (Windows)
This handles everything for you, including driver installation.

1.  Download the latest **`UseAs_Setup.exe`** from the [Releases Page](#).
2.  Run the installer as Administrator.
3.  Follow the prompts to install required drivers:
    * **ViGEmBus:** For Gamepad support.
    * **VB-Cable:** For Microphone support.
4.  Launch "Use As Server" from your desktop.

---

### Option B: Run from Source (Advanced)

If you prefer to run the raw Python script or are on Linux, follow these steps.

#### 1. Prerequisites
* **Python 3.10** or newer.
* **OBS Studio** (For Virtual Camera support).

#### 2. Install Drivers
* **Gamepad:** Download and install [ViGEmBus](https://github.com/ViGEm/ViGEmBus/releases) (Windows only).
* **Microphone:** Download and install [VB-CABLE](https://vb-audio.com/Cable/).
* **Display:** Download [usbmmidd_v2](https://www.amyuni.com/downloads/usbmmidd_v2.zip) (extract and run `deviceinstaller64 install usbmmidd.inf usbmmidd`).

#### 3. Install Dependencies
  Run requirement.bat file

#### 4. Run
  Double Click or Open with command "python unified_remote_gui.py"

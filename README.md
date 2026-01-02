<div align="center">

# Use As Server 🖥️📱

**Turn your Android device into a suite of powerful PC peripherals.** Control your computer, stream video, extend your display, and use your phone as a microphone, all with low latency over WiFi or USB.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-yellow.svg)](https://www.python.org/)

[**🌐 Visit Official Website & Guide**](https://manjeetdeswal.github.io/Use-As-Server/)

</div>

---

## ✨ Features

| Feature | Description |
| :--- | :--- |
| **🖱️ Smart Touchpad** | Smooth mouse control with acceleration (ballistics) and multi-monitor support. |
| **⌨️ Full Keyboard** | 83-key layout with modifier support (Ctrl, Alt, Shift, Win) and live text preview. |
| **🎮 Virtual Gamepad** | Emulates an **Xbox 360 controller** for playing PC games (supports analog triggers & sticks). |
| **📹 Virtual Webcam** | Supports **Unity Mode** (Direct Driver) and **OBS Mode** (Virtual Camera). |
| **🎤 Wireless Mic** | Routes phone audio to your PC as a microphone input with noise suppression. |
| **🖥️ Second Display** | Extend your Windows desktop workspace to your phone screen. |
| **📂 File Transfer** | Drag & drop files PC-to-Phone, or send Phone-to-PC instantly. |
| **⚡ Zero Latency** | Optimized for **USB Tethering** and **5GHz WiFi**. |

---

## 📥 Installation

You can run the server using the **Windows Installer** (Recommended) or from **Python Source**.

### Option A: Windows Installer (Recommended)

1.  Download the latest **`UseAs_Setup.exe`** from the [**Releases Page**](https://github.com/manjeetdeswal/Use-As-Server/releases).
2.  Run the installer to set up the server application.
3.  **Install Drivers:** (See box below).
4.  Launch **Use As Server** from your desktop.

> [!IMPORTANT]
> **Drivers are NOT included in the installer.**
> To unlock all features, you must download and install these specific drivers manually:
>
> * 🎮 **Gamepad:** [Download ViGEmBus](https://github.com/nefarius/ViGEmBus/releases/latest) (Required for Controller)
> * 🎤 **Microphone:** [Download VB-CABLE](https://vb-audio.com/Cable/) (Required for Audio)
> * 📸 **Webcam:** [Download Unity Capture](https://github.com/schellingb/UnityCapture) (Required for Camera)
> * 🖥️ **Display:** [Download usbmmidd_v2](https://www.datronicsoft.com/download/usbmmidd_v2.zip) (Required for Second Monitor)

---

### Option B: Run from Source (Advanced)

If you are a developer or on Linux, you can run the raw Python script.

#### 1. Prerequisites
* **Python 3.10** or newer installed.

#### 2. Install Drivers
* **Gamepad:** Install [ViGEmBus](https://github.com/nefarius/ViGEmBus/releases/latest).
* **Microphone:** Install [VB-CABLE](https://vb-audio.com/Cable/).
* **Webcam:** Download [Unity Capture](https://github.com/schellingb/UnityCapture) and run `Install.bat` as Administrator.
* **Display:** Download [usbmmidd_v2](https://www.amyuni.com/downloads/usbmmidd_v2.zip).
    * Extract the zip file.
    * Open Command Prompt **as Administrator** in that folder.
    * Run the following command:
        ```cmd
        deviceinstaller64 install usbmmidd.inf usbmmidd
        ```

#### 3. Install Dependencies
Run the included batch file to install Python libraries:
```cmd
requirements.bat


#### 4. Run Server
Launch the GUI:

```cmd
python unified_remote_gui.py

<<<<<< Built  by Manjeet Deswal >>>>>>>


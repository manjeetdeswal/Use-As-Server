<div align="center">

# Use As Server 🖥️📱

**Turn your Android device into a suite of powerful PC peripherals.** Control your computer, stream video, extend your display, and use your phone as a microphone, all with low latency over WiFi or USB.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-green.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-yellow.svg)](https://www.python.org/)

[**🌐 Visit Official Website & Guide**](https://manjeetdeswal.github.io/Use-As-Server/)

<p align="center">
  <img src="https://github.com/manjeetdeswal/Use-As-Server/blob/main/ss/Screenshot%202026-01-07%20152541.png?raw=true" width="45%" alt="Home Screen" />
  <img src="https://github.com/manjeetdeswal/Use-As-Server/blob/main/ss/Screenshot%202026-01-07%20152549.png?raw=true" width="45%" alt="Smart Touchpad" />
</p>
<p align="center">
  <img src="https://github.com/manjeetdeswal/Use-As-Server/blob/main/ss/Screenshot%202026-01-07%20152553.png?raw=true" width="45%" alt="Keyboard" />
  <img src="https://github.com/manjeetdeswal/Use-As-Server/blob/main/ss/Screenshot%202026-01-07%20152558.png?raw=true" width="45%" alt="Display Extension" />
</p>
<p align="center">
  <img src="https://github.com/manjeetdeswal/Use-As-Server/blob/main/ss/Screenshot%202026-01-07%20152608.png?raw=true" width="45%" alt="Mic Streaming" />
</p>


</div>

---

## ✨ Features

| Feature | Description |
| :--- | :--- |
| **🖱️ Smart Touchpad** | Smooth mouse control with acceleration (ballistics) and multi-monitor support. |
| **⌨️ Full Keyboard** | 83-key layout with modifier support (Ctrl, Alt, Shift, Win) and live text preview. |
| **🎮 Virtual Gamepad** | Emulates an **Xbox 360 controller** (Windows) or **Virtual Input** (Linux) for gaming. |
| **📹 Virtual Webcam** | Supports **Unity Capture** (Windows) and **v4l2loopback** (Linux). |
| **🎤 Wireless Mic** | Routes phone audio to your PC microphone input with noise suppression. |
| **🖥️ Second Display** | Extend your desktop workspace to your phone screen. |
| **📂 File Transfer** | Drag & drop files PC-to-Phone, or send Phone-to-PC instantly. |
| **⚡ Zero Latency** | Optimized for **USB Tethering** and **5GHz WiFi**. |

---

## 📥 Installation

You can install the server using the installers below or run it from the Python source.

### 🪟 Option A: Windows Installer

1.  Download the latest **`UseAs_Setup.exe`** from the [**Releases Page**](https://github.com/manjeetdeswal/Use-As-Server/releases).
2.  Run the installer to set up the server application.
3.  **Install Drivers:** (See box below).
4.  Launch **Use As Server** from your desktop.

> [!IMPORTANT]
> **Windows Drivers Required**
> To unlock all features, download these drivers manually:
> * 🎮 **Gamepad:** [Download ViGEmBus](https://github.com/nefarius/ViGEmBus/releases/latest)
> * 🎤 **Microphone:** [Download VB-CABLE](https://vb-audio.com/Cable/)
> * 📸 **Webcam:** [Download Unity Capture](https://github.com/schellingb/UnityCapture)
> * 🖥️ **Display:** [Download usbmmidd_v2](https://www.datronicsoft.com/download/usbmmidd_v2.zip)

---

### 🐧 Option B: Linux Installation (Debian/Ubuntu)

1.  Download the latest **`use-as-server_1.0_all.deb`** from the [**Releases Page**](https://github.com/manjeetdeswal/Use-As-Server/releases).
2.  Open a terminal in your download folder and run:
    ```bash
    sudo apt update
    sudo apt install ./use-as-server_1.0_all.deb
    ```
    *(This automatically installs Python, FFmpeg, and required tools).*

3.  **Linux Setup & Configuration:**

    | Feature | Linux Setup Command |
    | :--- | :--- |
    | **📸 Camera** | `sudo modprobe v4l2loopback exclusive_caps=1 card_label="UseAs Cam"` |
    | **🎤 Mic/Audio** | Install **PulseAudio Control** to route audio:<br>`sudo apt install pavucontrol`<br>Open `pavucontrol` to select "UseAs_Mic" as your input source. |
    | **🎮 Gamepad** | Handled automatically via `uinput`. If issues arise: `sudo chmod 666 /dev/uinput` |

---

### 🐍 Option C: Run from Source (Advanced)

If you are a developer, you can run the raw Python script.

#### 1. Prerequisites
* **Python 3.10** or newer installed.

#### 2. Install Dependencies
**Windows:**
```cmd
pip install -r requirements.txt

#### **Linux:**
```bash
sudo apt install python3-tk python3-pip python3-venv v4l2loopback-dkms portaudio19-dev
pip3 install -r requirements.txt --break-system-packages

#### 3. Run Server

```bash
python unified_remote_gui.py

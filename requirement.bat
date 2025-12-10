@echo off
setlocal
title Installing Unified Remote Server Dependencies
color 0A

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo ====================================================
    echo [ERROR] Python is not found in your system PATH!
    echo Please install Python 3.10+ and check "Add to PATH"
    echo ====================================================
    pause
    exit /b
)

echo ====================================================
echo      Use As Server - Dependency Installer
echo ====================================================
echo.

:: 1. UPGRADE PIP & SETUP TOOLS
echo [1/4] Upgrading pip and setup tools...
python -m pip install --upgrade pip setuptools wheel
echo.

:: 2. INSTALL PYTHON LIBRARIES
echo [2/4] Installing Python Libraries...
echo -----------------------------------
echo  - GUI:       tkinter (built-in)
echo  - Network:   websockets
echo  - Input:     pyautogui, pywin32
echo  - Video:     opencv-python, mss, pillow, numpy, pyvirtualcam
echo  - Audio:     pyaudio, pyaudiowpatch
echo  - System:    psutil
echo  - Gamepad:   vgamepad
echo -----------------------------------
echo.

pip install websockets pyautogui pyperclip mss pillow numpy opencv-python psutil pyvirtualcam pyaudio pyaudiowpatch vgamepad pywin32 requests

if %errorlevel% neq 0 (
    color 0E
    echo.
    echo [WARNING] Some libraries failed to install.
    echo If PyAudio failed, you may need C++ Build Tools.
    echo.
    pause
    color 0A
)

:: 3. GAMEPAD DRIVER (ViGEmBus) - WITH CHECK & FIX
echo.
echo [3/4] Checking Gamepad Driver (ViGEmBus)...

:: CHECK IF ALREADY INSTALLED (Checks for the System Service)
sc query ViGEmBus >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] ViGEmBus is already installed! Skipping...
    goto skip_driver
)

echo       Driver not found. This is REQUIRED for the Virtual Controller.
echo.
choice /M "Do you want to download and install the ViGEmBus Driver now?"
if %errorlevel% equ 1 (
    echo.
    echo Downloading ViGEmBus installer...
    
    :: Delete any broken partial downloads
    if exist ViGEmBus_Setup.exe del ViGEmBus_Setup.exe

    :: METHOD 1: Try CURL (Best for Windows 10/11)
    curl -L -o ViGEmBus_Setup.exe "https://github.com/ViGEm/ViGEmBus/releases/download/v1.21.442.0/ViGEmBus_Setup_1.21.442.x64.exe"
    
    :: METHOD 2: PowerShell Fallback (FIXED: Forces TLS 1.2)
    if not exist ViGEmBus_Setup.exe (
        echo Curl failed. Trying PowerShell with TLS 1.2 fix...
        powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/ViGEm/ViGEmBus/releases/download/v1.21.442.0/ViGEmBus_Setup_1.21.442.x64.exe' -OutFile 'ViGEmBus_Setup.exe'"
    )
    
    :: VALIDATE DOWNLOAD (Check if file exists and is > 0 bytes)
    for %%F in (ViGEmBus_Setup.exe) do if %%~zF LSS 1000 (
        echo [ERROR] Download failed or file is corrupted/empty.
        del ViGEmBus_Setup.exe
    )

    if exist ViGEmBus_Setup.exe (
        echo.
        echo [INSTALLING] Launching setup... Please allow Admin access.
        start /wait ViGEmBus_Setup.exe
        echo.
        echo Cleaning up...
        del ViGEmBus_Setup.exe
        echo ViGEmBus installation finished.
    ) else (
        color 0C
        echo.
        echo [ERROR] Failed to download driver automatically.
        echo Please download it manually from:
        echo https://github.com/ViGEm/ViGEmBus/releases/latest
        echo.
        pause
        color 0A
    )
) else (
    echo Skipping gamepad driver.
)

:skip_driver

:: 4. OPTIONAL DRIVERS (Audio/Video)
echo.
echo [4/4] Optional Drivers (Audio & Video)
echo.
echo To use "Audio Streaming", you need a virtual audio cable.
echo To use "Virtual Camera", you need OBS Studio installed.
echo.
choice /M "Do you want to open the download pages for VB-Cable and OBS?"
if %errorlevel% equ 1 (
    echo Opening VB-Cable website...
    start https://vb-audio.com/Cable/
    echo Opening OBS Studio website...
    start https://obsproject.com/download
)

echo.
echo ====================================================
echo        Installation Complete!
echo ====================================================
echo You can now run the server.
echo.
pause

#!/bin/sh
# Runs as root after the .deb / .rpm is installed or upgraded.
# Sets up what a normal app can't do by itself: the gamepad device and the virtual webcam.
# Nothing here is fatal: the app still installs if a step fails.

# --- leftovers of the old 1.5 package (Python venv and its config) -------------------------
rm -rf /usr/share/useas/venv 2>/dev/null
rmdir /usr/share/useas 2>/dev/null
rm -f /etc/udev/rules.d/99-useas-uinput.rules
if grep -qs "UseAs_Cam" /etc/modprobe.d/v4l2loopback.conf; then
    rm -f /etc/modprobe.d/v4l2loopback.conf          # it used video_nr=20; the app wants 10
fi

# --- gamepad: let the logged-in user use /dev/uinput (same rule Steam uses) -----------------
cat > /etc/udev/rules.d/60-use-as-server-uinput.rules <<'RULE'
KERNEL=="uinput", SUBSYSTEM=="misc", TAG+="uaccess", OPTIONS+="static_node=uinput"
RULE
echo uinput > /etc/modules-load.d/use-as-server-uinput.conf
modprobe uinput 2>/dev/null || true
if command -v udevadm >/dev/null 2>&1; then
    udevadm control --reload-rules 2>/dev/null || true
    udevadm trigger --sysname-match=uinput 2>/dev/null || true
fi

# --- webcam: v4l2loopback as /dev/video10, now and after every reboot -----------------------
cat > /etc/modprobe.d/use-as-server-v4l2loopback.conf <<'CONF'
options v4l2loopback devices=1 video_nr=10 exclusive_caps=1 card_label="UseAs Cam"
CONF
echo v4l2loopback > /etc/modules-load.d/use-as-server-v4l2loopback.conf
if modinfo v4l2loopback >/dev/null 2>&1; then
    if ! lsmod | grep -q '^v4l2loopback'; then
        modprobe v4l2loopback 2>/dev/null || true
    fi
    if [ ! -e /dev/video10 ]; then
        echo "Use As Server: the webcam (/dev/video10) will be ready after a restart."
    fi
else
    echo "Use As Server: for the webcam, install v4l2loopback (Ubuntu/Mint: v4l2loopback-dkms;"
    echo "Fedora: akmod-v4l2loopback from RPM Fusion) and restart."
fi

# menu icon
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -q -t /usr/share/icons/hicolor 2>/dev/null || true
fi
exit 0

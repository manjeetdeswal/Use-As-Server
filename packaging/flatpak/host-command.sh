#!/bin/sh
# Installed in the Flatpak as /app/bin/xdotool, /app/bin/pactl, /app/bin/systemctl, …
# Runs the first of:
#   1. the copy built into the Flatpak (/app/tools, made by build-tools.sh)
#   2. the Flatpak runtime's own copy
#   3. the computer's copy, via flatpak-spawn --host
# Power, session and app-launch commands always use 3: they must act on the computer itself.
t=$(basename "$0")
case "$t" in
    systemctl|loginctl|shutdown|gtk-launch|nvidia-smi|xdg-screensaver|gnome-screensaver-command)
        ;;
    *)
        if [ -x "/app/tools/bin/$t" ]; then
            export LD_LIBRARY_PATH="/app/tools/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            exec "/app/tools/bin/$t" "$@"
        fi
        if [ -x "/usr/bin/$t" ]; then
            exec "/usr/bin/$t" "$@"
        fi
        ;;
esac
exec flatpak-spawn --host "$t" "$@"

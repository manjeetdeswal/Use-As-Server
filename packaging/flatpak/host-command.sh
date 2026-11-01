#!/bin/sh
# Installed as /app/bin/xdotool, /app/bin/pactl, … inside the Flatpak: runs the real tool
# of the same name on the computer (install it there: see README_PLATFORMS.md).
exec flatpak-spawn --host "$(basename "$0")" "$@"

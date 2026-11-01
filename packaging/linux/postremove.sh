#!/bin/sh
# Runs as root after the package is removed. Only cleans up on a real removal, not on upgrades:
#   .deb passes "remove" or "purge" (and "upgrade" on upgrades);
#   .rpm passes how many copies stay installed (0 = removed, 1 = upgraded).
case "$1" in
    remove|purge|0) ;;
    *) exit 0 ;;
esac
rm -f /etc/udev/rules.d/60-use-as-server-uinput.rules \
      /etc/modules-load.d/use-as-server-uinput.conf \
      /etc/modprobe.d/use-as-server-v4l2loopback.conf \
      /etc/modules-load.d/use-as-server-v4l2loopback.conf
if command -v udevadm >/dev/null 2>&1; then
    udevadm control --reload-rules 2>/dev/null || true
fi
exit 0

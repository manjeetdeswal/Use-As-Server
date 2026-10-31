"""Pick the platform backend for the OS we're running on."""

import sys

_platform = None


def get_platform():
    global _platform
    if _platform is None:
        # Explicit imports (not importlib) so PyInstaller bundles the right module.
        if sys.platform == "win32":
            from .windows import WindowsPlatform as cls
        elif sys.platform == "darwin":
            from .mac import MacPlatform as cls
        else:
            from .linux import LinuxPlatform as cls
        _platform = cls()
    return _platform

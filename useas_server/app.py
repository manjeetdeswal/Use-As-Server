"""Entry point shared by the Windows, Linux and macOS launchers.

Command-line flags:
  --send FILE…  send files to the phone (used by the file manager's "Send to" menu);
                hands them to the running app if there is one
  --no-admin    don't relaunch as administrator even if Settings asks for it
  --elevated    (internal) this copy was started by the admin relaunch
"""

import asyncio
import faulthandler
import multiprocessing
import sys
import os
import traceback
from pathlib import Path


def _crash_log_path(platform):
    try:
        platform.settings_dir.mkdir(parents=True, exist_ok=True)
        return platform.settings_dir / "crash.log"
    except Exception:
        return Path.home() / "useas_server_crash.log"


def _send_args(args):
    """Absolute paths of the files after --send (file managers may pass relative names)."""
    if "--send" not in args:
        return None
    files = []
    for a in args[args.index("--send") + 1:]:
        if a.startswith("--"):
            continue
        files.append(os.path.abspath(a))
    return files


def _restore_library_path():
    """The Linux build (PyInstaller) points LD_LIBRARY_PATH at its own bundled libraries.
    Programs it starts (xdotool, pactl, playerctl, the browser…) inherit that and can pick up
    the wrong library versions, so give them the computer's original setting back. This
    process is unaffected: the loader read LD_LIBRARY_PATH when it started."""
    if not getattr(sys, "frozen", False) or sys.platform != "linux":
        return
    orig = os.environ.pop("LD_LIBRARY_PATH_ORIG", None)
    if orig is not None:
        os.environ["LD_LIBRARY_PATH"] = orig
    else:
        os.environ.pop("LD_LIBRARY_PATH", None)


def main():
    multiprocessing.freeze_support()
    _restore_library_path()
    args = sys.argv[1:]
    elevated = "--elevated" in args
    send_files = _send_args(args)

    # Already running? Hand over and exit, instead of starting a second server.
    from .ipc import send_to_running
    if send_files is not None:
        if send_to_running({"cmd": "send", "files": send_files}):
            print(f"Sent {len(send_files)} file(s) to the running Use As Server.", flush=True)
            return
    elif not elevated and send_to_running({"cmd": "show"}):
        print("Use As Server is already running — brought its window to the front.", flush=True)
        return

    from .common import APP_VERSION
    print(f"Use As Server {APP_VERSION} starting…", flush=True)

    from .platforms import get_platform
    platform = get_platform()
    crash_log = _crash_log_path(platform)
    crash_file = None
    try:
        # Native crashes (inside drivers / C libraries) print no Python error; this
        # records them, and every other error is written to the same file below.
        crash_file = open(crash_log, "w", encoding="utf-8")
        faulthandler.enable(file=crash_file, all_threads=True)
    except Exception:
        faulthandler.enable()

    try:
        platform.early_init()          # before any window exists (DPI awareness on Windows)

        from .common import load_prefs
        prefs = load_prefs(platform)   # also migrates settings from older server versions

        wants_admin = prefs.get("run_as_admin") and platform.supports_admin()
        if wants_admin and not elevated and "--no-admin" not in args and not platform.is_admin():
            print("Settings → 'Request admin rights' is on: restarting as administrator…\n"
                  "(Accept the Windows prompt. To skip this, run with --no-admin or turn the setting off.)",
                  flush=True)
            extra = ["--elevated"] + (["--send", *send_files] if send_files else [])
            if platform.relaunch_as_admin(extra):
                return
            print("Couldn't restart as administrator — continuing without admin rights.", flush=True)

        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        platform.ensure_firewall_rule()

        print(f"Opening the window ({platform.display_name})…", flush=True)
        from .gui import ServerGUI
        app = ServerGUI()
        if send_files:
            app.root.after(500, lambda: app.queue_files(send_files))
        app.root.mainloop()
        print("Use As Server closed.", flush=True)
    except SystemExit:
        raise
    except BaseException:
        details = traceback.format_exc()
        print("\n❌ Use As Server crashed:\n" + details, file=sys.stderr, flush=True)
        try:
            if crash_file is not None:
                crash_file.write(details)
                crash_file.flush()
            print(f"(Saved to {crash_log})", file=sys.stderr, flush=True)
        except Exception:
            pass
        if elevated:
            # The admin copy runs in its own console window: keep it open so the error can be read.
            try:
                input("Press Enter to close…")
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()

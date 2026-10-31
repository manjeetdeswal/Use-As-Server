"""Use As Server launcher (Linux).

All the code lives in the useas_server package (shared by Windows, Linux and macOS);
this file only exists so existing build scripts and shortcuts keep working.
Run: python UseAsServerLinux.py
"""

from useas_server.app import main

if __name__ == "__main__":
    main()

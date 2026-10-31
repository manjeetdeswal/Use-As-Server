"""Live PC status for the phone's PC tab: CPU, memory, GPU, disks, battery, network.

All values are best effort: anything this PC can't report is left out (None).
GPU load / temperature come from nvidia-smi (NVIDIA cards); CPU temperature from the
sensors psutil can read (Linux; Windows doesn't expose it without extra tools).
"""
import os
import subprocess
import sys
import threading
import time

import psutil

WINDOWS = sys.platform == "win32"
MAC = sys.platform == "darwin"

_lock = threading.Lock()
_last_net = {"t": 0.0, "sent": 0, "recv": 0}
_gpu = {"at": 0.0, "value": None, "retry_at": 0.0}
_cpu_name = None


def cpu_name():
    global _cpu_name
    if _cpu_name is not None:
        return _cpu_name
    name = ""
    try:
        if WINDOWS:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                name = winreg.QueryValueEx(k, "ProcessorNameString")[0]
        elif MAC:
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=3).stdout
        else:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1]
                        break
    except Exception:
        name = ""
    _cpu_name = " ".join(str(name).split())
    return _cpu_name


def _gpu_stats():
    """NVIDIA GPU via nvidia-smi, cached ~1.5 s; None when there's no such GPU."""
    now = time.time()
    if now - _gpu["at"] < 1.5 or now < _gpu["retry_at"]:
        return _gpu["value"]
    _gpu["at"] = now
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
            creationflags=0x08000000 if WINDOWS else 0)
        line = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ""
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            def num(v):
                try:
                    return float(v)
                except ValueError:
                    return None
            _gpu["value"] = {"name": parts[0], "load": num(parts[1]), "temp": num(parts[2]),
                             "mem_used": num(parts[3]), "mem_total": num(parts[4])}
        else:
            _gpu["value"] = None
            _gpu["retry_at"] = now + 60
    except Exception:
        _gpu["value"] = None
        _gpu["retry_at"] = now + 60          # no nvidia-smi: don't keep trying every poll
    return _gpu["value"]


def _cpu_temp():
    try:
        temps = psutil.sensors_temperatures()          # Linux / FreeBSD only
    except Exception:
        return None
    best = None
    for key in ("coretemp", "k10temp", "zenpower", "cpu_thermal", "acpitz"):
        for t in temps.get(key, []):
            if t.current and (best is None or t.current > best):
                best = t.current
        if best is not None:
            return round(best)
    return None


def _disks():
    out = []
    seen = set()
    for part in psutil.disk_partitions(all=False):
        mp = part.mountpoint
        if mp in seen:
            continue
        if WINDOWS:
            if "cdrom" in (part.opts or "").lower() or not part.fstype:
                continue
        elif not (mp == "/" or mp.startswith(("/home", "/media", "/mnt", "/Volumes", "/run/media"))):
            continue
        try:
            u = psutil.disk_usage(mp)
        except Exception:
            continue
        seen.add(mp)
        label = mp.rstrip("\\/") if WINDOWS else ("System" if mp == "/" else os.path.basename(mp))
        out.append({"name": label or mp, "used": u.used, "total": u.total})
        if len(out) >= 8:
            break
    return out


def stats():
    """One snapshot. Call every second or two (rates are since the last call)."""
    with _lock:
        now = time.time()
        vm = psutil.virtual_memory()
        net = psutil.net_io_counters()
        up = down = 0
        if _last_net["t"]:
            dt = max(0.2, now - _last_net["t"])
            up = max(0, int((net.bytes_sent - _last_net["sent"]) / dt))
            down = max(0, int((net.bytes_recv - _last_net["recv"]) / dt))
        _last_net.update(t=now, sent=net.bytes_sent, recv=net.bytes_recv)

        battery = None
        try:
            b = psutil.sensors_battery()
            if b is not None:
                secs = b.secsleft if isinstance(b.secsleft, int) and b.secsleft > 0 else None
                battery = {"percent": round(b.percent), "plugged": bool(b.power_plugged), "secs_left": secs}
        except Exception:
            battery = None

        try:
            freq = psutil.cpu_freq()
            mhz = round(freq.current) if freq and freq.current else None
        except Exception:
            mhz = None

        return {
            "cpu": psutil.cpu_percent(interval=None),
            "cpu_name": cpu_name(),
            "cpu_cores": psutil.cpu_count(logical=True),
            "cpu_mhz": mhz,
            "cpu_temp": _cpu_temp(),
            "ram_used": vm.total - vm.available,
            "ram_total": vm.total,
            "gpu": _gpu_stats(),
            "disks": _disks(),
            "battery": battery,
            "net_up": up,
            "net_down": down,
            "uptime": int(now - psutil.boot_time()),
            "processes": len(psutil.pids()),
        }

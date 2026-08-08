import os
import time
from typing import Any

_START_TIME = time.time()

try:
    import psutil
except Exception:  # pragma: no cover - optional fallback
    psutil = None

_PROCESS = psutil.Process(os.getpid()) if psutil is not None else None
if _PROCESS is not None:
    try:
        _PROCESS.cpu_percent(interval=None)
    except Exception:
        pass


def runtime_status() -> dict[str, Any]:
    now = time.time()
    status: dict[str, Any] = {
        "pid": os.getpid(),
        "uptime_s": round(now - _START_TIME, 1),
        "started_at": _START_TIME,
    }
    if _PROCESS is None:
        status["available"] = False
        return status

    status["available"] = True
    try:
        mem = _PROCESS.memory_info()
        status["rss_mb"] = round(mem.rss / (1024 * 1024), 1)
        status["vms_mb"] = round(mem.vms / (1024 * 1024), 1)
        status["memory_percent"] = round(_PROCESS.memory_percent(), 2)
    except Exception:
        status["memory_error"] = True

    try:
        status["cpu_percent"] = round(_PROCESS.cpu_percent(interval=None), 1)
    except Exception:
        status["cpu_error"] = True

    try:
        status["num_threads"] = _PROCESS.num_threads()
    except Exception:
        pass

    return status

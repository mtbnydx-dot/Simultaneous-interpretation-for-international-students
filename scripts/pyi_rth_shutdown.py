"""
PyInstaller runtime hook: prevent SIGSEGV during Py_FinalizeEx on macOS.
The conda Python + PyInstaller combination can crash during module cleanup
because C extension modules are unloaded before their Python wrapper objects
are deallocated, causing NULL pointer dereferences.

Since this is a GUI desktop app, skipping Python finalization is safe — the
OS reclaims all resources on process exit.  We flush all logging handlers
first so no diagnostic output is lost.
"""
import atexit
import logging
import os
import sys

if sys.platform == "darwin":
    def _flush_then_exit() -> None:
        for handler in list(logging.getLogger().handlers):
            try:
                handler.flush()
                handler.close()
            except Exception:
                pass
        os._exit(0)

    atexit.register(_flush_then_exit)

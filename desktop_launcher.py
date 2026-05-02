#!/usr/bin/env python
"""
TransLive desktop shell.

Starts the FastAPI backend on localhost, stores downloaded models in the
user's Application Support directory, and opens the existing web UI in a native
WebView window.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import quote

APP_NAME = "TransLive"
APP_VERSION = "0.88 (test)"
APP_CREDIT = "薛定谔的帮你偶"
DEFAULT_PORT = 8766
MT_MODEL_FILENAME = "HY-MT1.5-1.8B-Q4_K_M.gguf"
logger = logging.getLogger(__name__)


def _resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent


def _app_support_dir() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / APP_NAME
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / APP_NAME
    return home / ".local" / "share" / APP_NAME


def _find_free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free localhost port found near {preferred}")


def _prime_environment(port: int) -> None:
    from app.core.expiry import apply_build_config_environment

    apply_build_config_environment()
    support_dir = _app_support_dir()
    model_dir = support_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = os.environ.get("TRANS_DESKTOP_MT_MODEL_ID", str(model_dir / MT_MODEL_FILENAME))

    os.environ["TRANS_HOST"] = "127.0.0.1"
    os.environ["TRANS_PORT"] = str(port)
    os.environ["TRANS_DESKTOP_MODE"] = "true"
    os.environ.setdefault("TRANS_APP_VERSION", APP_VERSION)
    os.environ.setdefault("TRANS_APP_CREDIT", APP_CREDIT)
    # Packaged app mode should never download a 1GB+ model silently during
    # startup. The web UI prompts the user and calls /api/models/download.
    os.environ["TRANS_AUTO_DOWNLOAD_MODELS"] = "false"
    os.environ["TRANS_LOAD_MODELS_ON_STARTUP"] = "false"
    os.environ.setdefault("TRANS_ASR_BACKEND", "ct2")
    os.environ.setdefault("TRANS_ASR_MODEL_ID", "large-v3-turbo")
    os.environ.setdefault("TRANS_ASR_DEVICE", "cpu")
    os.environ.setdefault("TRANS_ASR_COMPUTE_TYPE", "int8")
    os.environ.setdefault("TRANS_ASR_BEAM_SIZE", "1")
    os.environ.setdefault("TRANS_ASR_BEST_OF", "1")
    # Desktop builds store downloaded models outside the .app bundle so updates
    # do not delete them and Gatekeeper does not block writes into app resources.
    os.environ["TRANS_MT_MODEL_ID"] = model_path
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("NNCF_PROGRESS_BAR", "false")
    os.environ.setdefault("RICH_FORCE_TERMINAL", "false")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _show_expired_message(message: str) -> None:
    print(message, file=sys.stderr)
    if sys.platform != "darwin":
        return
    script = (
        'display dialog "'
        + message.replace("\\", "\\\\").replace('"', '\\"')
        + '" with title "TransLive" buttons {"好"} default button "好"'
    )
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=12)
    except Exception:
        pass


def _wait_for_health(url: str, timeout_s: float = 90.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except (OSError, urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.25)
    return False


def _begin_realtime_activity() -> Any | None:
    if sys.platform != "darwin":
        return None
    try:
        from Foundation import (
            NSActivityLatencyCritical,
            NSActivityUserInitiatedAllowingIdleSystemSleep,
            NSProcessInfo,
        )

        options = NSActivityLatencyCritical | NSActivityUserInitiatedAllowingIdleSystemSleep
        return NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
            options,
            "TransLive realtime audio translation",
        )
    except Exception:
        return None


def _end_realtime_activity(activity: Any | None) -> None:
    if activity is None or sys.platform != "darwin":
        return
    try:
        from Foundation import NSProcessInfo

        NSProcessInfo.processInfo().endActivity_(activity)
    except Exception:
        pass


def _flush_logs() -> None:
    for handler in list(logging.getLogger().handlers):
        try:
            handler.flush()
        except Exception:
            pass


def _schedule_hard_exit(code: int = 0, delay_s: float = 12.0) -> threading.Timer | None:
    if not (getattr(sys, "frozen", False) and sys.platform == "darwin"):
        return None

    def _exit_if_cleanup_stalls() -> None:
        _flush_logs()
        os._exit(code)

    timer = threading.Timer(delay_s, _exit_if_cleanup_stalls)
    timer.daemon = True
    timer.start()
    return timer


class _ServerThread:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server: Any | None = None
        self.thread = threading.Thread(target=self._run, name="TransLiveServer", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        self.thread.join(timeout=5)
        if self.thread.is_alive() and self.server is not None:
            self.server.force_exit = True
            self.thread.join(timeout=2)

    def _run(self) -> None:
        import uvicorn
        from app.core.config import settings

        class DesktopServer(uvicorn.Server):
            def install_signal_handlers(self) -> None:
                return

        config = uvicorn.Config(
            "app.main:app",
            host=self.host,
            port=self.port,
            log_level="info",
            reload=False,
            ws_max_size=settings.websocket_max_message_bytes,
        )
        self.server = DesktopServer(config)
        self.server.run()


def _install_shutdown_handlers(server: _ServerThread) -> tuple[threading.Event, Any]:
    stop_requested = threading.Event()
    previous_handlers: dict[int, Any] = {}

    def _request_shutdown(signum: int, _frame: Any) -> None:
        stop_requested.set()
        if server.server is not None:
            server.server.should_exit = True
        raise KeyboardInterrupt

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        previous_handlers[sig] = signal.getsignal(sig)
        try:
            signal.signal(sig, _request_shutdown)
        except (ValueError, OSError):
            previous_handlers.pop(sig, None)

    def _restore() -> None:
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    return stop_requested, _restore


def _wait_for_server_exit(server: _ServerThread, stop_requested: threading.Event) -> None:
    while server.thread.is_alive() and not stop_requested.is_set():
        time.sleep(1)


class DesktopApi:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._overlay_window: Any | None = None
        self._lock = threading.Lock()
        self._native_audio: Any | None = None

    def _native_audio_bridge(self):
        if self._native_audio is None:
            from app.desktop.native_audio import NativeSystemAudioBridge

            self._native_audio = NativeSystemAudioBridge(self.base_url)
        return self._native_audio

    def native_audio_available(self) -> dict[str, Any]:
        return self._native_audio_bridge().available()

    def native_audio_status(self) -> dict[str, Any]:
        return self._native_audio_bridge().status()

    def start_system_audio_capture(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._native_audio_bridge().start(config or {})

    def update_system_audio_config(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._native_audio_bridge().update_config(config or {})

    def stop_system_audio_capture(self) -> dict[str, Any]:
        return self._native_audio_bridge().stop()

    def open_subtitle_window(self, mode: str = "bilingual") -> dict[str, Any]:
        import webview

        subtitle_mode = mode if mode in {"bilingual", "translation"} else "bilingual"
        overlay_url = f"{self.base_url}/overlay.html?mode={quote(subtitle_mode)}"

        with self._lock:
            if self._overlay_window is not None:
                try:
                    self._overlay_window.load_url(overlay_url)
                    self._overlay_window.on_top = True
                    self._overlay_window.show()
                    return {"ok": True, "reused": True}
                except Exception:
                    self._overlay_window = None

            window = webview.create_window(
                "TransLive 字幕",
                overlay_url,
                js_api=self,
                width=960,
                height=260,
                min_size=(420, 140),
                resizable=True,
                frameless=True,
                easy_drag=True,
                shadow=False,
                focus=False,
                on_top=True,
                transparent=True,
                background_color="#000000",
            )
            self._overlay_window = window
            if window is not None:
                window.events.closed += self._overlay_closed
            return {"ok": window is not None, "reused": False}

    def close_subtitle_window(self) -> dict[str, Any]:
        with self._lock:
            window = self._overlay_window
            self._overlay_window = None
        if window is not None:
            try:
                window.destroy()
            except Exception:
                pass
        return {"ok": True}

    def subtitle_window_state(self) -> dict[str, Any]:
        with self._lock:
            window = self._overlay_window
        if window is None:
            return {"ok": False}
        return {
            "ok": True,
            "width": int(getattr(window, "width", 0) or 0),
            "height": int(getattr(window, "height", 0) or 0),
            "x": int(getattr(window, "x", 0) or 0),
            "y": int(getattr(window, "y", 0) or 0),
        }

    def resize_subtitle_window(self, width: int, height: int) -> dict[str, Any]:
        with self._lock:
            window = self._overlay_window
        if window is None:
            return {"ok": False, "error": "subtitle_window_not_open"}

        safe_width = max(420, min(int(width), 1800))
        safe_height = max(140, min(int(height), 900))
        try:
            window.resize(safe_width, safe_height)
        except Exception:
            try:
                window.set_window_size(safe_width, safe_height)
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        return {"ok": True, "width": safe_width, "height": safe_height}

    def _overlay_closed(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            self._overlay_window = None

    def shutdown(self) -> None:
        try:
            self.stop_system_audio_capture()
        except Exception:
            logger.exception("Failed to stop native audio during shutdown")

        with self._lock:
            overlay = self._overlay_window
            self._overlay_window = None
        if overlay is not None:
            try:
                overlay.destroy()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch TransLive as a desktop app")
    parser.add_argument("--no-window", action="store_true", help="start backend only, for smoke tests")
    parser.add_argument("--browser", action="store_true", help="open the system browser instead of WebView")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="preferred localhost port")
    args = parser.parse_args()

    root = _resource_root()
    os.chdir(root)
    sys.path.insert(0, str(root))

    realtime_activity = _begin_realtime_activity()
    from app.core.expiry import apply_build_config_environment, expiry_status

    apply_build_config_environment()
    app_expiry = expiry_status()
    if app_expiry.expired:
        _show_expired_message(app_expiry.message or f"{APP_NAME} 测试版已到期。")
        _end_realtime_activity(realtime_activity)
        return 75

    port = _find_free_port(args.port)
    _prime_environment(port)

    host = "127.0.0.1"
    url = f"http://{host}:{port}/"
    health_url = f"http://{host}:{port}/api/health"

    server = _ServerThread(host, port)
    server.start()
    stop_requested, restore_shutdown_handlers = _install_shutdown_handlers(server)
    desktop_api: DesktopApi | None = None
    hard_exit_timer: threading.Timer | None = None

    try:
        if not _wait_for_health(health_url):
            raise RuntimeError("TransLive backend did not become ready in time")

        if args.no_window:
            print(url)
            try:
                _wait_for_server_exit(server, stop_requested)
            except KeyboardInterrupt:
                pass
            return 0

        if args.browser:
            webbrowser.open(url)
            try:
                _wait_for_server_exit(server, stop_requested)
            except KeyboardInterrupt:
                pass
            return 0

        try:
            import webview
        except ImportError:
            webbrowser.open(url)
            try:
                _wait_for_server_exit(server, stop_requested)
            except KeyboardInterrupt:
                pass
            return 0

        desktop_api = DesktopApi(url)
        window = webview.create_window(
            f"{APP_NAME} {APP_VERSION}",
            url,
            js_api=desktop_api,
            width=1180,
            height=820,
            min_size=(920, 640),
        )
        if window is not None:
            def _main_window_closed(*_args: Any, **_kwargs: Any) -> None:
                stop_requested.set()
                if server.server is not None:
                    server.server.should_exit = True

            try:
                window.events.closed += _main_window_closed
            except Exception:
                pass
        try:
            storage_path = str(_app_support_dir() / "webview")
            webview.start(private_mode=False, storage_path=storage_path)
        finally:
            # Keep a reference so pywebview does not optimize the variable away in
            # frozen builds before shutdown callbacks complete.
            _ = window
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        hard_exit_timer = _schedule_hard_exit(0)
        if desktop_api is not None:
            desktop_api.shutdown()
        server.stop()
        restore_shutdown_handlers()
        _end_realtime_activity(realtime_activity)
        if hard_exit_timer is not None:
            hard_exit_timer.cancel()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    exit_code = main()
    if getattr(sys, "frozen", False) and sys.platform == "darwin":
        _flush_logs()
        os._exit(exit_code)
    raise SystemExit(exit_code)

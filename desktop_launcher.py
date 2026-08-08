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
import json
import logging
import os
import platform
import secrets
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

from app.version import APP_CREDIT, APP_NAME, APP_VERSION

DEFAULT_PORT = 8766
MT_MODEL_FILENAME = "Hy-MT2-1.8B-Q4_K_M.gguf"
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


def _is_packaged_apple_silicon() -> bool:
    return (
        bool(getattr(sys, "frozen", False))
        and sys.platform == "darwin"
        and platform.machine() == "arm64"
    )


def _reserve_local_port(preferred: int) -> tuple[socket.socket, int]:
    for port in range(preferred, preferred + 50):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            sock.listen(2048)
            sock.setblocking(False)
            return sock, port
        except OSError:
            sock.close()
    raise RuntimeError(f"No free localhost port found near {preferred}")


def _prime_environment(port: int, api_token: str, instance_id: str) -> None:
    from app.core.expiry import apply_build_config_environment

    apply_build_config_environment()
    support_dir = _app_support_dir()
    model_dir = support_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = os.environ.get("TRANS_DESKTOP_MT_MODEL_ID", str(model_dir / MT_MODEL_FILENAME))

    os.environ["TRANS_HOST"] = "127.0.0.1"
    os.environ["TRANS_PORT"] = str(port)
    os.environ["TRANS_DESKTOP_MODE"] = "true"
    os.environ["TRANS_LOCAL_API_TOKEN"] = api_token
    os.environ["TRANS_INSTANCE_ID"] = instance_id
    os.environ.setdefault("TRANS_APP_VERSION", APP_VERSION)
    os.environ.setdefault("TRANS_APP_CREDIT", APP_CREDIT)
    # Packaged app mode should never download a 1GB+ model silently during
    # startup. The web UI prompts the user and calls /api/models/download.
    os.environ["TRANS_AUTO_DOWNLOAD_MODELS"] = "false"
    os.environ["TRANS_LOAD_MODELS_ON_STARTUP"] = "false"
    os.environ.setdefault("TRANS_MODEL_PROFILE", "auto")
    if _is_packaged_apple_silicon():
        # The arm64 distribution intentionally ships one ASR path. Do not
        # silently change model families when a Qwen cache is incomplete.
        os.environ["TRANS_ASR_BACKEND"] = "qwen3"
        os.environ["TRANS_ASR_DEVICE"] = "mlx"
        os.environ["TRANS_ASR_COMPUTE_TYPE"] = "8bit"
        os.environ["TRANS_ASR_FALLBACKS_ENABLED"] = "false"
    else:
        os.environ.setdefault("TRANS_ASR_BACKEND", "auto")
        os.environ.setdefault("TRANS_ASR_DEVICE", "auto")
        os.environ.setdefault("TRANS_ASR_COMPUTE_TYPE", "default")
        os.environ.setdefault("TRANS_ASR_FALLBACKS_ENABLED", "true")
    os.environ.setdefault("TRANS_ASR_MAX_NEW_TOKENS", "96")
    os.environ.setdefault("TRANS_ASR_BEAM_SIZE", "1")
    os.environ.setdefault("TRANS_ASR_BEST_OF", "1")
    os.environ.setdefault("TRANS_ASR_TEMPERATURE", "[0.0]")
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


def _wait_for_ready(url: str, expected_instance_id: str, timeout_s: float = 90.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                if (
                    resp.status == 200
                    and payload.get("status") == "ok"
                    and secrets.compare_digest(
                        str(payload.get("instance_id") or ""),
                        expected_instance_id,
                    )
                ):
                    return True
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(0.25)
    return False


def _flush_logs() -> None:
    for handler in list(logging.getLogger().handlers):
        try:
            handler.flush()
        except Exception:
            pass


def _release_backend_resources() -> None:
    """Best-effort cleanup for ML runtimes before the desktop shell exits."""
    try:
        from app.mt.engine import mt_engine

        if not mt_engine.close(wait_s=8.0):
            logger.warning("MT model is still in use; leaving it intact for process teardown")
    except Exception:
        logger.exception("Failed to close MT engine during desktop shutdown")

    try:
        from app.asr.engine import unload_asr_engine

        unload_asr_engine()
    except Exception:
        logger.exception("Failed to unload ASR engine during desktop shutdown")

    from app.core.memory import collect_model_memory

    collect_model_memory()


def _schedule_hard_exit(code: int = 0, delay_s: float = 12.0) -> threading.Timer | None:
    if not (getattr(sys, "frozen", False) and sys.platform == "darwin"):
        return None
    os.environ["TRANS_EXIT_CODE"] = str(code)

    def _exit_if_cleanup_stalls() -> None:
        _flush_logs()
        os._exit(code)

    timer = threading.Timer(delay_s, _exit_if_cleanup_stalls)
    timer.daemon = True
    timer.start()
    return timer


class _ServerThread:
    def __init__(self, host: str, port: int, reserved_socket: socket.socket):
        self.host = host
        self.port = port
        self.server: Any | None = None
        self.reserved_socket = reserved_socket
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
        try:
            self.server.run(sockets=[self.reserved_socket])
        finally:
            with contextlib.suppress(OSError):
                self.reserved_socket.close()


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


_ACTIVATION_OBSERVER_CLASS = None


def _activation_observer_class():
    """
    PyObjC 的类只能注册一次，所以这里和 native_audio 一样用模块级缓存，
    不能在方法里直接 class 定义（第二次调用会因为重复注册报错）。
    """
    global _ACTIVATION_OBSERVER_CLASS
    if _ACTIVATION_OBSERVER_CLASS is not None:
        return _ACTIVATION_OBSERVER_CLASS

    import objc
    from Foundation import NSObject

    class TransLiveActivationObserver(NSObject):
        def initWithCallback_(self, callback):
            self = objc.super(TransLiveActivationObserver, self).init()
            if self is None:
                return None
            self.callback = callback
            return self

        def applicationDidBecomeActive_(self, notification):
            try:
                self.callback()
            except Exception:
                logger.exception("Activation handler failed")

    _ACTIVATION_OBSERVER_CLASS = TransLiveActivationObserver
    return _ACTIVATION_OBSERVER_CLASS


class DesktopJsApi:
    """Small authenticated facade; internal lifecycle methods stay unexposed."""

    def __init__(self, desktop_api: "DesktopApi", api_token: str):
        self._desktop_api = desktop_api
        self._api_token = api_token

    def _call(self, token: str, method: str, *args: Any) -> Any:
        if not token or not secrets.compare_digest(str(token), self._api_token):
            return {"ok": False, "error": "unauthorized"}
        return getattr(self._desktop_api, method)(*args)

    def native_audio_available(self, token: str) -> Any:
        return self._call(token, "native_audio_available")

    def native_audio_diagnostics(self, token: str) -> Any:
        return self._call(token, "native_audio_diagnostics")

    def native_audio_status(self, token: str) -> Any:
        return self._call(token, "native_audio_status")

    def start_system_audio_capture(self, token: str, config: dict[str, Any] | None = None) -> Any:
        return self._call(token, "start_system_audio_capture", config)

    def update_system_audio_config(self, token: str, config: dict[str, Any] | None = None) -> Any:
        return self._call(token, "update_system_audio_config", config)

    def stop_system_audio_capture(self, token: str) -> Any:
        return self._call(token, "stop_system_audio_capture")

    def open_subtitle_window(self, token: str, mode: str = "bilingual", state: dict[str, Any] | None = None) -> Any:
        return self._call(token, "open_subtitle_window", mode, state)

    def show_main_window(self, token: str) -> Any:
        return self._call(token, "show_main_window")

    def ensure_subtitle_window_visible(self, token: str) -> Any:
        return self._call(token, "ensure_subtitle_window_visible")

    def set_subtitle_interactive_zones(self, token: str, zones: Any = None) -> Any:
        return self._call(token, "set_subtitle_interactive_zones", zones)

    def set_subtitle_click_through(self, token: str, enabled: bool = True) -> Any:
        return self._call(token, "set_subtitle_click_through", enabled)

    def subtitle_click_through(self, token: str) -> Any:
        return self._call(token, "subtitle_click_through")

    def close_subtitle_window(self, token: str) -> Any:
        return self._call(token, "close_subtitle_window")

    def subtitle_window_state(self, token: str) -> Any:
        return self._call(token, "subtitle_window_state")

    def resize_subtitle_window(self, token: str, width: int, height: int) -> Any:
        return self._call(token, "resize_subtitle_window", width, height)

    def move_subtitle_window(self, token: str, x: int, y: int) -> Any:
        return self._call(token, "move_subtitle_window", x, y)


class DesktopApi:
    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.js_api = DesktopJsApi(self, api_token)
        self._overlay_window: Any | None = None
        # 字幕窗复用路径会在持锁时进入 _configure_overlay_window()，后者通过
        # _overlay_ns_window() 再读取同一状态；这里必须允许同线程重入。
        self._lock = threading.RLock()
        self._native_audio: Any | None = None
        self._shutdown_started = False
        self._click_through = False
        # 前端上报的「不穿透」区域（工具栏、缩放手柄），CSS px、窗口左上角原点
        self._interactive_zones: list[tuple[float, float, float, float]] = []
        self._ignoring_mouse: bool | None = None
        self._mouse_monitors: list[Any] = []
        self._main_window: Any | None = None
        self._activation_observer: Any | None = None

    def _native_audio_bridge(self):
        if self._native_audio is None:
            from app.desktop.native_audio import NativeSystemAudioBridge

            self._native_audio = NativeSystemAudioBridge(self.base_url, self.api_token)
        return self._native_audio

    def native_audio_available(self) -> dict[str, Any]:
        return self._native_audio_bridge().available()

    def native_audio_diagnostics(self) -> dict[str, Any]:
        from app.desktop.native_audio import native_system_audio_diagnostics

        return native_system_audio_diagnostics()

    def native_audio_status(self) -> dict[str, Any]:
        return self._native_audio_bridge().status()

    def start_system_audio_capture(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._native_audio_bridge().start(config or {})

    def update_system_audio_config(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._native_audio_bridge().update_config(config or {})

    def stop_system_audio_capture(self, graceful: bool = True) -> dict[str, Any]:
        if self._native_audio is None:
            return {"ok": True, "running": False}
        return self._native_audio.stop(graceful=graceful)

    def open_subtitle_window(self, mode: str = "bilingual", state: dict[str, Any] | None = None) -> dict[str, Any]:
        import webview

        subtitle_mode = mode if mode in {"bilingual", "translation"} else "bilingual"
        overlay_url = (
            f"{self.base_url}/overlay.html?mode={quote(subtitle_mode)}"
            f"#token={quote(self.api_token, safe='')}"
        )
        state = state or {}
        width = self._safe_window_dimension(state.get("width"), 960, 420, 1800)
        height = self._safe_window_dimension(state.get("height"), 260, 140, 900)
        x = self._safe_window_coordinate(state.get("x"))
        y = self._safe_window_coordinate(state.get("y"))

        with self._lock:
            if self._overlay_window is not None:
                try:
                    self._overlay_window.load_url(overlay_url)
                    self._resize_window(self._overlay_window, width, height)
                    if x is not None and y is not None:
                        self._move_window(self._overlay_window, x, y)
                    self._overlay_window.on_top = True
                    self._overlay_window.show()
                    self._configure_overlay_window()
                    return {"ok": True, "reused": True, "width": width, "height": height, "x": x, "y": y}
                except Exception:
                    self._overlay_window = None

            window = webview.create_window(
                "TransLive 字幕",
                overlay_url,
                js_api=self.js_api,
                width=width,
                height=height,
                x=x,
                y=y,
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
                # 窗口是异步创建的（AppHelper.callAfter），所以等 shown 再配置原生行为
                try:
                    window.events.shown += lambda *_a, **_k: self._configure_overlay_window()
                except Exception:
                    self._configure_overlay_window()
            return {"ok": window is not None, "reused": False, "width": width, "height": height, "x": x, "y": y}

    def attach_main_window(self, window: Any) -> None:
        """
        记录主窗口，并监听 App 激活事件。

        macOS 的默认行为：App 只要还有可见窗口（字幕悬浮窗就算），点 Dock 图标
        时 AppKit 就不会去恢复已最小化的窗口 —— pywebview 也没有实现
        applicationShouldHandleReopen:hasVisibleWindows:。结果就是开着悬浮窗
        最小化主窗口之后，主窗口再也叫不回来。

        点 Dock 图标至少会让 App 变为 active，所以这里监听激活通知，
        顺手把最小化的主窗口恢复出来。
        """
        self._main_window = window
        if sys.platform != "darwin":
            return
        try:
            import AppKit
            from Foundation import NSNotificationCenter

            observer_cls = _activation_observer_class()
            self._activation_observer = observer_cls.alloc().initWithCallback_(
                self._restore_main_window_if_minimized
            )
            NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
                self._activation_observer,
                "applicationDidBecomeActive:",
                AppKit.NSApplicationDidBecomeActiveNotification,
                None,
            )
        except Exception:
            logger.exception("Could not observe app activation; use the overlay button to restore")

    def _main_ns_window(self):
        window = self._main_window
        if window is None or sys.platform != "darwin":
            return None
        try:
            from webview.platforms.cocoa import BrowserView

            instance = BrowserView.instances.get(window.uid)
            return instance.window if instance is not None else None
        except Exception:
            return None

    def _restore_main_window_if_minimized(self) -> None:
        ns_window = self._main_ns_window()
        if ns_window is None:
            return
        try:
            if ns_window.isMiniaturized():
                ns_window.deminiaturize_(None)
                logger.info("Restored the minimized main window on app activation")
        except Exception:
            logger.exception("Failed to restore the main window")

    def show_main_window(self) -> dict[str, Any]:
        """从字幕窗把主窗口叫回来（最小化之后 Dock 图标点不出来时的出路）。"""
        if self._main_window is None:
            return {"ok": False, "error": "main_window_unavailable"}
        try:
            from PyObjCTools import AppHelper

            def _apply() -> None:
                ns_window = self._main_ns_window()
                try:
                    if ns_window is not None:
                        if ns_window.isMiniaturized():
                            ns_window.deminiaturize_(None)
                        ns_window.makeKeyAndOrderFront_(None)
                    import AppKit

                    AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
                except Exception:
                    logger.exception("Failed to bring the main window forward")

            if sys.platform == "darwin":
                AppHelper.callAfter(_apply)
            else:
                self._main_window.restore()
                self._main_window.show()
        except Exception as exc:
            logger.exception("show_main_window failed")
            return {"ok": False, "error": str(exc)}
        return {"ok": True}

    def _overlay_ns_window(self):
        """拿到字幕窗底层的 NSWindow。pywebview 没有暴露这些能力，只能自己取。"""
        with self._lock:
            window = self._overlay_window
        if window is None or sys.platform != "darwin":
            return None
        try:
            from webview.platforms.cocoa import BrowserView

            instance = BrowserView.instances.get(window.uid)
            return instance.window if instance is not None else None
        except Exception:
            logger.exception("Could not reach the native subtitle window")
            return None

    def _configure_overlay_window(self) -> None:
        """
        字幕窗需要两个 pywebview 不提供的原生行为：

        1) 跨 Space + 覆盖全屏应用。pywebview 给窗口设的是
           NSWindowCollectionBehaviorManaged(4)，意味着窗口只属于一个 Space ——
           用户把视频 / Zoom / Keynote 切到全屏（会新建一个 Space）时，字幕窗
           会留在原来的桌面上直接看不见了。而全屏看视频恰恰是同传字幕最主要的
           使用场景。改成 CanJoinAllSpaces|FullScreenAuxiliary|Stationary(273)。

        2) 开窗后把窗口拉回可见区域，见 ensure_visible()。
        """
        if sys.platform != "darwin":
            return
        try:
            import AppKit
            from PyObjCTools import AppHelper
        except Exception:
            return

        def _apply() -> None:
            ns_window = self._overlay_ns_window()
            if ns_window is None:
                return
            try:
                ns_window.setCollectionBehavior_(
                    AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                    | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
                    | AppKit.NSWindowCollectionBehaviorStationary
                )
            except Exception:
                logger.exception("Failed to set subtitle window collection behavior")
            self._clamp_overlay_into_view(ns_window)

        AppHelper.callAfter(_apply)

    @staticmethod
    def _clamp_overlay_into_view(ns_window) -> bool:
        """
        把窗口拉回某块屏幕的可见区域。

        两个真实故障都靠它兜底：
        - pywebview 的 mouseDragged_ 钳制算错了方向：窗口顶部一旦越过屏幕上沿，
          它会把 y 设成 screenTop + windowHeight，也就是整个窗口飞到屏幕外。
          字幕窗是无边框置顶的，飞出去就再也抓不回来。
        - 上次的位置存在 localStorage 里；如果那是一台已经拔掉的外接显示器，
          下次打开字幕窗就直接开在看不见的地方，表现为「字幕窗打不开」。
        """
        try:
            import AppKit

            frame = ns_window.frame()
            screens = list(AppKit.NSScreen.screens())
            if not screens:
                return False

            def overlap(visible) -> float:
                dx = max(0.0, min(frame.origin.x + frame.size.width, visible.origin.x + visible.size.width)
                         - max(frame.origin.x, visible.origin.x))
                dy = max(0.0, min(frame.origin.y + frame.size.height, visible.origin.y + visible.size.height)
                         - max(frame.origin.y, visible.origin.y))
                return dx * dy

            target = max((s.visibleFrame() for s in screens), key=overlap)
            area = max(1.0, frame.size.width * frame.size.height)
            if overlap(target) / area >= 0.6:
                return False  # 已经基本可见，别打扰用户摆好的位置

            x = min(max(frame.origin.x, target.origin.x),
                    max(target.origin.x, target.origin.x + target.size.width - frame.size.width))
            y = min(max(frame.origin.y, target.origin.y),
                    max(target.origin.y, target.origin.y + target.size.height - frame.size.height))
            ns_window.setFrameOrigin_(AppKit.NSPoint(x, y))
            logger.info("Pulled the subtitle window back on screen: (%.0f, %.0f)", x, y)
            return True
        except Exception:
            logger.exception("Failed to clamp the subtitle window into view")
            return False

    def ensure_subtitle_window_visible(self) -> dict[str, Any]:
        """拖拽结束后由前端调用一次，保证窗口没被拖丢。"""
        if sys.platform != "darwin":
            return {"ok": False, "error": "macos_only"}
        ns_window = self._overlay_ns_window()
        if ns_window is None:
            return {"ok": False, "error": "subtitle_window_not_open"}
        try:
            from PyObjCTools import AppHelper

            AppHelper.callAfter(lambda: self._clamp_overlay_into_view(ns_window))
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True}

    def set_subtitle_interactive_zones(self, zones: Any = None) -> dict[str, Any]:
        """
        前端上报「不该被穿透」的区域（工具栏、缩放手柄），坐标是 CSS px、
        以窗口左上角为原点。窗口无边框，所以 contentRect == frame，换算简单。
        """
        parsed: list[tuple[float, float, float, float]] = []
        for zone in zones or []:
            try:
                x = float(zone["x"])
                y = float(zone["y"])
                w = float(zone["w"])
                h = float(zone["h"])
            except (TypeError, KeyError, ValueError):
                continue
            if w > 0 and h > 0:
                parsed.append((x, y, w, h))
        self._interactive_zones = parsed
        if self._click_through:
            self._refresh_ignore_state()
        return {"ok": True, "zones": len(parsed)}

    def _point_is_over_chrome(self, ns_window, point: Any = None) -> bool:
        """
        光标是否落在工具栏/缩放手柄上。

        point 是屏幕坐标（AppKit 约定：左下角原点、y 向上）；不传就取当前鼠标位置。
        interactive_zones 来自前端，是 CSS 坐标（左上角原点、y 向下），所以要翻转 y。
        """
        try:
            import AppKit

            if point is None:
                point = AppKit.NSEvent.mouseLocation()
            frame = ns_window.frame()
            for x, y, w, h in self._interactive_zones:
                left = frame.origin.x + x
                bottom = frame.origin.y + frame.size.height - (y + h)
                if left <= point.x <= left + w and bottom <= point.y <= bottom + h:
                    return True
        except Exception:
            logger.exception("Failed to hit-test the subtitle window chrome")
        return False

    def _refresh_ignore_state(self) -> None:
        """按光标位置切换整窗的鼠标忽略状态。必须在主线程调用。"""
        ns_window = self._overlay_ns_window()
        if ns_window is None:
            return
        want_ignore = self._click_through and not self._point_is_over_chrome(ns_window)
        if want_ignore == self._ignoring_mouse:
            return  # 状态没变就别碰窗口，鼠标移动事件很密集
        self._ignoring_mouse = want_ignore
        try:
            ns_window.setIgnoresMouseEvents_(want_ignore)
        except Exception:
            logger.exception("Failed to update click-through state")

    def _install_mouse_monitors(self) -> None:
        if self._mouse_monitors or sys.platform != "darwin":
            return
        try:
            import AppKit

            mask = AppKit.NSEventMaskMouseMoved

            def _on_global(event):
                self._refresh_ignore_state()

            def _on_local(event):
                self._refresh_ignore_state()
                return event

            # 窗口被设成忽略鼠标之后就收不到任何事件了，没法自己发现「光标移回工具栏」，
            # 所以必须靠全局监听。本应用在前台时全局监听不触发，需要再配一个 local。
            self._mouse_monitors = [
                AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, _on_global),
                AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(mask, _on_local),
            ]
        except Exception:
            logger.exception("Failed to install mouse monitors for click-through")
            self._mouse_monitors = []

    def _remove_mouse_monitors(self) -> None:
        monitors, self._mouse_monitors = self._mouse_monitors, []
        if not monitors:
            return
        try:
            import AppKit

            for monitor in monitors:
                if monitor is not None:
                    AppKit.NSEvent.removeMonitor_(monitor)
        except Exception:
            logger.exception("Failed to remove mouse monitors")

    def set_subtitle_click_through(self, enabled: bool = True) -> dict[str, Any]:
        """
        让字幕正文区域的点击穿透到下面的应用，**工具栏和缩放手柄仍然可点**。

        NSWindow 的 ignoresMouseEvents 是整窗开关，没有按区域生效的 API。
        所以这里装一个鼠标移动监听，光标在工具栏/手柄上时把忽略关掉，
        移开再打开 —— 效果上就是「只有底下的字幕区域穿透」。
        监听只在开启穿透期间存在，关掉就移除，不常驻消耗。
        """
        if sys.platform != "darwin":
            return {"ok": False, "error": "click_through_requires_macos"}
        if self._overlay_ns_window() is None:
            return {"ok": False, "error": "subtitle_window_not_open"}

        want = bool(enabled)
        try:
            from PyObjCTools import AppHelper

            def _apply() -> None:
                self._click_through = want
                if want:
                    self._install_mouse_monitors()
                else:
                    self._remove_mouse_monitors()
                    self._ignoring_mouse = None  # 强制下面这次真正写回窗口
                self._refresh_ignore_state()

            AppHelper.callAfter(_apply)
        except Exception as exc:
            logger.exception("Click-through toggle failed")
            return {"ok": False, "error": str(exc)}

        self._click_through = want
        return {"ok": True, "click_through": want}

    def subtitle_click_through(self) -> dict[str, Any]:
        return {"ok": True, "click_through": self._click_through}

    def close_subtitle_window(self) -> dict[str, Any]:
        with self._lock:
            window = self._overlay_window
            self._overlay_window = None
        self._click_through = False
        self._ignoring_mouse = None
        self._interactive_zones = []
        self._remove_mouse_monitors()
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

    @staticmethod
    def _safe_window_dimension(value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(parsed, maximum))

    @staticmethod
    def _safe_window_coordinate(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return max(-5000, min(parsed, 10000))

    @staticmethod
    def _resize_window(window: Any, width: int, height: int) -> None:
        try:
            window.resize(width, height)
        except Exception:
            window.set_window_size(width, height)

    @staticmethod
    def _move_window(window: Any, x: int, y: int) -> None:
        try:
            window.move(x, y)
        except Exception:
            window.set_window_position(x, y)

    def resize_subtitle_window(self, width: int, height: int) -> dict[str, Any]:
        with self._lock:
            window = self._overlay_window
        if window is None:
            return {"ok": False, "error": "subtitle_window_not_open"}

        safe_width = self._safe_window_dimension(width, 960, 420, 1800)
        safe_height = self._safe_window_dimension(height, 260, 140, 900)
        try:
            self._resize_window(window, safe_width, safe_height)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        return {"ok": True, "width": safe_width, "height": safe_height}

    def move_subtitle_window(self, x: int, y: int) -> dict[str, Any]:
        with self._lock:
            window = self._overlay_window
        if window is None:
            return {"ok": False, "error": "subtitle_window_not_open"}

        safe_x = self._safe_window_coordinate(x)
        safe_y = self._safe_window_coordinate(y)
        if safe_x is None or safe_y is None:
            return {"ok": False, "error": "invalid_position"}

        try:
            self._move_window(window, safe_x, safe_y)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        return {"ok": True, "x": safe_x, "y": safe_y}

    def _overlay_closed(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            self._overlay_window = None
        # 下次打开的字幕窗默认可以点击，不要继承上一个窗口的穿透状态
        self._click_through = False
        self._ignoring_mouse = None
        self._interactive_zones = []
        self._remove_mouse_monitors()

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True
        try:
            self.stop_system_audio_capture(graceful=False)
        except Exception:
            logger.exception("Failed to stop native audio during shutdown")

        self._remove_mouse_monitors()

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

    # 注意：macOS 实时活动声明（NSActivityLatencyCritical）已移到
    # app.core.power，由 StreamSession 在同传开始/结束时按会话申请和释放。
    # 以前在这里整个进程生命周期都持有，等于空闲时也阻止系统降频。
    from app.core.expiry import apply_build_config_environment, expiry_status

    apply_build_config_environment()
    app_expiry = expiry_status()
    if app_expiry.expired:
        _show_expired_message(app_expiry.message or f"{APP_NAME} 测试版已到期。")
        return 75

    reserved_socket, port = _reserve_local_port(args.port)
    api_token = secrets.token_urlsafe(32)
    instance_id = secrets.token_hex(16)
    try:
        _prime_environment(port, api_token, instance_id)
    except Exception:
        reserved_socket.close()
        raise

    host = "127.0.0.1"
    base_url = f"http://{host}:{port}"
    launch_url = f"{base_url}/#token={quote(api_token, safe='')}"
    ready_url = f"{base_url}/api/ready"

    server = _ServerThread(host, port, reserved_socket)
    server.start()
    stop_requested, restore_shutdown_handlers = _install_shutdown_handlers(server)
    desktop_api: DesktopApi | None = None
    hard_exit_timer: threading.Timer | None = None

    try:
        if not _wait_for_ready(ready_url, instance_id):
            raise RuntimeError("TransLive backend did not become ready in time")

        if args.no_window:
            print(launch_url)
            try:
                _wait_for_server_exit(server, stop_requested)
            except KeyboardInterrupt:
                pass
            return 0

        if args.browser:
            webbrowser.open(launch_url)
            try:
                _wait_for_server_exit(server, stop_requested)
            except KeyboardInterrupt:
                pass
            return 0

        try:
            import webview
        except ImportError:
            webbrowser.open(launch_url)
            try:
                _wait_for_server_exit(server, stop_requested)
            except KeyboardInterrupt:
                pass
            return 0

        desktop_api = DesktopApi(base_url, api_token)
        window = webview.create_window(
            f"{APP_NAME} {APP_VERSION}",
            launch_url,
            js_api=desktop_api.js_api,
            width=1180,
            height=820,
            min_size=(920, 640),
        )
        if window is not None:
            desktop_api.attach_main_window(window)

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
        _release_backend_resources()
        restore_shutdown_handlers()
        if hard_exit_timer is not None:
            hard_exit_timer.cancel()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    exit_code = main()
    if getattr(sys, "frozen", False) and sys.platform == "darwin":
        os.environ["TRANS_EXIT_CODE"] = str(exit_code)
        _flush_logs()
        os._exit(exit_code)
    raise SystemExit(exit_code)

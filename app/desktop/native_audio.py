from __future__ import annotations

import asyncio
import contextlib
import ctypes
import json
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse, urlunparse

import numpy as np

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_QUEUE = 96
_OBJC_STREAM_OUTPUT_CLASS = None


def _error_message(error: Any) -> str:
    if error is None:
        return ""
    try:
        return str(error.localizedDescription())
    except Exception:
        return str(error)


def is_native_system_audio_available() -> tuple[bool, str]:
    if sys.platform != "darwin":
        return False, "仅 macOS 桌面版支持原生系统音频采集"
    try:
        import ScreenCaptureKit  # noqa: F401
        import CoreMedia  # noqa: F401
        import CoreAudio  # noqa: F401
        import objc  # noqa: F401
    except Exception as exc:
        return False, f"缺少 macOS 原生音频依赖: {exc}"
    return True, "ScreenCaptureKit 可用"


def _dispatch_queue(label: bytes):
    import objc

    lib = ctypes.CDLL("/usr/lib/libSystem.dylib")
    lib.dispatch_queue_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    lib.dispatch_queue_create.restype = ctypes.c_void_p
    ptr = lib.dispatch_queue_create(label, None)
    if not ptr:
        raise RuntimeError("无法创建 macOS dispatch queue")
    return objc.objc_object(c_void_p=ptr)


def _wait_for_objc_result(start_call: Callable[[Callable[..., None]], None], timeout: float, label: str):
    done = threading.Event()
    result: dict[str, Any] = {}

    def completion(*args):
        if len(args) >= 2:
            result["value"] = args[-2]
            result["error"] = args[-1]
        elif len(args) == 1:
            result["error"] = args[0]
        done.set()

    start_call(completion)
    if not done.wait(timeout):
        raise TimeoutError(f"{label} 超时")
    error = result.get("error")
    if error is not None:
        raise RuntimeError(f"{label} 失败: {_error_message(error)}")
    return result.get("value")


@dataclass
class _AudioFormat:
    sample_rate: int
    channels: int
    bits: int
    flags: int
    bytes_per_frame: int
    non_interleaved: bool
    is_float: bool
    is_signed_int: bool


class _StreamOutput:
    """Small Python wrapper around the SCStreamOutput Objective-C protocol."""

    def __init__(self, callback: Callable[[bytes], None]):
        self.callback = callback
        self._objc_output = self._make_objc_output()

    @property
    def objc_output(self):
        return self._objc_output

    def _make_objc_output(self):
        cls = _objc_stream_output_class()
        return cls.alloc().initWithCallback_(self.callback)


def _objc_stream_output_class():
    global _OBJC_STREAM_OUTPUT_CLASS
    if _OBJC_STREAM_OUTPUT_CLASS is not None:
        return _OBJC_STREAM_OUTPUT_CLASS

    import ScreenCaptureKit  # noqa: F401
    import objc
    from Foundation import NSObject

    class TransLiveStreamOutput(NSObject, protocols=[objc.protocolNamed("SCStreamOutput")]):
        def initWithCallback_(self, callback):
            self = objc.super(TransLiveStreamOutput, self).init()
            if self is None:
                return None
            self.callback = callback
            return self

        def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, output_type):
            try:
                chunk = _sample_buffer_to_int16(sample_buffer)
                if chunk:
                    self.callback(chunk)
            except Exception:
                logger.exception("Failed to process ScreenCaptureKit audio buffer")

    _OBJC_STREAM_OUTPUT_CLASS = TransLiveStreamOutput
    return _OBJC_STREAM_OUTPUT_CLASS


class ScreenCaptureKitAudioCapture:
    def __init__(self, on_audio: Callable[[bytes], None]):
        self.on_audio = on_audio
        self._stream: Any | None = None
        self._output: _StreamOutput | None = None
        self._queue: Any | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        ok, reason = is_native_system_audio_available()
        if not ok:
            raise RuntimeError(reason)

        import ScreenCaptureKit as S

        with self._lock:
            if self._stream is not None:
                return

            content = _wait_for_objc_result(
                lambda cb: S.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
                    False,
                    True,
                    cb,
                ),
                timeout=12.0,
                label="读取可采集屏幕列表",
            )
            displays = list(content.displays() or [])
            if not displays:
                raise RuntimeError("没有可采集的显示器")

            display = displays[0]
            filter_obj = S.SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, [])

            config = S.SCStreamConfiguration.alloc().init()
            config.setCapturesAudio_(True)
            if hasattr(config, "setCaptureMicrophone_"):
                config.setCaptureMicrophone_(False)
            if hasattr(config, "setExcludesCurrentProcessAudio_"):
                config.setExcludesCurrentProcessAudio_(True)
            if hasattr(config, "setSampleRate_"):
                config.setSampleRate_(TARGET_SAMPLE_RATE)
            config.setWidth_(max(2, min(int(display.width()), 1280)))
            config.setHeight_(max(2, min(int(display.height()), 720)))
            config.setQueueDepth_(3)

            self._stream = S.SCStream.alloc().initWithFilter_configuration_delegate_(filter_obj, config, None)
            self._output = _StreamOutput(self.on_audio)
            self._queue = _dispatch_queue(b"com.translive.native-system-audio")

            ok, error = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
                self._output.objc_output,
                S.SCStreamOutputTypeAudio,
                self._queue,
                None,
            )
            if not ok:
                self._stream = None
                raise RuntimeError(f"添加系统音频输出失败: {_error_message(error)}")

            _wait_for_objc_result(
                lambda cb: self._stream.startCaptureWithCompletionHandler_(cb),
                timeout=12.0,
                label="启动系统音频采集",
            )

    def stop(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
            self._output = None
            self._queue = None
        if stream is None:
            return
        try:
            _wait_for_objc_result(
                lambda cb: stream.stopCaptureWithCompletionHandler_(cb),
                timeout=6.0,
                label="停止系统音频采集",
            )
        except Exception:
            logger.exception("Failed to stop ScreenCaptureKit capture")


def _audio_format(sample_buffer: Any) -> _AudioFormat:
    import CoreAudio as CA
    import CoreMedia as CM

    fmt = CM.CMSampleBufferGetFormatDescription(sample_buffer)
    asbd = CM.CMAudioFormatDescriptionGetStreamBasicDescription(fmt)
    if asbd is None:
        raise RuntimeError("系统音频 sample buffer 缺少音频格式")

    flags = int(asbd.mFormatFlags)
    return _AudioFormat(
        sample_rate=max(1, int(round(asbd.mSampleRate))),
        channels=max(1, int(asbd.mChannelsPerFrame)),
        bits=max(1, int(asbd.mBitsPerChannel)),
        flags=flags,
        bytes_per_frame=max(1, int(asbd.mBytesPerFrame)),
        non_interleaved=bool(flags & CA.kAudioFormatFlagIsNonInterleaved),
        is_float=bool(flags & CA.kAudioFormatFlagIsFloat),
        is_signed_int=bool(flags & CA.kAudioFormatFlagIsSignedInteger),
    )


def _copy_block_buffer_data(block: Any) -> bytes:
    import CoreMedia as CM

    length = int(CM.CMBlockBufferGetDataLength(block))
    if length <= 0:
        return b""
    copied = CM.CMBlockBufferCopyDataBytes(block, 0, length, None)
    if isinstance(copied, tuple):
        status = copied[0]
        data = copied[1] if len(copied) > 1 else b""
        if status:
            raise RuntimeError(f"CMBlockBufferCopyDataBytes failed: {status}")
        return bytes(data)
    return bytes(copied or b"")


def _sample_buffer_to_int16(sample_buffer: Any) -> bytes:
    import CoreMedia as CM

    block = CM.CMSampleBufferGetDataBuffer(sample_buffer)
    if block is None:
        return b""

    fmt = _audio_format(sample_buffer)
    raw = _copy_block_buffer_data(block)
    if not raw:
        return b""

    audio = _decode_pcm(raw, fmt)
    if audio.size == 0:
        return b""
    if fmt.sample_rate != TARGET_SAMPLE_RATE:
        audio = _resample(audio, fmt.sample_rate, TARGET_SAMPLE_RATE)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16).tobytes()


def _decode_pcm(raw: bytes, fmt: _AudioFormat) -> np.ndarray:
    sample_width = max(1, fmt.bits // 8)
    if fmt.is_float and fmt.bits == 32:
        data = np.frombuffer(raw, dtype=np.float32).astype(np.float32, copy=False)
    elif fmt.is_float and fmt.bits == 64:
        data = np.frombuffer(raw, dtype=np.float64).astype(np.float32)
    elif fmt.is_signed_int and fmt.bits == 16:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif fmt.is_signed_int and fmt.bits == 32:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"不支持的系统音频格式: bits={fmt.bits} flags={fmt.flags}")

    if fmt.channels <= 1:
        return data.reshape(-1)

    frame_count = len(raw) // max(sample_width, fmt.bytes_per_frame)
    if fmt.non_interleaved:
        per_channel = data.size // fmt.channels
        if per_channel <= 0:
            return np.array([], dtype=np.float32)
        return data[: per_channel * fmt.channels].reshape(fmt.channels, per_channel).mean(axis=0)

    usable = (data.size // fmt.channels) * fmt.channels
    if usable <= 0:
        return np.array([], dtype=np.float32)
    if frame_count > 0:
        usable = min(usable, frame_count * fmt.channels)
    return data[:usable].reshape(-1, fmt.channels).mean(axis=1)


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly

        gcd = np.gcd(source_rate, target_rate)
        return resample_poly(audio, target_rate // gcd, source_rate // gcd).astype(np.float32)
    except Exception:
        ratio = target_rate / source_rate
        new_len = max(1, int(round(audio.size * ratio)))
        x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


class NativeSystemAudioBridge:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._websocket: Any | None = None
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._error: str | None = None
        self._capture: ScreenCaptureKitAudioCapture | None = None

    def available(self) -> dict[str, Any]:
        ok, reason = is_native_system_audio_available()
        return {"ok": ok, "reason": reason}

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive() and self._error is None
            return {"running": running, "error": self._error}

    def start(self, config: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {"ok": True, "running": True, "reused": True}
            self._stop_event.clear()
            self._started_event.clear()
            self._error = None
            self._thread = threading.Thread(
                target=self._thread_main,
                args=(dict(config or {}),),
                name="TransLiveNativeSystemAudio",
                daemon=True,
            )
            self._thread.start()

        if not self._started_event.wait(14.0):
            self.stop()
            error = self._error or "启动系统音频采集超时"
            return {"ok": False, "error": error}
        if self._error:
            return {"ok": False, "error": self._error}
        return {"ok": True, "running": True}

    def update_config(self, config: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
            loop = self._loop
            websocket = self._websocket
        if thread is None or not thread.is_alive() or loop is None or websocket is None:
            return {"ok": False, "error": "系统音频采集未运行"}
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._send_session_config(websocket, config),
                loop,
            )
            future.result(timeout=3.0)
            return {"ok": True, "running": True}
        except Exception as exc:
            logger.exception("Failed to update native audio session config")
            return {"ok": False, "error": str(exc)}

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        with self._lock:
            capture = self._capture
            thread = self._thread
            loop = self._loop
            websocket = self._websocket

        if loop is not None and websocket is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(websocket.close(), loop)
                future.result(timeout=1.5)
            except Exception:
                pass
        elif capture is not None:
            capture.stop()

        if thread is not None and thread.is_alive():
            thread.join(timeout=4.0)
            if thread.is_alive():
                logger.warning("Native system audio thread did not stop within timeout")
        with self._lock:
            if self._thread is thread:
                self._thread = None
            if self._capture is capture:
                self._capture = None
            if self._loop is loop:
                self._loop = None
            if self._websocket is websocket:
                self._websocket = None
        return {"ok": True, "running": False}

    def _thread_main(self, config: dict[str, Any]) -> None:
        try:
            asyncio.run(self._run(config))
        except Exception as exc:
            logger.exception("Native system audio bridge failed")
            self._error = str(exc)
            self._started_event.set()
        finally:
            self._stop_event.set()

    async def _run(self, config: dict[str, Any]) -> None:
        import websockets

        audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=MAX_AUDIO_QUEUE)

        def on_audio(chunk: bytes) -> None:
            if self._stop_event.is_set():
                return
            try:
                audio_queue.put_nowait(chunk)
            except queue.Full:
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    audio_queue.put_nowait(chunk)
                except queue.Full:
                    pass

        ws_url = self._stream_ws_url()
        async with websockets.connect(ws_url, max_size=None) as websocket:
            with self._lock:
                self._loop = asyncio.get_running_loop()
                self._websocket = websocket
            await self._send_session_config(websocket, config)

            capture = ScreenCaptureKitAudioCapture(on_audio)
            capture.start()
            self._capture = capture
            self._started_event.set()

            reader = asyncio.create_task(self._drain_messages(websocket))
            try:
                while not self._stop_event.is_set():
                    try:
                        chunk = await asyncio.to_thread(audio_queue.get, True, 0.25)
                    except queue.Empty:
                        continue
                    await websocket.send(chunk)
            finally:
                reader.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await reader
                capture.stop()
                self._capture = None
                with self._lock:
                    self._websocket = None
                    self._loop = None

    async def _send_session_config(self, websocket: Any, config: dict[str, Any]) -> None:
        await websocket.send(json.dumps({
            "type": "config",
            "source_lang": config.get("source_lang") or "en",
            "target_lang": config.get("target_lang") or "zh",
        }))
        glossary = config.get("glossary")
        if isinstance(glossary, dict):
            await websocket.send(json.dumps({"type": "glossary", "glossary": glossary}))

    async def _drain_messages(self, websocket: Any) -> None:
        while not self._stop_event.is_set():
            try:
                await websocket.recv()
            except asyncio.CancelledError:
                raise
            except Exception:
                return

    def _stream_ws_url(self) -> str:
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return urlunparse((scheme, parsed.netloc, "/ws/stream", "", "", ""))

    @staticmethod
    def settle_delay() -> None:
        time.sleep(0.05)

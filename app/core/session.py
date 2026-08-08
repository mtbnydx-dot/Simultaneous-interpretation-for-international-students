import asyncio
import json
import logging
import re
import secrets
import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect
from app.core.config import settings
from app.asr.engine import asr_engine
from app.mt.engine import mt_engine
from app.core.audio_preprocess import audio_preprocessor
from app.core.perf_monitor import perf_monitor
from app.core.subtitle_hub import subtitle_hub
from app.core.transcript_store import transcript_store
from app.core.languages import (
    AUTO_LANGUAGE_CODE,
    language_code_from_name,
    normalize_language_code,
    normalize_source_language,
)
from app.core.power import realtime_activity
from app.core.vad import FRAME_SAMPLES as VAD_FRAME_SAMPLES

logger = logging.getLogger(__name__)

_ASR_TEXT_NORMALIZE_RE = re.compile(r"[\s\.,!?;:，。！？、；：\"'`“”‘’\[\]\(\){}<>《》…~\-–—_]+")
_ASR_FILTER_LOCK = threading.Lock()
_ASR_FILTER_SUPPRESSED = 0
_ASR_FILTER_RECENT: deque[dict] = deque(maxlen=8)
_SEGMENT_ID_LOCK = threading.Lock()
_SEGMENT_ID_MAX = 2**53 - 1
_SEGMENT_ID = min(
    int(time.time() * 1000) * 1000 + secrets.randbelow(1000),
    _SEGMENT_ID_MAX - 1_000_000,
)
_SEGMENT_QUEUE_STOP = object()


def _next_segment_id() -> int:
    """Return a process-wide, JavaScript-safe, monotonically increasing ID."""
    global _SEGMENT_ID
    with _SEGMENT_ID_LOCK:
        if _SEGMENT_ID >= _SEGMENT_ID_MAX:
            raise RuntimeError("segment ID space exhausted")
        _SEGMENT_ID += 1
        return _SEGMENT_ID


def _normalize_asr_text(text: str) -> str:
    return _ASR_TEXT_NORMALIZE_RE.sub("", text.casefold())


def _audio_rms(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def _record_asr_filter_suppression(text: str, reason: str, audio_duration: float, raw_rms: float, speech_ratio: float) -> None:
    global _ASR_FILTER_SUPPRESSED
    item = {
        "text": text[:80],
        "reason": reason,
        "audio_duration_s": round(audio_duration, 2),
        "rms": round(raw_rms, 5),
        "speech_ratio": round(speech_ratio, 3),
        "ts": time.time(),
    }
    with _ASR_FILTER_LOCK:
        _ASR_FILTER_SUPPRESSED += 1
        _ASR_FILTER_RECENT.append(item)


def asr_filter_status() -> dict:
    with _ASR_FILTER_LOCK:
        return {
            "enabled": settings.asr_hallucination_filter_enabled,
            "suppressed_segments": _ASR_FILTER_SUPPRESSED,
            "recent": list(_ASR_FILTER_RECENT),
            "low_speech_ratio": settings.asr_hallucination_low_speech_ratio,
            "max_speech_duration": settings.asr_hallucination_max_speech_duration,
        }


def _should_emit_mt_partial(
    accumulated_text: str,
    now: float,
    last_emit_at: float,
    last_emit_len: int,
) -> bool:
    if not accumulated_text:
        return False
    if last_emit_at <= 0:
        return True
    min_interval = max(0, settings.mt_stream_partial_min_interval_ms) / 1000.0
    min_chars = max(1, int(settings.mt_stream_partial_min_chars or 1))
    return (now - last_emit_at) >= min_interval or (len(accumulated_text) - last_emit_len) >= min_chars


def _should_emit_asr_preview(
    accumulated_text: str,
    now: float,
    last_emit_at: float,
) -> bool:
    normalized = _normalize_asr_text(accumulated_text)
    if len(normalized) < max(1, int(settings.asr_streaming_preview_min_chars or 1)):
        return False
    if last_emit_at <= 0:
        return True
    min_interval = max(0, int(settings.asr_streaming_preview_min_interval_ms or 0)) / 1000.0
    return (now - last_emit_at) >= min_interval


def _latency_mode_settings(mode: str | None) -> dict[str, float | int | str]:
    normalized = str(mode or settings.session_latency_mode or "balanced").strip().lower()
    if normalized not in {"low", "balanced", "stable"}:
        normalized = "balanced"

    if normalized == "low":
        return {
            "mode": "low",
            "segment_min_duration": 0.45,
            "segment_max_duration": min(float(settings.segment_max_duration), 5.0),
            "segment_silence_duration_ms": 250,
        }
    if normalized == "stable":
        return {
            "mode": "stable",
            "segment_min_duration": max(float(settings.segment_min_duration), 1.2),
            "segment_max_duration": max(float(settings.segment_max_duration), 10.0),
            "segment_silence_duration_ms": max(int(settings.segment_silence_duration_ms), 700),
        }
    return {
        "mode": "balanced",
        "segment_min_duration": float(settings.segment_min_duration),
        "segment_max_duration": float(settings.segment_max_duration),
        "segment_silence_duration_ms": int(settings.segment_silence_duration_ms),
    }


@dataclass(frozen=True)
class _SegmentJob:
    segment_id: int
    audio: np.ndarray
    speech_duration: float
    source_lang: str
    target_lang: str
    latency_mode: str
    glossary: dict[str, str]
    persist_history: bool
    streaming_preview: bool
    queued_at: float


class StreamSession:
    def __init__(self, websocket: WebSocket, subprotocol: str | None = None):
        self.ws = websocket
        self._subprotocol = subprotocol
        self._audio_buffer: list[np.ndarray] = []
        self._is_running = False
        self._client_connected = False
        self._source_lang: str = normalize_source_language(settings.source_lang, "en")
        self._target_lang: str = normalize_language_code(settings.target_lang, "zh")
        self._speech_seen = False
        self._speech_duration = 0.0
        self._trailing_silence_duration = 0.0
        self._vad = None
        # Silero 只吃固定 512 采样点的帧，网络分片长度不保证对齐（系统音频路径
        # 尤其如此），这里保留不足一帧的余数，等下一片凑齐再送进去。
        self._vad_residual = np.empty(0, dtype=np.float32)
        self._vad_degraded = False
        # 增量维护采样计数：原来每片都要 sum() 整个 buffer 列表，一段 8s 音频
        # 会退化成 O(n²)（约 150 万次求和）。
        self._buffered_samples = 0
        # 会话级术语表 —— 不污染全局 settings.mt_glossary，避免多客户端互相覆盖
        self._glossary: dict[str, str] = {}
        self._persist_history = bool(settings.transcript_history_enabled)
        self._latency_mode = "balanced"
        self._segment_min_duration = float(settings.segment_min_duration)
        self._segment_max_duration = float(settings.segment_max_duration)
        self._segment_silence_duration_ms = int(settings.segment_silence_duration_ms)
        self._segment_queue: asyncio.Queue[_SegmentJob | object] = asyncio.Queue(
            maxsize=max(1, int(settings.stream_segment_queue_max_size or 1))
        )
        self._processor_task: asyncio.Task | None = None
        self._dropped_segments = 0
        self._streaming_preview_enabled = bool(settings.asr_streaming_preview_enabled)
        self._apply_latency_mode(settings.session_latency_mode)

    async def run(self):
        await self.ws.accept(subprotocol=self._subprotocol)
        self._is_running = True
        self._client_connected = True
        # 只在真正跑同传期间阻止系统降频/休眠，空闲时让机器该省电就省电
        realtime_activity.acquire()
        self._processor_task = asyncio.create_task(self._process_segment_queue())
        logger.info("Stream session started (src=%s, tgt=%s)",
                     self._source_lang, self._target_lang)

        graceful_stop = False
        try:
            while self._is_running:
                message = await self.ws.receive()
                msg_type = message.get("type", "")

                if msg_type == "websocket.disconnect":
                    break
                elif msg_type == "websocket.receive" and "text" in message:
                    text = message["text"]
                    if self._message_too_large(text):
                        await self._close_with_error(
                            1009,
                            "message_too_large",
                            "WebSocket 消息过大，连接已关闭。",
                        )
                        break
                    control_type = self._handle_config(text)
                    if control_type == "stop":
                        await self._shutdown_pipeline(graceful=True)
                        await self._send_session_stopped()
                        graceful_stop = True
                        break
                elif msg_type == "websocket.receive" and "bytes" in message:
                    raw = message["bytes"]
                    if self._message_too_large(raw):
                        await self._close_with_error(
                            1009,
                            "message_too_large",
                            "音频分片过大，连接已关闭。",
                        )
                        break
                    if len(raw) % 2:
                        await self._close_with_error(
                            1003,
                            "invalid_audio_chunk",
                            "音频分片格式无效。",
                        )
                        break
                    int16_chunk = np.frombuffer(raw, dtype=np.int16)
                    audio_chunk = int16_chunk.astype(np.float32)
                    audio_chunk *= 1.0 / 32768.0
                    self._audio_buffer.append(audio_chunk)
                    self._buffered_samples += audio_chunk.size
                    self._update_voice_state(audio_chunk)

                    if self._should_drop_silence():
                        self._reset_buffer()
                    elif self._buffer_duration() > self._segment_max_duration:
                        logger.warning("Audio buffer exceeded %.0fs, processing early",
                                       self._segment_max_duration)
                        await self._enqueue_current_buffer("max_duration")
                    elif self._should_process():
                        await self._enqueue_current_buffer("vad_silence")

        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception:
            logger.exception("Session error")
        finally:
            if not graceful_stop:
                await self._shutdown_pipeline(graceful=False)
            self._is_running = False
            self._client_connected = False
            realtime_activity.release()

    def _message_too_large(self, value: str | bytes) -> bool:
        max_bytes = max(1, int(settings.websocket_max_message_bytes or 1))
        if isinstance(value, str):
            return len(value.encode("utf-8")) > max_bytes
        return len(value) > max_bytes

    async def _close_with_error(self, code: int, error: str, message: str) -> None:
        logger.warning("Closing stream websocket: %s (%s)", error, message)
        try:
            await self.ws.send_json({"type": "error", "error": error, "text": message})
        except Exception:
            pass
        try:
            await self.ws.close(code=code)
        except Exception:
            pass
        self._is_running = False
        self._client_connected = False

    def _handle_config(self, text: str) -> str | None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Ignoring non-JSON text message: %s", text[:100])
            return None

        if data.get("type") == "stop":
            return "stop"

        if data.get("type") == "config":
            if "source_lang" in data:
                raw_source = data["source_lang"]
                # 源语言允许 "auto"（Qwen3-ASR 自带语种识别，可应付中英混说）
                normalized = normalize_source_language(raw_source, self._source_lang)
                if normalized != raw_source:
                    logger.warning("Invalid source_lang %r; using %s", raw_source, normalized)
                self._source_lang = normalized
            if "target_lang" in data:
                raw_target = data["target_lang"]
                normalized = normalize_language_code(raw_target, self._target_lang)
                if normalized != raw_target:
                    logger.warning("Invalid target_lang %r; using %s", raw_target, normalized)
                self._target_lang = normalized
            if "latency_mode" in data:
                self._apply_latency_mode(data.get("latency_mode"))
            if "persist_history" in data:
                self._persist_history = bool(data.get("persist_history")) and bool(settings.transcript_history_enabled)
            if "streaming_preview" in data:
                self._set_streaming_preview_enabled(bool(data.get("streaming_preview")))
            logger.info(
                "Session config updated: src=%s, tgt=%s, latency=%s, persist_history=%s, streaming_preview=%s",
                self._source_lang,
                self._target_lang,
                self._latency_mode,
                self._persist_history,
                self._streaming_preview_enabled,
            )

        elif data.get("type") == "glossary":
            glossary = data.get("glossary", {})
            if isinstance(glossary, dict):
                # 仅保留非空字符串键值对，绑定到本会话
                self._glossary = {
                    str(k): str(v) for k, v in glossary.items()
                    if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
                }
                logger.info("Session glossary updated: %d entries", len(self._glossary))
        return None

    def _get_vad(self):
        if self._vad is None:
            from app.core.vad import get_vad
            self._vad = get_vad()
        return self._vad

    def _should_process(self) -> bool:
        return (
            self._speech_seen
            and self._buffer_duration() >= self._segment_min_duration
            and self._trailing_silence_duration >= self._segment_silence_duration_ms / 1000
        )

    def _should_drop_silence(self) -> bool:
        return not self._speech_seen and self._buffer_duration() >= self._segment_min_duration

    def _apply_latency_mode(self, mode: str | None) -> None:
        values = _latency_mode_settings(mode)
        self._latency_mode = str(values["mode"])
        self._segment_min_duration = float(values["segment_min_duration"])
        self._segment_max_duration = float(values["segment_max_duration"])
        self._segment_silence_duration_ms = int(values["segment_silence_duration_ms"])

    def _set_streaming_preview_enabled(self, enabled: bool) -> None:
        self._streaming_preview_enabled = enabled

    def _frame_has_speech(self, frame: np.ndarray) -> bool:
        try:
            return self._get_vad().detect_speech(frame)
        except Exception:
            if not self._vad_degraded:
                self._vad_degraded = True
                logger.warning(
                    "VAD failed on a %d-sample frame; falling back to RMS for this session",
                    frame.size,
                    exc_info=True,
                )
            rms = float(np.sqrt(np.dot(frame, frame) / max(1, frame.size)))
            return rms >= settings.audio_vad_rms_threshold

    def _update_voice_state(self, audio_chunk: np.ndarray):
        if audio_chunk.size == 0:
            return

        pending = (
            np.concatenate((self._vad_residual, audio_chunk))
            if self._vad_residual.size
            else audio_chunk
        )
        frame_count = pending.size // VAD_FRAME_SAMPLES
        if frame_count == 0:
            self._vad_residual = pending
            return

        usable = frame_count * VAD_FRAME_SAMPLES
        self._vad_residual = pending[usable:]
        frame_duration = VAD_FRAME_SAMPLES / settings.sample_rate

        for start in range(0, usable, VAD_FRAME_SAMPLES):
            if self._frame_has_speech(pending[start:start + VAD_FRAME_SAMPLES]):
                self._speech_seen = True
                self._speech_duration += frame_duration
                self._trailing_silence_duration = 0.0
            elif self._speech_seen:
                self._trailing_silence_duration += frame_duration

    def _reset_buffer(self):
        self._audio_buffer.clear()
        self._buffered_samples = 0
        self._speech_seen = False
        self._speech_duration = 0.0
        self._trailing_silence_duration = 0.0
        self._vad_residual = np.empty(0, dtype=np.float32)
        # Silero 是 RNN，跨段沿用隐状态会让新段开头的判断被上一段污染
        if self._vad is not None:
            try:
                self._vad.reset_states()
            except Exception:
                logger.debug("VAD reset_states failed", exc_info=True)

    def _buffer_duration(self) -> float:
        return self._buffered_samples / settings.sample_rate

    async def _send_preview_clear(self, seg_id: int) -> None:
        await self._send_event(
            {"type": "original_preview_clear", "segment_id": seg_id},
            persist_history=False,
        )

    def _ws_sendable(self) -> bool:
        return self._is_running and self._client_connected

    async def _send_event(self, payload: dict, persist_history: bool | None = None):
        if not self._is_running:
            return
        await subtitle_hub.publish(payload)
        if self._ws_sendable():
            try:
                await self.ws.send_json(payload)
            except Exception:
                self._client_connected = False
                logger.exception("Failed to send websocket event")
        should_persist = self._persist_history if persist_history is None else persist_history
        if should_persist:
            await transcript_store.record_event(payload)

    async def _enqueue_current_buffer(self, reason: str, *, allow_drop: bool = True) -> None:
        if not self._audio_buffer:
            return

        audio = np.concatenate(self._audio_buffer)
        speech_duration = self._speech_duration
        seg_id = _next_segment_id()
        self._reset_buffer()

        job = _SegmentJob(
            segment_id=seg_id,
            audio=audio,
            speech_duration=speech_duration,
            source_lang=self._source_lang,
            target_lang=self._target_lang,
            latency_mode=self._latency_mode,
            glossary=dict(self._glossary),
            persist_history=self._persist_history,
            streaming_preview=self._streaming_preview_enabled,
            queued_at=time.perf_counter(),
        )

        if self._segment_queue.full() and allow_drop:
            dropped = self._drop_oldest_segment()
            if dropped is not None:
                self._dropped_segments += 1
                logger.warning(
                    "Dropped queued segment %s because ASR/MT pipeline is behind (new=%s, reason=%s)",
                    dropped.segment_id,
                    seg_id,
                    reason,
                )
                await self._send_event(
                    {
                        "type": "error",
                        "segment_id": dropped.segment_id,
                        "text": "实时处理积压，已丢弃过旧音频片段以追上当前语音。",
                        "warning": "segment_queue_overflow",
                        "source_lang": dropped.source_lang,
                        "target_lang": dropped.target_lang,
                        "latency_mode": dropped.latency_mode,
                    },
                    persist_history=dropped.persist_history,
                )

        if not allow_drop:
            await self._segment_queue.put(job)
            return

        try:
            self._segment_queue.put_nowait(job)
        except asyncio.QueueFull:
            self._dropped_segments += 1
            logger.warning("Dropped new segment %s because segment queue is still full", seg_id)
            await self._send_event(
                {
                    "type": "error",
                    "segment_id": seg_id,
                    "text": "实时处理积压，当前音频片段已跳过。",
                    "warning": "segment_queue_overflow",
                    "source_lang": job.source_lang,
                    "target_lang": job.target_lang,
                    "latency_mode": job.latency_mode,
                },
                persist_history=job.persist_history,
            )

    def _drop_oldest_segment(self) -> _SegmentJob | None:
        try:
            dropped = self._segment_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        self._segment_queue.task_done()
        return dropped if isinstance(dropped, _SegmentJob) else None

    async def _process_segment_queue(self) -> None:
        try:
            while True:
                queued = await self._segment_queue.get()
                try:
                    if queued is _SEGMENT_QUEUE_STOP:
                        return
                    await self._process_segment(queued)
                finally:
                    self._segment_queue.task_done()
        except asyncio.CancelledError:
            raise

    async def _shutdown_pipeline(self, graceful: bool) -> None:
        """Flush useful tail audio and drain work, or cancel promptly on disconnect."""
        task = self._processor_task
        if task is None:
            self._reset_buffer()
            return

        if graceful:
            if self._audio_buffer and self._speech_seen:
                await self._enqueue_current_buffer("client_stop", allow_drop=False)
            else:
                self._reset_buffer()
            await self._segment_queue.put(_SEGMENT_QUEUE_STOP)
            try:
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=max(1.0, float(settings.session_drain_timeout_s or 1.0)),
                )
            except asyncio.TimeoutError:
                logger.warning("Session drain timed out; cancelling remaining work")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        else:
            self._reset_buffer()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._processor_task = None

    async def _send_session_stopped(self) -> None:
        if not self._ws_sendable():
            return
        try:
            await self.ws.send_json({"type": "session_stopped"})
        except Exception:
            self._client_connected = False

    async def _process_buffer(self):
        await self._enqueue_current_buffer("manual")

    async def _run_asr(self, audio: np.ndarray, job: _SegmentJob, loop):
        if not job.streaming_preview:
            result = await loop.run_in_executor(
                None,
                asr_engine.transcribe_result,
                audio,
                job.source_lang,
            )
            return result, False

        partial_queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue(maxsize=1)
        started_at = time.perf_counter()

        def _offer_latest(text: str, language: str | None) -> None:
            if not self._is_running:
                return
            try:
                partial_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            partial_queue.put_nowait((text, language))

        def _on_partial(text: str, language: str | None) -> None:
            loop.call_soon_threadsafe(_offer_latest, text, language)

        asr_future = loop.run_in_executor(
            None,
            asr_engine.transcribe_stream_result,
            audio,
            job.source_lang,
            _on_partial,
        )
        preview_sent = False
        last_emit_at = 0.0
        last_text = ""

        while not asr_future.done() or not partial_queue.empty():
            try:
                text, detected_language = await asyncio.wait_for(
                    partial_queue.get(),
                    timeout=0.05,
                )
            except asyncio.TimeoutError:
                continue

            now = time.perf_counter()
            if text == last_text or not _should_emit_asr_preview(text, now, last_emit_at):
                continue
            if not self._ws_sendable():
                continue

            source_lang = job.source_lang
            if source_lang == AUTO_LANGUAGE_CODE:
                source_lang = language_code_from_name(detected_language, job.target_lang)
            await self._send_event(
                {
                    "type": "original_preview",
                    "segment_id": job.segment_id,
                    "text": text,
                    "audio_duration_ms": round(len(audio) / max(1, settings.sample_rate) * 1000),
                    "preview_ms": round((now - started_at) * 1000),
                    "source_lang": source_lang,
                    "target_lang": job.target_lang,
                    "latency_mode": job.latency_mode,
                },
                persist_history=False,
            )
            preview_sent = True
            last_emit_at = now
            last_text = text

        return await asr_future, preview_sent

    async def _process_segment(self, job: _SegmentJob):
        audio = job.audio
        speech_duration = job.speech_duration
        audio_duration = len(audio) / settings.sample_rate
        seg_id = job.segment_id

        queue_wait_ms = round(max(0.0, time.perf_counter() - job.queued_at) * 1000)
        timer = perf_monitor.create_timer()
        timer.start(audio_duration)

        loop = asyncio.get_running_loop()
        preview_sent = False

        try:
            # Step 1: 音频预处理
            raw_audio = audio
            raw_rms = _audio_rms(raw_audio)
            audio = await loop.run_in_executor(
                None, audio_preprocessor.process, audio
            )
            timer.mark_preprocess_done()

            # Step 2: ASR 转写
            asr_result, preview_sent = await self._run_asr(audio, job, loop)
            original = asr_result.text
            # 源语言选了"自动检测"时，用 ASR 识别出的语种去做翻译，
            # 否则 MT 会拿着 "auto" 当语言名写进 prompt。
            if job.source_lang == AUTO_LANGUAGE_CODE:
                source_lang = language_code_from_name(asr_result.language, self._target_lang)
            else:
                source_lang = job.source_lang
            timer.mark_asr_done()

            if not original.strip():
                timer.mark_mt_done()
                if preview_sent:
                    await self._send_preview_clear(seg_id)
                return

            should_ignore, ignore_reason, speech_ratio = self._asr_filter_decision(
                original,
                audio_duration,
                raw_rms,
                speech_duration,
            )
            if should_ignore:
                logger.info(
                    "Ignoring likely ASR hallucination: %r (reason=%s duration=%.2fs speech=%.2fs ratio=%.2f rms=%.4f)",
                    original,
                    ignore_reason,
                    audio_duration,
                    speech_duration,
                    speech_ratio,
                    raw_rms,
                )
                _record_asr_filter_suppression(original, ignore_reason, audio_duration, raw_rms, speech_ratio)
                timer.mark_mt_done()
                if preview_sent:
                    await self._send_preview_clear(seg_id)
                return

            if not self._ws_sendable():
                timer.mark_mt_done()
                return

            await self._send_event({
                "type": "original",
                "segment_id": seg_id,
                "text": original,
                "audio_duration_ms": round(audio_duration * 1000),
                "asr_ms": round(timer.asr_time * 1000),
                "source_lang": source_lang,
                "target_lang": job.target_lang,
                "latency_mode": job.latency_mode,
                "queue_wait_ms": queue_wait_ms,
                "dropped_segments": self._dropped_segments,
            }, persist_history=job.persist_history)

            # Step 3: MT 流式翻译 (generator 在 executor 中运行，避免阻塞事件循环)
            mt_start = time.perf_counter()
            token_queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, int(settings.mt_stream_queue_max_size or 1)))
            sentinel = object()
            stop_stream = threading.Event()
            stream_timeout_s = max(1.0, float(settings.mt_stream_timeout_s or 1.0))
            stream_deadline = mt_start + stream_timeout_s
            max_output_chars = max(0, int(settings.mt_stream_max_output_chars or 0))
            queue_put_timeout = max(0.1, float(settings.mt_stream_queue_put_timeout_s or 0.1))

            session_glossary = job.glossary or None

            def _push_token(item):
                if stop_stream.is_set() and item is not sentinel:
                    return
                try:
                    future = asyncio.run_coroutine_threadsafe(token_queue.put(item), loop)
                    future.result(timeout=queue_put_timeout)
                except Exception:
                    stop_stream.set()
                    pass

            def _run_stream():
                """在线程中运行同步 generator，将 token 放入队列"""
                try:
                    for token in mt_engine.translate_stream(
                        original,
                        source_lang,
                        job.target_lang,
                        glossary=session_glossary,
                    ):
                        if stop_stream.is_set():
                            break
                        _push_token(token)
                except Exception as exc:
                    _push_token(exc)
                finally:
                    _push_token(sentinel)

            executor_task = loop.run_in_executor(None, _run_stream)
            accumulated_text = ""
            last_partial_at = 0.0
            last_partial_len = 0
            mt_warning: str | None = None
            truncated = False

            while True:
                if settings.mt_stream_stop_on_disconnect and not self._ws_sendable():
                    stop_stream.set()
                    mt_warning = "client_disconnected"
                    break

                remaining = stream_deadline - time.perf_counter()
                if remaining <= 0:
                    stop_stream.set()
                    mt_warning = "mt_timeout"
                    break

                try:
                    token = await asyncio.wait_for(token_queue.get(), timeout=min(1.0, remaining))
                except asyncio.TimeoutError:
                    continue

                if token is sentinel:
                    break
                if isinstance(token, Exception):
                    raise token
                accumulated_text += token
                if max_output_chars and len(accumulated_text) >= max_output_chars:
                    accumulated_text = accumulated_text[:max_output_chars].rstrip()
                    stop_stream.set()
                    mt_warning = "mt_output_truncated"
                    truncated = True
                    await self._send_event({
                        "type": "translated_partial",
                        "segment_id": seg_id,
                        "text": "",
                        "accumulated": accumulated_text,
                        "source_lang": source_lang,
                        "target_lang": job.target_lang,
                        "latency_mode": job.latency_mode,
                        "warning": mt_warning,
                        "truncated": truncated,
                    }, persist_history=job.persist_history)
                    break
                now = time.perf_counter()
                if _should_emit_mt_partial(accumulated_text, now, last_partial_at, last_partial_len):
                    await self._send_event({
                        "type": "translated_partial",
                        "segment_id": seg_id,
                        "text": token,
                        "accumulated": accumulated_text,
                        "source_lang": source_lang,
                        "target_lang": job.target_lang,
                        "latency_mode": job.latency_mode,
                    }, persist_history=job.persist_history)
                    last_partial_at = now
                    last_partial_len = len(accumulated_text)

            if stop_stream.is_set():
                try:
                    await asyncio.wait_for(asyncio.shield(executor_task), timeout=1.0)
                except asyncio.TimeoutError:
                    logger.warning("MT stream worker did not stop promptly (segment=%s warning=%s)", seg_id, mt_warning)
                except Exception:
                    raise
            else:
                await executor_task  # 确保线程完成
            full_text = mt_engine.clean_translation_output(
                accumulated_text,
                source_text=original,
                source_lang=source_lang,
                target_lang=job.target_lang,
            )
            if full_text != accumulated_text.strip() and not mt_warning:
                mt_warning = "mt_output_cleaned"
            mt_time = time.perf_counter() - mt_start
            timer.mark_mt_done(mt_time)

            if mt_warning == "mt_timeout" and not full_text:
                raise TimeoutError(f"MT stream timed out after {stream_timeout_s:.0f}s")

            await self._send_event({
                "type": "translated",
                "segment_id": seg_id,
                "text": full_text,
                "audio_duration_ms": round(audio_duration * 1000),
                "asr_ms": round(timer.asr_time * 1000),
                "mt_ms": round(mt_time * 1000),
                "total_ms": round(timer.total_time * 1000),
                "rtf": round(timer.rtf, 3),
                "source_lang": source_lang,
                "target_lang": job.target_lang,
                "latency_mode": job.latency_mode,
                "warning": mt_warning,
                "truncated": truncated,
                "queue_wait_ms": queue_wait_ms,
                "dropped_segments": self._dropped_segments,
            }, persist_history=job.persist_history)

        except Exception as e:
            logger.exception("Processing error")
            if preview_sent:
                await self._send_preview_clear(seg_id)
            if self._ws_sendable():
                try:
                    await self._send_event({
                        "type": "error",
                        "segment_id": seg_id,
                        "text": str(e),
                        "source_lang": job.source_lang,
                        "target_lang": job.target_lang,
                        "latency_mode": job.latency_mode,
                    }, persist_history=job.persist_history)
                except Exception:
                    pass

    def _asr_filter_decision(
        self,
        text: str,
        audio_duration: float,
        raw_rms: float,
        speech_duration: float,
    ) -> tuple[bool, str, float]:
        if not settings.asr_hallucination_filter_enabled:
            return False, "", 0.0

        normalized = _normalize_asr_text(text)
        if not normalized:
            return True, "empty_text", 0.0

        speech_ratio = max(0.0, min(1.0, speech_duration / audio_duration)) if audio_duration > 0 else 0.0
        is_short_audio = audio_duration <= settings.asr_hallucination_max_duration
        is_quiet_audio = raw_rms <= settings.asr_hallucination_low_rms
        is_low_speech_ratio = speech_ratio <= settings.asr_hallucination_low_speech_ratio
        is_brief_speech = speech_duration <= settings.asr_hallucination_max_speech_duration
        is_tiny_text = len(normalized) <= settings.asr_min_content_chars

        if is_tiny_text and (is_quiet_audio or is_low_speech_ratio):
            return True, "tiny_text_low_signal", speech_ratio

        phrases = {
            _normalize_asr_text(phrase)
            for phrase in settings.asr_hallucination_phrases
            if phrase
        }
        if normalized in phrases:
            if is_short_audio:
                return True, "known_phrase_short_audio", speech_ratio
            if is_quiet_audio:
                return True, "known_phrase_quiet_audio", speech_ratio
            if is_low_speech_ratio:
                return True, "known_phrase_low_speech_ratio", speech_ratio
            if is_brief_speech:
                return True, "known_phrase_brief_speech", speech_ratio

        return False, "", speech_ratio

    def _should_ignore_asr_text(self, text: str, audio_duration: float, raw_rms: float) -> bool:
        return self._asr_filter_decision(text, audio_duration, raw_rms, audio_duration)[0]

    def stop(self):
        self._is_running = False
        self._client_connected = False
        self._reset_buffer()

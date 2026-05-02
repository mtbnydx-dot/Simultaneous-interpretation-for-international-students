import asyncio
import json
import logging
import re
import time
import numpy as np

from fastapi import WebSocket, WebSocketDisconnect
from app.core.config import settings
from app.asr.engine import asr_engine
from app.mt.engine import mt_engine
from app.core.audio_preprocess import audio_preprocessor
from app.core.perf_monitor import perf_monitor
from app.core.subtitle_hub import subtitle_hub

logger = logging.getLogger(__name__)

_ASR_TEXT_NORMALIZE_RE = re.compile(r"[\s\.,!?;:，。！？、；：\"'`“”‘’\[\]\(\){}<>《》…~\-–—_]+")


def _normalize_asr_text(text: str) -> str:
    return _ASR_TEXT_NORMALIZE_RE.sub("", text.casefold())


def _audio_rms(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


class StreamSession:
    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self._audio_buffer: list[np.ndarray] = []
        self._is_running = False
        self._client_connected = False
        self._source_lang: str = settings.source_lang
        self._target_lang: str = settings.target_lang
        self._speech_seen = False
        self._trailing_silence_duration = 0.0
        self._segment_counter = 0
        self._vad = None
        # 会话级术语表 —— 不污染全局 settings.mt_glossary，避免多客户端互相覆盖
        self._glossary: dict[str, str] = {}

    async def run(self):
        await self.ws.accept()
        self._is_running = True
        self._client_connected = True
        logger.info("Stream session started (src=%s, tgt=%s)",
                     self._source_lang, self._target_lang)

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
                    self._handle_config(text)
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
                    audio_chunk = int16_chunk.astype(np.float32) / 32768.0
                    self._audio_buffer.append(audio_chunk)
                    self._update_voice_state(audio_chunk)

                    if self._should_drop_silence():
                        self._reset_buffer()
                    elif self._buffer_duration() > settings.segment_max_duration:
                        logger.warning("Audio buffer exceeded %.0fs, processing early",
                                       settings.segment_max_duration)
                        await self._process_buffer()
                    elif self._should_process():
                        await self._process_buffer()

        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception:
            logger.exception("Session error")
        finally:
            self._is_running = False
            self._client_connected = False

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

    def _handle_config(self, text: str):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Ignoring non-JSON text message: %s", text[:100])
            return

        if data.get("type") == "config":
            if "source_lang" in data:
                self._source_lang = data["source_lang"]
            if "target_lang" in data:
                self._target_lang = data["target_lang"]
            logger.info("Session config updated: src=%s, tgt=%s",
                        self._source_lang, self._target_lang)

        elif data.get("type") == "glossary":
            glossary = data.get("glossary", {})
            if isinstance(glossary, dict):
                # 仅保留非空字符串键值对，绑定到本会话
                self._glossary = {
                    str(k): str(v) for k, v in glossary.items()
                    if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
                }
                logger.info("Session glossary updated: %d entries", len(self._glossary))

    def _get_vad(self):
        if self._vad is None:
            from app.core.vad import get_vad
            self._vad = get_vad()
        return self._vad

    def _should_process(self) -> bool:
        return (
            self._speech_seen
            and self._buffer_duration() >= settings.segment_min_duration
            and self._trailing_silence_duration >= settings.segment_silence_duration_ms / 1000
        )

    def _should_drop_silence(self) -> bool:
        return not self._speech_seen and self._buffer_duration() >= settings.segment_min_duration

    def _update_voice_state(self, audio_chunk: np.ndarray):
        if len(audio_chunk) == 0:
            return

        try:
            has_speech = self._get_vad().detect_speech(audio_chunk)
        except Exception:
            rms = float(np.sqrt(np.mean(np.square(audio_chunk))))
            has_speech = rms >= settings.audio_vad_rms_threshold

        chunk_duration = len(audio_chunk) / settings.sample_rate

        if has_speech:
            self._speech_seen = True
            self._trailing_silence_duration = 0.0
        elif self._speech_seen:
            self._trailing_silence_duration += chunk_duration

    def _reset_buffer(self):
        self._audio_buffer.clear()
        self._speech_seen = False
        self._trailing_silence_duration = 0.0

    def _buffer_duration(self) -> float:
        total_samples = sum(len(c) for c in self._audio_buffer)
        return total_samples / settings.sample_rate

    def _ws_sendable(self) -> bool:
        return self._is_running and self._client_connected

    async def _send_event(self, payload: dict):
        if not self._is_running:
            return
        await subtitle_hub.publish(payload)
        if not self._ws_sendable():
            return
        try:
            await self.ws.send_json(payload)
        except Exception:
            self._client_connected = False
            logger.exception("Failed to send websocket event")

    async def _process_buffer(self):
        if not self._audio_buffer:
            return

        audio = np.concatenate(self._audio_buffer)
        self._reset_buffer()
        audio_duration = len(audio) / settings.sample_rate

        self._segment_counter += 1
        seg_id = self._segment_counter

        timer = perf_monitor.create_timer()
        timer.start(audio_duration)

        loop = asyncio.get_running_loop()

        try:
            # Step 1: 音频预处理
            raw_audio = audio
            raw_rms = _audio_rms(raw_audio)
            audio = await loop.run_in_executor(
                None, audio_preprocessor.process, audio
            )
            timer.mark_preprocess_done()

            # Step 2: ASR 转写
            original = await loop.run_in_executor(
                None, asr_engine.transcribe, audio, self._source_lang,
            )
            timer.mark_asr_done()

            if not original.strip():
                timer.mark_mt_done()
                return

            if self._should_ignore_asr_text(original, audio_duration, raw_rms):
                logger.info(
                    "Ignoring likely ASR hallucination: %r (duration=%.2fs rms=%.4f)",
                    original,
                    audio_duration,
                    raw_rms,
                )
                timer.mark_mt_done()
                return

            if not self._ws_sendable():
                timer.mark_mt_done()
                return

            await self._send_event({
                "type": "original",
                "segment_id": seg_id,
                "text": original,
                "audio_duration_ms": round(audio_duration * 1000),
                "asr_ms": round(timer._metrics.asr_time * 1000),
                "source_lang": self._source_lang,
                "target_lang": self._target_lang,
            })

            # Step 3: MT 流式翻译 (generator 在 executor 中运行，避免阻塞事件循环)
            mt_start = time.perf_counter()
            token_queue: asyncio.Queue = asyncio.Queue()
            sentinel = object()

            session_glossary = self._glossary or None

            def _push_token(item):
                try:
                    loop.call_soon_threadsafe(token_queue.put_nowait, item)
                except RuntimeError:
                    pass

            def _run_stream():
                """在线程中运行同步 generator，将 token 放入队列"""
                try:
                    for token in mt_engine.translate_stream(
                        original,
                        self._source_lang,
                        self._target_lang,
                        glossary=session_glossary,
                    ):
                        _push_token(token)
                except Exception as exc:
                    _push_token(exc)
                finally:
                    _push_token(sentinel)

            executor_task = loop.run_in_executor(None, _run_stream)
            accumulated = []

            while True:
                token = await token_queue.get()
                if token is sentinel:
                    break
                if isinstance(token, Exception):
                    raise token
                accumulated.append(token)
                await self._send_event({
                    "type": "translated_partial",
                    "segment_id": seg_id,
                    "text": token,
                    "accumulated": "".join(accumulated),
                    "source_lang": self._source_lang,
                    "target_lang": self._target_lang,
                })

            await executor_task  # 确保线程完成
            full_text = "".join(accumulated)
            mt_time = time.perf_counter() - mt_start
            timer._metrics.mt_time = mt_time
            timer.mark_mt_done()

            await self._send_event({
                "type": "translated",
                "segment_id": seg_id,
                "text": full_text,
                "audio_duration_ms": round(audio_duration * 1000),
                "asr_ms": round(timer._metrics.asr_time * 1000),
                "mt_ms": round(mt_time * 1000),
                "total_ms": round(timer._metrics.total_time * 1000),
                "rtf": round(timer._metrics.rtf, 3),
                "source_lang": self._source_lang,
                "target_lang": self._target_lang,
            })

        except Exception as e:
            logger.exception("Processing error")
            if self._ws_sendable():
                try:
                    await self._send_event({
                        "type": "error",
                        "segment_id": seg_id,
                        "text": str(e),
                        "source_lang": self._source_lang,
                        "target_lang": self._target_lang,
                    })
                except Exception:
                    pass

    def _should_ignore_asr_text(self, text: str, audio_duration: float, raw_rms: float) -> bool:
        if not settings.asr_hallucination_filter_enabled:
            return False

        normalized = _normalize_asr_text(text)
        if not normalized:
            return True

        is_short_audio = audio_duration <= settings.asr_hallucination_max_duration
        is_quiet_audio = raw_rms <= settings.asr_hallucination_low_rms
        is_tiny_text = len(normalized) <= settings.asr_min_content_chars

        if is_tiny_text and is_quiet_audio:
            return True

        phrases = {
            _normalize_asr_text(phrase)
            for phrase in settings.asr_hallucination_phrases
            if phrase
        }
        if normalized in phrases and (is_short_audio or is_quiet_audio):
            return True

        return False

    def stop(self):
        self._is_running = False

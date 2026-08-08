import asyncio

import numpy as np

from app.asr.backends.base import TranscribeResult
from app.core.config import settings
from app.core.session import (
    VAD_FRAME_SAMPLES,
    StreamSession,
    _latency_mode_settings,
    _next_segment_id,
)


class _StubVAD:
    """记录每次收到的帧长，用来断言帧对齐。"""

    def __init__(self, speech: bool = True):
        self.speech = speech
        self.frame_sizes: list[int] = []
        self.resets = 0

    def detect_speech(self, frame, sr=16000):
        self.frame_sizes.append(len(frame))
        return self.speech

    def reset_states(self):
        self.resets += 1


def _session_with_vad(vad):
    session = StreamSession(websocket=None)
    session._vad = vad
    return session


def test_websocket_message_size_limit(monkeypatch):
    monkeypatch.setattr(settings, "websocket_max_message_bytes", 4)
    session = StreamSession(websocket=None)

    assert not session._message_too_large(b"1234")
    assert session._message_too_large(b"12345")
    assert session._message_too_large("你好")


def test_latency_mode_low_is_shorter_than_stable():
    low = _latency_mode_settings("low")
    stable = _latency_mode_settings("stable")

    assert low["segment_silence_duration_ms"] < stable["segment_silence_duration_ms"]
    assert low["segment_min_duration"] < stable["segment_min_duration"]


def test_vad_always_receives_exactly_one_frame_per_call():
    """回归测试：前端曾按 128 采样点发包，Silero 只接受 512，
    结果每一片都抛异常并被吞掉，VAD 实际上从未生效过。"""
    vad = _StubVAD()
    session = _session_with_vad(vad)

    session._update_voice_state(np.zeros(1024, dtype=np.float32))

    assert vad.frame_sizes == [VAD_FRAME_SAMPLES, VAD_FRAME_SAMPLES]
    assert session._vad_residual.size == 0


def test_unaligned_chunks_are_buffered_across_calls():
    """系统音频（ScreenCaptureKit）给的分片长度不保证是 512 的倍数。"""
    vad = _StubVAD()
    session = _session_with_vad(vad)

    session._update_voice_state(np.zeros(700, dtype=np.float32))
    assert vad.frame_sizes == [VAD_FRAME_SAMPLES]
    assert session._vad_residual.size == 700 - VAD_FRAME_SAMPLES

    session._update_voice_state(np.zeros(700, dtype=np.float32))
    assert all(size == VAD_FRAME_SAMPLES for size in vad.frame_sizes)
    assert session._vad_residual.size == (700 * 2) % VAD_FRAME_SAMPLES


def test_buffer_duration_tracks_samples_incrementally():
    session = StreamSession(websocket=None)
    assert session._buffer_duration() == 0.0

    for _ in range(5):
        chunk = np.zeros(1024, dtype=np.float32)
        session._audio_buffer.append(chunk)
        session._buffered_samples += chunk.size

    assert session._buffer_duration() == (5 * 1024) / settings.sample_rate

    session._reset_buffer()
    assert session._buffered_samples == 0
    assert session._buffer_duration() == 0.0


def test_trailing_silence_accumulates_only_after_speech():
    silent = _StubVAD(speech=False)
    session = _session_with_vad(silent)

    # 还没听到过语音，静音不该被算作"尾部静音"，否则会立刻误切一段
    session._update_voice_state(np.zeros(1024, dtype=np.float32))
    assert session._trailing_silence_duration == 0.0
    assert session._speech_seen is False

    session._vad = _StubVAD(speech=True)
    session._update_voice_state(np.zeros(1024, dtype=np.float32))
    assert session._speech_seen is True
    assert session._speech_duration > 0

    session._vad = silent
    session._update_voice_state(np.zeros(1024, dtype=np.float32))
    assert session._trailing_silence_duration > 0


def test_reset_buffer_resets_vad_rnn_state():
    """Silero 是 RNN，跨段沿用隐状态会污染新段开头的判断。"""
    vad = _StubVAD()
    session = _session_with_vad(vad)

    session._update_voice_state(np.zeros(1024, dtype=np.float32))
    session._reset_buffer()

    assert vad.resets == 1
    assert session._vad_residual.size == 0


def test_segment_ids_are_process_unique_across_sessions():
    first = _next_segment_id()
    second = _next_segment_id()

    assert second > first
    assert second <= 2**53 - 1


def test_graceful_shutdown_flushes_tail_audio_and_drains_queue(monkeypatch):
    session = StreamSession(websocket=None)
    session._is_running = True
    session._client_connected = True
    session._speech_seen = True
    session._speech_duration = 0.4
    session._audio_buffer.append(np.ones(4096, dtype=np.float32))
    session._buffered_samples = 4096
    processed = []

    async def _record(job):
        processed.append(job)

    monkeypatch.setattr(session, "_process_segment", _record)

    async def _run():
        session._processor_task = asyncio.create_task(session._process_segment_queue())
        await session._shutdown_pipeline(graceful=True)

    asyncio.run(_run())

    assert len(processed) == 1
    assert processed[0].audio.size == 4096
    assert session._audio_buffer == []


def test_short_quiet_qwen_okay_hallucination_is_suppressed():
    session = StreamSession(websocket=None)

    ignored, reason, speech_ratio = session._asr_filter_decision(
        "Okay.",
        audio_duration=1.0,
        raw_rms=0.0,
        speech_duration=0.1,
    )

    assert ignored is True
    assert reason == "known_phrase_short_audio"
    assert speech_ratio == 0.1


def test_real_spoken_okay_is_not_suppressed():
    session = StreamSession(websocket=None)

    ignored, reason, _ = session._asr_filter_decision(
        "Okay.",
        audio_duration=2.5,
        raw_rms=0.08,
        speech_duration=2.0,
    )

    assert ignored is False
    assert reason == ""


def test_streaming_preview_setting_is_captured_by_segment_job():
    session = StreamSession(websocket=None)
    session._streaming_preview_enabled = True
    session._speech_seen = True
    session._speech_duration = 1.25
    audio = np.full(20_000, 0.08, dtype=np.float32)
    session._audio_buffer.append(audio)
    session._buffered_samples = audio.size

    async def _run():
        await session._enqueue_current_buffer("test")
        job = session._segment_queue.get_nowait()
        session._segment_queue.task_done()
        return job

    job = asyncio.run(_run())

    assert job.streaming_preview is True
    assert np.array_equal(job.audio, audio)


def test_streaming_preview_uses_partial_text_from_same_asr_run(monkeypatch):
    session = StreamSession(websocket=None)
    session._is_running = True
    session._client_connected = True
    audio = np.ones(16_000, dtype=np.float32)
    events = []

    monkeypatch.setattr(settings, "asr_streaming_preview_min_interval_ms", 0)

    def _stream_once(value, language, on_partial):
        assert value is audio
        on_partial("hello", "English")
        on_partial("hello world", "English")
        return TranscribeResult(text="hello world", language="English")

    monkeypatch.setattr(
        "app.core.session.asr_engine.transcribe_stream_result",
        _stream_once,
    )

    async def _capture(payload, persist_history=None):
        events.append((payload, persist_history))

    monkeypatch.setattr(session, "_send_event", _capture)

    async def _run():
        session._streaming_preview_enabled = True
        session._speech_seen = True
        session._speech_duration = 1.0
        session._audio_buffer.append(audio)
        session._buffered_samples = audio.size
        await session._enqueue_current_buffer("test")
        job = session._segment_queue.get_nowait()
        session._segment_queue.task_done()
        return await session._run_asr(audio, job, asyncio.get_running_loop())

    result, preview_sent = asyncio.run(_run())

    assert result.text == "hello world"
    assert preview_sent is True
    assert events[-1][0]["type"] == "original_preview"
    assert events[-1][0]["text"] == "hello world"
    assert events[-1][1] is False

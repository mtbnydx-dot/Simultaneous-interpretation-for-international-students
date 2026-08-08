import numpy as np
import pytest

from app.core.vad import FRAME_SAMPLES, RMSVAD, SileroVAD, _resample_audio


def test_resample_audio_changes_sample_count():
    audio = np.linspace(-0.5, 0.5, 48000, dtype=np.float32)
    resampled = _resample_audio(audio, 48000, 16000)

    assert resampled.dtype == np.float32
    assert 15900 <= resampled.size <= 16100


def test_silero_frame_size_is_the_model_contract():
    # Silero v5/v6 在 16kHz 下只接受恰好 512 个采样点。这个常量被
    # StreamSession 用来做帧对齐，改了就会让 VAD 静默退化成 RMS。
    assert FRAME_SAMPLES == 512


def test_rms_vad_detects_loud_audio_and_ignores_silence():
    vad = RMSVAD()
    assert vad.detect_speech(np.full(FRAME_SAMPLES, 0.4, dtype=np.float32)) is True
    assert vad.detect_speech(np.zeros(FRAME_SAMPLES, dtype=np.float32)) is False
    assert vad.detect_speech(np.empty(0, dtype=np.float32)) is False


def test_silero_accepts_frame_sized_input_without_raising():
    """回归测试：以前前端按 128 采样点发包，Silero 每次都抛
    'Input audio chunk is too short'，被吞掉后整条 VAD 静默失效。"""
    onnxruntime = pytest.importorskip("onnxruntime")
    assert onnxruntime is not None

    vad = SileroVAD()
    try:
        vad._ensure_loaded()
    except Exception as exc:  # 没有权重/没有网络的环境跳过
        pytest.skip(f"Silero ONNX weights unavailable: {exc}")

    probability = vad.speech_probability(np.zeros(FRAME_SAMPLES, dtype=np.float32))
    assert 0.0 <= probability <= 1.0

    # 多帧输入按窗口切分，短于一帧的尾巴补零，都不应该抛
    assert 0.0 <= vad.speech_probability(np.zeros(FRAME_SAMPLES * 3, dtype=np.float32)) <= 1.0
    assert 0.0 <= vad.speech_probability(np.zeros(100, dtype=np.float32)) <= 1.0


def test_silero_reset_states_clears_rnn_context():
    pytest.importorskip("onnxruntime")
    vad = SileroVAD()
    try:
        vad._ensure_loaded()
    except Exception as exc:
        pytest.skip(f"Silero ONNX weights unavailable: {exc}")

    vad.speech_probability(np.full(FRAME_SAMPLES, 0.2, dtype=np.float32))
    assert vad._context.any()

    vad.reset_states()
    assert not vad._context.any()
    assert not vad._state.any()

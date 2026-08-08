"""
Qwen3-ASR 后端的选型与防呆逻辑。

运行时选的是 mlx-audio：mlx-community 为 Qwen3-ASR 发布了
4bit/5bit/6bit/8bit/bf16 全套 MLX 转换，mlx-audio 都能加载。
默认 8bit —— 实测比 bf16 快约 30%、少占约 1.6GB 内存，准确率没有下降。

（另一个候选 qwen3-asr-mlx 不依赖 transformers，但它完全不支持量化权重，
加载 8bit 会报 `Received 394 parameters not in model: ...scales/...biases`，
所以没有采用。）

这里挡住的坑：TRANS_ASR_MODEL_ID 是所有后端共用的，实际配置里经常指向
Whisper 仓库；把 Whisper 权重喂给 Qwen3 只会得到一堆张量名错误。
"""

import numpy as np
import pytest

from app.asr.backends import create_backend
from app.asr.backends.base import TranscribeResult
from app.asr.backends.qwen3_backend import (
    DEFAULT_QWEN3_MODEL_ID,
    Qwen3Backend,
    resolve_qwen3_model_id,
)
from app.core.config import settings
from app.asr.engine import ASREngine, _model_id_for_backend, unload_asr_engine


def test_backend_is_registered():
    assert isinstance(create_backend("qwen3"), Qwen3Backend)


def test_default_is_8bit():
    # 1.7B 本身就不大，为了识别率不用降到 4bit；8bit 又比 bf16 更快更省内存
    assert DEFAULT_QWEN3_MODEL_ID.endswith("-8bit")
    assert settings.asr_qwen3_model_id.endswith("-8bit")


@pytest.mark.parametrize("alias, expected", [
    ("8bit", "mlx-community/Qwen3-ASR-1.7B-8bit"),
    ("6bit", "mlx-community/Qwen3-ASR-1.7B-6bit"),
    ("5bit", "mlx-community/Qwen3-ASR-1.7B-5bit"),
    ("4bit", "mlx-community/Qwen3-ASR-1.7B-4bit"),
    ("bf16", "mlx-community/Qwen3-ASR-1.7B-bf16"),
    ("0.6B", "mlx-community/Qwen3-ASR-0.6B-bf16"),
    ("qwen3", DEFAULT_QWEN3_MODEL_ID),
])
def test_quantization_aliases_resolve(alias, expected):
    assert resolve_qwen3_model_id(alias) == expected


@pytest.mark.parametrize("repo", [
    "mlx-community/Qwen3-ASR-1.7B-8bit",
    "mlx-community/Qwen3-ASR-1.7B-4bit",
    "mlx-community/Qwen3-ASR-0.6B-bf16",
])
def test_explicit_qwen3_repos_pass_through(repo):
    """量化仓库不再被拒绝 —— mlx-audio 全都支持。"""
    assert resolve_qwen3_model_id(repo) == repo


def test_whisper_model_id_is_ignored_not_passed_through():
    resolved = resolve_qwen3_model_id("mlx-community/whisper-large-v3-turbo-8bit")
    assert resolved == settings.asr_qwen3_model_id
    assert "whisper" not in resolved.lower()


def test_none_falls_back_to_configured_default():
    assert resolve_qwen3_model_id(None) == settings.asr_qwen3_model_id


def test_transcribe_before_load_raises():
    import numpy as np

    with pytest.raises(RuntimeError, match="not loaded"):
        Qwen3Backend().transcribe(np.zeros(16000, dtype=np.float32), "en")


def test_batched_language_metadata_is_normalized_to_one_code():
    import numpy as np

    class Result:
        text = "你好"
        language = ["zh"]

    class Model:
        def generate(self, *_args, **_kwargs):
            return Result()

    backend = Qwen3Backend()
    backend._model = Model()
    result = backend.transcribe(np.zeros(16000, dtype=np.float32), "auto")

    assert result.language == "zh"


def test_streaming_decode_reuses_one_authoritative_generation(monkeypatch):
    calls = []

    class Emission:
        def __init__(self, text, language="en"):
            self.text = text
            self.language = language

    class Model:
        def generate(self, *_args, **kwargs):
            calls.append(kwargs)
            return iter([Emission("hello"), Emission(" world"), Emission("")])

    monkeypatch.setattr(settings, "asr_max_new_tokens", 96)
    backend = Qwen3Backend()
    backend._model = Model()
    partials = []

    result = backend.transcribe_stream(
        np.zeros(16000, dtype=np.float32),
        "en",
        lambda text, language: partials.append((text, language)),
    )

    assert result.text == "hello world"
    assert partials == [("hello", "en"), ("hello world", "en")]
    assert len(calls) == 1
    assert calls[0]["stream"] is True
    assert calls[0]["max_tokens"] == 96


def test_unload_clears_model_metadata():
    backend = Qwen3Backend()
    backend._model = object()
    backend._model_id = DEFAULT_QWEN3_MODEL_ID
    backend._model_path = "/tmp/qwen3"

    backend.unload()

    assert backend.is_loaded is False
    assert backend.model_id is None
    assert backend.model_path is None


def test_fallback_backends_receive_compatible_model_ids(monkeypatch):
    qwen = "mlx-community/Qwen3-ASR-1.7B-8bit"
    whisper = "mlx-community/whisper-large-v3-turbo"
    monkeypatch.setattr(settings, "asr_qwen3_model_id", qwen)
    monkeypatch.setattr(settings, "asr_mlx_fallback_model_id", whisper)
    monkeypatch.setattr(settings, "asr_model_id", None)

    assert _model_id_for_backend("qwen3", qwen) == qwen
    assert _model_id_for_backend("mlx", qwen) == whisper
    assert "qwen" not in (_model_id_for_backend("ct2", qwen) or "").lower()
    assert "qwen" not in (_model_id_for_backend("transformers-whisper", qwen) or "").lower()


def test_partially_loaded_backend_is_unloaded_before_fallback(monkeypatch):
    class FailingBackend:
        is_loaded = False

        def __init__(self):
            self.unloaded = False

        def load(self, **_kwargs):
            raise RuntimeError("broken weights")

        def unload(self):
            self.unloaded = True

    backend = FailingBackend()
    monkeypatch.setattr("app.asr.backends.create_backend", lambda _name: backend)

    with pytest.raises(RuntimeError, match="broken weights"):
        ASREngine()._try_load("qwen3", "mlx", "8bit", DEFAULT_QWEN3_MODEL_ID)

    assert backend.unloaded is True


def test_unload_wrapper_collects_runtime_memory_after_backend_returns(monkeypatch):
    class Backend:
        is_loaded = True

        def __init__(self):
            self.unloaded = False

        def unload(self):
            self.unloaded = True

    backend = Backend()
    cleanup_calls = []
    engine = ASREngine()
    engine._backend = backend
    monkeypatch.setattr("app.asr.engine.collect_model_memory", lambda: cleanup_calls.append(True))

    unload_asr_engine(engine)

    assert backend.unloaded is True
    assert cleanup_calls == [True]


def test_engine_routes_streaming_decode_through_backend_once():
    engine = ASREngine()
    calls = []

    class Backend:
        is_loaded = True

        def transcribe_stream(self, audio, language, on_partial):
            calls.append((audio.size, language))
            on_partial("draft", language)
            return TranscribeResult(text="final", language=language)

    engine._backend = Backend()
    partials = []
    result = engine.transcribe_stream_result(
        np.zeros(1600, dtype=np.float32),
        "en",
        lambda text, language: partials.append((text, language)),
    )

    assert result.text == "final"
    assert calls == [(1600, "en")]
    assert partials == [("draft", "en")]

"""
Qwen3-ASR 后端（Apple Silicon / MLX），通过 mlx-audio 运行。

模型：Qwen/Qwen3-ASR（Apache-2.0，52 种语言和方言），由 mlx-community 转成
MLX 格式，提供 4bit / 5bit / 6bit / 8bit / bf16 全套量化。本应用支持的
19 种语言全部在覆盖范围内。

默认用 **8bit**：实测比 bf16 更快、内存少 1.6GB，准确率没有下降
（同一组音频：英文 15dB 噪声下 8.3% vs bf16 10.0%，其余持平）。
1.7B 本身就不大，没必要为了省内存去用 4bit 牺牲识别率。

运行时选型说明：另一个候选 qwen3-asr-mlx 不依赖 transformers，但它
**完全不支持量化权重**（加载 8bit 会报 `Received 394 parameters not in
model: ...scales/...biases`），只能跑 bf16。mlx-audio 支持全套量化，
代价是会带上 transformers + mlx_lm（用于 tokenizer 和特征提取），
但不需要 torch。
"""

import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np

from app.asr.backends.base import ASRBackend, TranscribeResult
from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_QWEN3_MODEL_ID = "mlx-community/Qwen3-ASR-1.7B-8bit"
_QWEN3_ALIASES = {
    "qwen3": DEFAULT_QWEN3_MODEL_ID,
    "qwen3-asr": DEFAULT_QWEN3_MODEL_ID,
    "1.7b": DEFAULT_QWEN3_MODEL_ID,
    "8bit": DEFAULT_QWEN3_MODEL_ID,
    "6bit": "mlx-community/Qwen3-ASR-1.7B-6bit",
    "5bit": "mlx-community/Qwen3-ASR-1.7B-5bit",
    "4bit": "mlx-community/Qwen3-ASR-1.7B-4bit",
    "bf16": "mlx-community/Qwen3-ASR-1.7B-bf16",
    "0.6b": "mlx-community/Qwen3-ASR-0.6B-bf16",
}


def resolve_qwen3_model_id(model_id: str | None = None) -> str:
    """
    解析 Qwen3-ASR 的模型 id。

    注意 TRANS_ASR_MODEL_ID 是所有后端共用的，实际配置里经常指向 Whisper 仓库。
    把 Whisper 权重喂给 Qwen3 只会得到一堆看不懂的张量错误，所以这里只接受
    明显是 Qwen3 的 id，其余一律忽略并回落到 asr_qwen3_model_id。
    """
    raw = (model_id or "").strip()
    if raw:
        alias = _QWEN3_ALIASES.get(raw.lower())
        if alias:
            return alias
        if "qwen3-asr" in raw.lower():
            return raw
        logger.warning(
            "Ignoring asr_model_id=%r for the Qwen3 backend (not a Qwen3-ASR model); using %s",
            raw,
            settings.asr_qwen3_model_id,
        )
    configured = (settings.asr_qwen3_model_id or DEFAULT_QWEN3_MODEL_ID).strip()
    return _QWEN3_ALIASES.get(configured.lower(), configured)


def prepare_qwen3_model_path(model_id: str | None = None) -> str:
    resolved = resolve_qwen3_model_id(model_id)
    local_path = Path(resolved).expanduser()
    downloaded = not local_path.exists()
    if downloaded:
        from app.core.model_download import download_hf_snapshot

        local_path = Path(download_hf_snapshot(resolved))

    try:
        _validate_qwen3_model_path(local_path)
    except Exception:
        if not downloaded:
            raise
        from app.core.model_download import download_hf_snapshot

        logger.exception("Cached Qwen3-ASR model is invalid; forcing a fresh download: %s", resolved)
        local_path = Path(download_hf_snapshot(resolved, force=True))
        _validate_qwen3_model_path(local_path)
    return str(local_path)


def _validate_qwen3_model_path(model_path: Path) -> None:
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise RuntimeError(f"Qwen3-ASR model config is missing: {model_path}")
    weights = model_path / "weights.safetensors"
    alternate = model_path / "model.safetensors"
    candidate = weights if weights.is_file() else alternate
    if not candidate.is_file() or candidate.stat().st_size < 100 * 1024 * 1024:
        raise RuntimeError(f"Qwen3-ASR model weights are missing or incomplete: {model_path}")


class Qwen3Backend(ASRBackend):
    """Qwen3-ASR via mlx-audio。"""

    def __init__(self):
        self._model = None
        self._model_id: str | None = None
        self._model_path: str | None = None

    def load(self, device: str, compute_type: str, model_id: str | None = None) -> None:
        if device not in ("mlx", "mps", "metal", "auto"):
            logger.warning("Qwen3 backend ignores unsupported device=%s and uses Apple Metal", device)

        resolved = resolve_qwen3_model_id(model_id)
        model_path = prepare_qwen3_model_path(resolved)

        try:
            from mlx_audio.stt.generate import load_model
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-ASR 需要 mlx-audio。请重新安装应用或运行：pip install mlx-audio。"
            ) from exc

        logger.info("Loading Qwen3-ASR model: %s (%s)", resolved, model_path)
        self._model = load_model(model_path)
        self._model_id = resolved
        self._model_path = model_path
        logger.info("Qwen3-ASR model loaded successfully")

        if settings.asr_mlx_warmup:
            self._warmup()

    def _warmup(self) -> None:
        try:
            self._transcribe_raw(np.zeros(16000, dtype=np.float32), "en")
            logger.info("Qwen3-ASR warmup completed")
        except Exception as exc:
            logger.warning("Qwen3-ASR warmup failed: %s", exc)

    def _transcribe_raw(
        self,
        audio: np.ndarray,
        language: str | None,
        max_tokens: int | None = None,
    ):
        token_limit = settings.asr_max_new_tokens if max_tokens is None else max_tokens
        return self._model.generate(
            audio.astype(np.float32, copy=False),
            language=language,
            temperature=0.0,
            # 同传是短片段，没必要留 8192 的默认上限
            max_tokens=max(8, int(token_limit)) if token_limit else 8192,
        )

    def _transcribe_with_limit(
        self,
        audio: np.ndarray,
        language: str | None,
        max_tokens: int | None,
    ) -> TranscribeResult:
        if self._model is None:
            raise RuntimeError("Qwen3-ASR model not loaded")

        t0 = time.perf_counter()
        audio_duration = len(audio) / settings.sample_rate
        lang = language or settings.source_lang
        lang = None if not lang or lang == "auto" else lang

        result = self._transcribe_raw(audio, lang, max_tokens=max_tokens)
        text = getattr(result, "text", None)
        if text is None:
            text = str(result)
        detected_language = getattr(result, "language", None)
        if isinstance(detected_language, (list, tuple)):
            detected_language = next(
                (item for item in detected_language if isinstance(item, str) and item.strip()),
                None,
            )
        if not isinstance(detected_language, str) or not detected_language.strip():
            detected_language = lang

        return TranscribeResult(
            text=text.strip(),
            language=detected_language,
            duration=audio_duration,
            process_time=time.perf_counter() - t0,
        )

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> TranscribeResult:
        return self._transcribe_with_limit(audio, language, settings.asr_max_new_tokens)

    def transcribe_stream(
        self,
        audio: np.ndarray,
        language: str | None,
        on_partial: Callable[[str, str | None], None],
    ) -> TranscribeResult:
        if self._model is None:
            raise RuntimeError("Qwen3-ASR model not loaded")

        started_at = time.perf_counter()
        audio_duration = len(audio) / settings.sample_rate
        lang = language or settings.source_lang
        lang = None if not lang or lang == "auto" else lang
        token_limit = settings.asr_max_new_tokens
        stream = self._model.generate(
            audio.astype(np.float32, copy=False),
            language=lang,
            temperature=0.0,
            max_tokens=max(8, int(token_limit)) if token_limit else 8192,
            stream=True,
        )

        parts: list[str] = []
        detected_language = lang
        for emission in stream:
            candidate_language = getattr(emission, "language", None)
            if isinstance(candidate_language, str) and candidate_language.strip():
                detected_language = candidate_language
            token = str(getattr(emission, "text", "") or "")
            if not token:
                continue
            parts.append(token)
            on_partial("".join(parts), detected_language)

        return TranscribeResult(
            text="".join(parts).strip(),
            language=detected_language,
            duration=audio_duration,
            process_time=time.perf_counter() - started_at,
        )

    def unload(self) -> None:
        self._model = None
        self._model_id = None
        self._model_path = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def model_path(self) -> str | None:
        return self._model_path

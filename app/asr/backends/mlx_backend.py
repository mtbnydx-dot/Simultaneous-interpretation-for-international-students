"""
Apple Silicon MLX ASR 后端。

使用 mlx-whisper 跑 MLX 格式 Whisper 模型。默认模型为
mlx-community/whisper-large-v3-turbo-8bit；该仓库使用 model.safetensors，
这里会在本地缓存中创建 weights.safetensors 兼容链接以复用 mlx-whisper。
"""

import logging
import os
import shutil
import struct
import time
import zipfile
from pathlib import Path

import numpy as np

from app.asr.backends.base import ASRBackend, TranscribeResult
from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_MLX_MODEL_ID = "mlx-community/whisper-large-v3-turbo-8bit"
_MLX_MODEL_ALIASES = {
    "large-v3-turbo": DEFAULT_MLX_MODEL_ID,
    "whisper-large-v3-turbo": DEFAULT_MLX_MODEL_ID,
    "large-v3-turbo-8bit": DEFAULT_MLX_MODEL_ID,
    "whisper-large-v3-turbo-8bit": DEFAULT_MLX_MODEL_ID,
}


def resolve_mlx_model_id(model_id: str | None = None) -> str:
    raw = (model_id or settings.asr_model_id or DEFAULT_MLX_MODEL_ID).strip()
    return _MLX_MODEL_ALIASES.get(raw, raw)


def prepare_mlx_model_path(model_id: str | None = None) -> str:
    resolved = resolve_mlx_model_id(model_id)
    local_path = Path(resolved).expanduser()

    if local_path.exists():
        model_path = local_path
    else:
        from app.core.model_download import download_hf_snapshot

        logger.info("Downloading MLX ASR model: %s", resolved)
        model_path = Path(download_hf_snapshot(resolved))

    try:
        _prepare_weights_alias(model_path)
        _validate_mlx_model_path(model_path)
    except Exception:
        if local_path.exists():
            raise
        from app.core.model_download import download_hf_snapshot

        logger.exception("Cached MLX ASR model is invalid; forcing a fresh download: %s", resolved)
        model_path = Path(download_hf_snapshot(resolved, force=True))
        _prepare_weights_alias(model_path)
        _validate_mlx_model_path(model_path)

    return str(model_path)


def _prepare_weights_alias(model_path: Path) -> None:
    weights_path = model_path / "weights.safetensors"
    alt_weights_path = model_path / "model.safetensors"
    if weights_path.exists() or not alt_weights_path.exists():
        return
    try:
        weights_path.symlink_to(alt_weights_path.name)
        logger.info("Created MLX weights compatibility symlink: %s", weights_path)
    except FileExistsError:
        pass
    except OSError:
        try:
            os.link(alt_weights_path, weights_path)
            logger.info("Created MLX weights compatibility hardlink: %s", weights_path)
        except OSError:
            logger.warning(
                "Could not link %s; copying to %s instead",
                alt_weights_path.name,
                weights_path.name,
            )
            shutil.copy2(alt_weights_path, weights_path)


def _validate_safetensors(path: Path) -> None:
    if path.stat().st_size < 10 * 1024 * 1024:
        raise RuntimeError(f"MLX weights look incomplete: {path}")
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise RuntimeError(f"MLX weights header is incomplete: {path}")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len <= 0 or header_len > 64 * 1024 * 1024:
            raise RuntimeError(f"MLX safetensors header is invalid: {path}")
        header = f.read(min(header_len, 512))
    if not header.lstrip().startswith(b"{"):
        raise RuntimeError(f"MLX safetensors header is not JSON: {path}")


def _validate_mlx_model_path(model_path: Path) -> None:
    if not model_path.is_dir():
        raise RuntimeError(f"MLX model path is not a directory: {model_path}")
    if not (model_path / "config.json").is_file():
        raise RuntimeError(f"MLX model config.json is missing: {model_path}")

    safetensors_path = model_path / "weights.safetensors"
    npz_path = model_path / "weights.npz"
    if safetensors_path.is_file():
        _validate_safetensors(safetensors_path)
        return
    if npz_path.is_file():
        if not zipfile.is_zipfile(npz_path):
            raise RuntimeError(
                f"MLX weights.npz is not a valid zip/npz file: {npz_path}. "
                "Delete the cached model and download again."
            )
        return
    raise RuntimeError(f"MLX model weights are missing under {model_path}")


class MLXBackend(ASRBackend):
    """mlx-whisper 后端，适合 Apple Silicon 低延迟识别。"""

    def __init__(self):
        self._model = None
        self._model_path: str | None = None
        self._model_id: str | None = None
        self._fp16 = True

    def load(self, device: str, compute_type: str, model_id: str | None = None) -> None:
        if device not in ("mlx", "mps", "metal", "auto"):
            logger.warning("MLX backend ignores unsupported device=%s and uses Apple Metal", device)

        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        self._fp16 = compute_type != "float32"
        dtype = mx.float16 if self._fp16 else mx.float32
        self._model_id = resolve_mlx_model_id(model_id)
        self._model_path = prepare_mlx_model_path(self._model_id)

        logger.info(
            "Loading MLX ASR model: %s (%s, dtype=%s)",
            self._model_id,
            self._model_path,
            "float16" if self._fp16 else "float32",
        )
        try:
            self._model = ModelHolder.get_model(self._model_path, dtype)
        except Exception:
            fallback = settings.asr_mlx_fallback_model_id
            if not fallback or fallback == self._model_id:
                raise
            logger.exception(
                "MLX model %s failed to load; falling back to %s",
                self._model_id,
                fallback,
            )
            self._model_id = resolve_mlx_model_id(fallback)
            self._model_path = prepare_mlx_model_path(self._model_id)
            self._model = ModelHolder.get_model(self._model_path, dtype)

        logger.info("MLX ASR model loaded successfully")
        if settings.asr_mlx_warmup:
            self._warmup()

    def _warmup(self) -> None:
        try:
            audio = np.zeros(16000, dtype=np.float32)
            self._transcribe_raw(audio, language="en", sample_len=8)
            logger.info("MLX ASR warmup completed")
        except Exception as exc:
            logger.warning("MLX ASR warmup failed: %s", exc)

    def _temperature(self) -> float | tuple[float, ...]:
        values = settings.asr_temperature or [0.0]
        if len(values) == 1:
            return float(values[0])
        return tuple(float(v) for v in values)

    def _sample_len(self, audio_duration: float, override: int | None = None) -> int:
        if override is not None:
            return max(1, int(override))
        configured = max(16, int(settings.asr_max_new_tokens or 96))
        # Whisper 的 token 上限对短片段也有固定成本；同传更适合短上限。
        return min(configured, max(24, int(audio_duration * 18) + 24))

    def _transcribe_raw(
        self,
        audio: np.ndarray,
        language: str | None,
        sample_len: int | None = None,
    ) -> dict:
        if self._model_path is None:
            raise RuntimeError("MLX model path is not initialized")

        import mlx_whisper

        lang = None if not language or language == "auto" else language
        kwargs = {
            "path_or_hf_repo": self._model_path,
            "verbose": False,
            "temperature": self._temperature(),
            "compression_ratio_threshold": settings.asr_compression_ratio_threshold,
            "logprob_threshold": settings.asr_logprob_threshold,
            "no_speech_threshold": settings.asr_no_speech_threshold,
            "condition_on_previous_text": settings.asr_condition_on_prev_tokens,
            "word_timestamps": False,
            "task": "transcribe",
            "fp16": self._fp16,
            "without_timestamps": True,
            "sample_len": self._sample_len(len(audio) / settings.sample_rate, sample_len),
        }
        if lang:
            kwargs["language"] = lang
        if settings.asr_beam_size and settings.asr_beam_size > 1:
            kwargs["beam_size"] = int(settings.asr_beam_size)

        return mlx_whisper.transcribe(audio.astype(np.float32, copy=False), **kwargs)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> TranscribeResult:
        if self._model is None:
            raise RuntimeError("MLX model not loaded")

        t0 = time.perf_counter()
        audio_duration = len(audio) / settings.sample_rate
        lang = language or settings.source_lang

        result = self._transcribe_raw(audio, language=lang)
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        elapsed = time.perf_counter() - t0

        return TranscribeResult(
            text=text.strip(),
            language=lang,
            duration=audio_duration,
            process_time=elapsed,
        )

    def unload(self) -> None:
        try:
            from mlx_whisper.transcribe import ModelHolder

            if self._model_path and ModelHolder.model_path == self._model_path:
                ModelHolder.model = None
                ModelHolder.model_path = None
        except Exception:
            pass
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

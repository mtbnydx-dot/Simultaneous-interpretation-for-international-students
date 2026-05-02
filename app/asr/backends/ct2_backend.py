"""
faster-whisper (CTranslate2) ASR 后端。
支持 NVIDIA GPU (CUDA) 和 CPU。
"""

import time
import logging
import os
import numpy as np

from app.asr.backends.base import ASRBackend, TranscribeResult
from app.core.config import settings

logger = logging.getLogger(__name__)


class CT2Backend(ASRBackend):
    """faster-whisper (CTranslate2) 后端"""

    def __init__(self):
        self._model = None

    def load(self, device: str, compute_type: str, model_id: str | None = None) -> None:
        from faster_whisper import WhisperModel

        raw = model_id or settings.asr_model_id or settings.asr_model_size

        # 将 HF 模型 ID 转换为 faster-whisper 的 size 字符串
        # 例如 "distil-whisper/distil-medium.en" → "distil-medium.en"
        if "/" in raw:
            raw = raw.split("/")[-1]
            # "whisper-large-v3" → "large-v3"
            if raw.startswith("whisper-"):
                raw = raw[len("whisper-"):]
        model_size = raw

        logger.info("Loading CT2 model: %s (device=%s, compute_type=%s)",
                     model_size, device, compute_type)

        cpu_threads = int(settings.asr_cpu_threads or 0)
        if cpu_threads <= 0 and device == "cpu":
            # 留一部分核心给 MT 和 UI，避免实时场景下整机被 ASR 吃满。
            cpu_threads = max(1, min(8, (os.cpu_count() or 4) - 2))

        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        logger.info("CT2 model loaded successfully")

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> TranscribeResult:
        if self._model is None:
            raise RuntimeError("CT2 model not loaded")

        t0 = time.perf_counter()
        audio_duration = len(audio) / settings.sample_rate
        lang = language or settings.source_lang

        kwargs = {
            "language": lang,
            "vad_filter": settings.asr_ct2_vad_filter,
            "beam_size": max(1, int(settings.asr_beam_size or 1)),
            "best_of": max(1, int(settings.asr_best_of or 1)),
            "condition_on_previous_text": settings.asr_condition_on_prev_tokens,
            "temperature": tuple(settings.asr_temperature or [0.0]),
            "compression_ratio_threshold": settings.asr_compression_ratio_threshold,
            "log_prob_threshold": settings.asr_logprob_threshold,
            "no_speech_threshold": settings.asr_no_speech_threshold,
            "max_new_tokens": settings.asr_max_new_tokens,
        }
        if settings.asr_ct2_vad_filter:
            kwargs["vad_parameters"] = dict(
                min_silence_duration_ms=settings.silence_duration_ms,
                threshold=settings.vad_threshold,
            )

        segments, info = self._model.transcribe(audio, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments)
        elapsed = time.perf_counter() - t0

        return TranscribeResult(
            text=text,
            language=lang,
            duration=audio_duration,
            process_time=elapsed,
        )

    def unload(self) -> None:
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

"""Remote ASR backend for future cloud inference deployments."""

from __future__ import annotations

import time

import numpy as np

from app.asr.backends.base import ASRBackend, TranscribeResult
from app.core.config import settings
from app.core import cloud_client


class CloudASRBackend(ASRBackend):
    def __init__(self) -> None:
        self._loaded = False
        self._model_id: str | None = None

    def load(self, device: str, compute_type: str, model_id: str | None = None) -> None:
        # Remote backends do not allocate a local model. We still validate config
        # at load time so health/download flows fail early and clearly.
        if not settings.cloud_enabled or not settings.cloud_base_url:
            raise RuntimeError("Cloud ASR requires TRANS_CLOUD_ENABLED=true and TRANS_CLOUD_BASE_URL")
        self._model_id = model_id or settings.cloud_asr_model
        self._loaded = True

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> TranscribeResult:
        if not self._loaded:
            raise RuntimeError("Cloud ASR backend is not loaded")
        started = time.perf_counter()
        result = cloud_client.transcribe(audio, language=language)
        elapsed = time.perf_counter() - started
        text = result.get("text") or result.get("transcript") or ""
        return TranscribeResult(
            text=str(text).strip(),
            language=result.get("language") or language,
            duration=len(audio) / settings.sample_rate,
            process_time=elapsed,
        )

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_id(self) -> str | None:
        return self._model_id

"""
VAD (Voice Activity Detection) 模块。
支持 Silero VAD (ONNX) 和 RMS 回退。
"""

import logging
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class SileroVAD:
    """Silero VAD 封装，ONNX 推理，轻量精确"""

    def __init__(self):
        self._model = None
        self._sample_rate = 16000
        self._loaded = False
        self._last_sr: int | None = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir=settings.vad_silero_repo,
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            (self._get_speech_timestamps, _, _, _, _) = utils
            self._loaded = True
            logger.info("Silero VAD loaded successfully")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Silero VAD: {exc}. "
                "Install: pip install torch. Or set TRANS_VAD_ENGINE=rms for fallback."
            ) from exc

    def detect_speech(self, audio: np.ndarray, sr: int = 16000) -> bool:
        """检测音频是否包含语音"""
        self._ensure_loaded()
        import torch

        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        audio_tensor = torch.from_numpy(audio).float()
        speech_prob = self._model(audio_tensor, sr).item()
        return speech_prob >= settings.vad_silero_threshold

    def get_speech_segments(
        self, audio: np.ndarray, sr: int = 16000
    ) -> list[dict]:
        """返回语音段列表 [{start_ms, end_ms}, ...]"""
        self._ensure_loaded()
        import torch

        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        audio_tensor = torch.from_numpy(audio).float()
        segments = self._get_speech_timestamps(
            audio_tensor,
            self._model,
            threshold=settings.vad_silero_threshold,
            min_speech_duration_ms=settings.vad_silero_min_speech_ms,
            min_silence_duration_ms=settings.segment_silence_duration_ms,
        )
        return segments


class RMSVAD:
    """简单 RMS 阈值 VAD (回退方案)"""

    def detect_speech(self, audio: np.ndarray, sr: int = 16000) -> bool:
        if len(audio) == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(audio))))
        return rms >= settings.audio_vad_rms_threshold

    def get_speech_segments(
        self, audio: np.ndarray, sr: int = 16000
    ) -> list[dict]:
        # RMS VAD 不做精细分段，返回整段
        return [{"start_ms": 0, "end_ms": int(len(audio) / sr * 1000)}]


_vad_instance: SileroVAD | RMSVAD | None = None


def get_vad():
    global _vad_instance
    if _vad_instance is None:
        if settings.vad_engine == "silero":
            try:
                _vad_instance = SileroVAD()
                _vad_instance._ensure_loaded()
            except Exception as exc:
                logger.warning("Silero VAD unavailable, falling back to RMS: %s", exc)
                _vad_instance = RMSVAD()
        else:
            _vad_instance = RMSVAD()
    return _vad_instance

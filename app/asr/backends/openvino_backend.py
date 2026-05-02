"""
Intel OpenVINO ASR 后端。
支持 Intel GPU 加速。
"""

import os
import time
import logging
from copy import deepcopy
import numpy as np

# 必须在 import optimum.intel / nncf 之前设置，避免 Rich 库写 Windows GBK 控制台时崩溃
os.environ.setdefault("NNCF_PROGRESS_BAR", "false")
os.environ.setdefault("RICH_FORCE_TERMINAL", "false")
os.environ.setdefault("PROGRESS_BAR", "off")

from app.asr.backends.base import ASRBackend, TranscribeResult
from app.core.config import settings

logger = logging.getLogger(__name__)


class OpenVINOBackend(ASRBackend):
    """Intel OpenVINO 后端"""

    def __init__(self):
        self._model = None
        self._processor = None

    def load(self, device: str, compute_type: str, model_id: str | None = None) -> None:
        from optimum.intel import OVModelForSpeechSeq2Seq
        from transformers import WhisperProcessor

        # 支持完整 HF 模型 ID（含 "/"，例如 distil-whisper/distil-medium.en）
        # 兼容旧用法：纯 size 字符串自动展开为 openai/whisper-{size}
        raw = model_id or settings.asr_model_id or settings.asr_model_size
        model_id_str = raw if "/" in raw else f"openai/whisper-{raw}"

        device_map = {
            "intel_gpu": "GPU", "gpu": "GPU",
            "intel_npu": "NPU", "npu": "NPU",
            "cpu": "CPU", "auto": "AUTO",
        }
        ov_device = device_map.get(device.lower(), "CPU") if isinstance(device, str) else "CPU"

        logger.info("Loading OpenVINO model: %s (device=%s)", model_id_str, ov_device)

        self._model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id_str,
            export=True,
            device=ov_device,
            load_in_8bit=True,
        )
        self._processor = WhisperProcessor.from_pretrained(model_id_str)

        # 清除预训练 config 中的 forced_decoder_ids，避免 deprecated 警告
        if hasattr(self._model.generation_config, "forced_decoder_ids"):
            self._model.generation_config.forced_decoder_ids = None

        logger.info("OpenVINO model loaded successfully")

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> TranscribeResult:
        if self._model is None or self._processor is None:
            raise RuntimeError("OpenVINO model not loaded")

        import torch

        t0 = time.perf_counter()
        audio_duration = len(audio) / settings.sample_rate
        lang = language or settings.source_lang

        forced_decoder_ids = self._processor.get_decoder_prompt_ids(
            language=lang, task="transcribe"
        )

        input_features = self._processor(
            audio, sampling_rate=settings.sample_rate, return_tensors="pt"
        ).input_features

        max_target_positions = getattr(self._model.config, "max_target_positions", 448)
        prompt_len = len(forced_decoder_ids) if forced_decoder_ids else 0
        available_tokens = max_target_positions - prompt_len - 1
        max_new_tokens = min(settings.asr_max_new_tokens, available_tokens)
        if max_new_tokens < 1:
            raise RuntimeError(
                "Whisper decoder prompt leaves no room for generated tokens "
                f"(max_target_positions={max_target_positions}, prompt_len={prompt_len})"
            )

        generation_config = deepcopy(self._model.generation_config)
        generation_config.max_new_tokens = max_new_tokens
        generation_config.max_length = max_target_positions
        # 将 forced_decoder_ids 写入 generation_config，而非作为 generate() 参数
        # OpenVINO 的 GenerationConfig 没有 lang_to_id，不支持 language/task 新式参数
        generation_config.forced_decoder_ids = forced_decoder_ids

        predicted_ids = self._model.generate(
            input_features,
            generation_config=generation_config,
        )

        result = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        text = " ".join(s.strip() for s in result if s.strip())
        elapsed = time.perf_counter() - t0

        return TranscribeResult(
            text=text,
            language=lang,
            duration=audio_duration,
            process_time=elapsed,
        )

    def unload(self) -> None:
        self._model = None
        self._processor = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

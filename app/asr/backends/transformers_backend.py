"""
HuggingFace Transformers ASR 后端。
支持 Distil-Whisper 和标准 Whisper 模型。
兼容 CUDA / MPS / CPU。
"""

import time
import logging
import numpy as np

from app.asr.backends.base import ASRBackend, TranscribeResult
from app.core.config import settings

logger = logging.getLogger(__name__)

# Distil-Whisper 是英文识别取向；多语言同传默认通过 TRANS_ASR_MODEL_ID
# 使用 openai/whisper-large-v3-turbo。
DISTIL_WHISPER_MODELS = {
    "medium": "distil-whisper/distil-medium.en",
    "large-v3": "distil-whisper/distil-large-v3",
}


class TransformersBackend(ASRBackend):
    """
    HuggingFace Transformers ASR 后端。

    支持:
    - Distil-Whisper (推荐，比标准 Whisper 快 6 倍)
    - 标准 OpenAI Whisper
    - CUDA / MPS / CPU 设备
    """

    def __init__(self):
        self._pipe = None
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._torch_dtype = None
        self._model_id = None

    def load(self, device: str, compute_type: str, model_id: str | None = None) -> None:
        import torch
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )

        # 解析模型 ID
        resolved_model_id = self._resolve_model_id(model_id)
        self._model_id = resolved_model_id

        # 确定 torch dtype
        torch_dtype = self._resolve_torch_dtype(device, compute_type)

        # 确定设备字符串
        device_str = self._resolve_device_str(device)

        logger.info("Loading transformers ASR model: %s (device=%s, dtype=%s)",
                     resolved_model_id, device_str, torch_dtype)

        # 加载模型
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            resolved_model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device_str)
        model.eval()

        # 清除预训练 config 中的 forced_decoder_ids，避免 deprecated 警告
        if hasattr(model.generation_config, "forced_decoder_ids"):
            model.generation_config.forced_decoder_ids = None

        # 加载处理器
        processor = AutoProcessor.from_pretrained(resolved_model_id)

        # 构建 pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device_str,
            chunk_length_s=30,          # 30s chunks (Whisper 训练长度)
            stride_length_s=[6, 0],     # 6s overlap，减少边界截断
            return_timestamps=False,    # 实时翻译不需要时间戳
        )

        self._pipe = pipe
        self._model = model
        self._processor = processor
        self._device = device_str
        self._torch_dtype = torch_dtype

        logger.info("Transformers ASR model loaded successfully")

        # MPS warmup: 首次推理需要 JIT 编译
        if device_str == "mps":
            logger.info("Performing MPS warmup inference...")
            self._warmup()

    def _resolve_model_id(self, model_id: str | None) -> str:
        """解析模型 ID"""
        if model_id:
            if "/" not in model_id and not model_id.startswith("distil-"):
                return f"openai/whisper-{model_id}"
            return model_id

        if settings.asr_backend == "transformers-distil":
            return DISTIL_WHISPER_MODELS.get(
                settings.asr_model_size,
                DISTIL_WHISPER_MODELS["large-v3"],
            )

        return f"openai/whisper-{settings.asr_model_size}"

    def _resolve_torch_dtype(self, device: str, compute_type: str):
        """确定 torch dtype"""
        import torch

        if compute_type == "float16":
            return torch.float16
        elif compute_type == "bfloat16":
            return torch.bfloat16
        elif compute_type == "float32":
            return torch.float32

        # auto
        if device == "cuda":
            return torch.float16
        elif device == "mps":
            return torch.float32   # MPS: float32 最稳定
        else:
            return torch.float32   # CPU

    def _resolve_device_str(self, device: str) -> str:
        """确定设备字符串"""
        if device == "mps":
            return "mps"
        elif device == "cuda":
            return "cuda:0"
        elif device == "intel_gpu":
            # Intel GPU 不支持 transformers，使用 CPU
            logger.info("Intel GPU not supported by transformers backend, using CPU")
            return "cpu"
        else:
            return "cpu"

    def _warmup(self) -> None:
        """MPS warmup: 首次推理需要 JIT 编译"""
        try:
            # 生成 1 秒静音音频
            warmup_audio = np.zeros(16000, dtype=np.float32)
            self._pipe(warmup_audio, batch_size=1)
            logger.info("MPS warmup completed")
        except Exception as e:
            logger.warning("MPS warmup failed: %s", e)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> TranscribeResult:
        if self._pipe is None:
            raise RuntimeError("Transformers ASR model not loaded")

        t0 = time.perf_counter()
        audio_duration = len(audio) / settings.sample_rate

        # 构建 generate_kwargs
        generate_kwargs = {}
        lang = language or settings.source_lang

        # Distil-Whisper large-v3 支持多语言，需要设置语言
        # distil-medium.en 是英文专用，不能设置 language 和 task
        if not self._is_english_only():
            if lang:
                generate_kwargs["language"] = lang
            generate_kwargs["task"] = "transcribe"
            generate_kwargs["condition_on_prev_tokens"] = settings.asr_condition_on_prev_tokens
            if settings.asr_temperature:
                generate_kwargs["temperature"] = tuple(settings.asr_temperature)
            if settings.asr_compression_ratio_threshold is not None:
                generate_kwargs["compression_ratio_threshold"] = settings.asr_compression_ratio_threshold
            if settings.asr_logprob_threshold is not None:
                generate_kwargs["logprob_threshold"] = settings.asr_logprob_threshold
            if settings.asr_no_speech_threshold is not None:
                generate_kwargs["no_speech_threshold"] = settings.asr_no_speech_threshold

        # 预测 token 上限
        # Whisper: 30s audio → max 448 tokens
        # 对于短音频，按比例缩减以加速
        max_new_tokens = min(
            settings.asr_max_new_tokens,
            max(128, int(audio_duration / 30.0 * 448))
        )
        generate_kwargs["max_new_tokens"] = max_new_tokens

        # 使用 pipeline 进行推理
        result = self._pipe(
            audio,                   # float32 numpy array
            generate_kwargs=generate_kwargs,
            batch_size=1,            # 实时场景，单条推理
        )

        text = result["text"].strip() if isinstance(result, dict) else str(result).strip()
        elapsed = time.perf_counter() - t0

        return TranscribeResult(
            text=text,
            language=lang,
            duration=audio_duration,
            process_time=elapsed,
        )

    def _is_english_only(self) -> bool:
        """判断当前模型是否是英文专用（如 distil-medium.en）"""
        if self._model_id:
            return ".en" in self._model_id
        return False

    def unload(self) -> None:
        import torch

        if self._model is not None:
            del self._model
            self._model = None
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

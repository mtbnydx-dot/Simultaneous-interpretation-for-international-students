"""Hardware-aware local/cloud model recommendations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.core.config import settings
from app.core.hardware import HardwareProfile, detect_hardware


@dataclass(frozen=True)
class RuntimeChoice:
    backend: str
    device: str
    compute_type: str
    model_id: str | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ModelPlan:
    profile: str
    hardware: dict
    asr: RuntimeChoice
    mt: RuntimeChoice
    cloud_ready: bool
    notes: list[str]

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "hardware": self.hardware,
            "asr": self.asr.to_dict(),
            "mt": self.mt.to_dict(),
            "cloud_ready": self.cloud_ready,
            "notes": self.notes,
        }


def _profile() -> str:
    return (settings.model_profile or "auto").strip().lower()


def select_model_plan(hardware: HardwareProfile | None = None) -> ModelPlan:
    hw = hardware or detect_hardware()
    profile = _profile()
    cloud_ready = bool(settings.cloud_enabled and settings.cloud_base_url)
    notes: list[str] = []

    if profile == "cloud" or (cloud_ready and settings.asr_backend == "cloud"):
        return ModelPlan(
            profile=profile,
            hardware=hw.to_dict(),
            asr=RuntimeChoice("cloud", "cloud", "server", settings.cloud_asr_model, "Configured for remote ASR."),
            mt=RuntimeChoice("cloud", "cloud", "server", settings.cloud_mt_model, "Configured for remote MT."),
            cloud_ready=cloud_ready,
            notes=["Cloud routing is configured; local model download can be skipped for cloud-only deployments."],
        )

    if hw.cuda_available:
        cuda_model = "large-v3" if profile == "quality" else "large-v3-turbo"
        asr = RuntimeChoice(
            "ct2",
            "cuda",
            "float16",
            settings.asr_model_id or cuda_model,
            "NVIDIA CUDA is available; faster-whisper/CTranslate2 is mature here.",
        )
        if hw.cuda_vram_gb and hw.cuda_vram_gb >= 12:
            notes.append("For NVIDIA servers, evaluate Parakeet/Canary via NeMo/Riva as a cloud ASR profile.")
    elif hw.apple_silicon and hw.mlx_available:
        if hw.qwen3_asr_available:
            asr = RuntimeChoice(
                "qwen3",
                "mlx",
                "8bit",
                settings.asr_model_id or settings.asr_qwen3_model_id,
                "Apple Silicon with mlx-audio; Qwen3-ASR is markedly stronger than "
                "Whisper on Chinese and better on English.",
            )
            notes.append(
                "Qwen3-ASR runs 8-bit by default (~2.9GB resident, ~30% faster than bf16 "
                "with no accuracy loss). 4/5/6-bit and bf16 conversions are also published."
            )
        else:
            mlx_model = (
                settings.asr_mlx_fallback_model_id
                if profile == "quality"
                else "mlx-community/whisper-large-v3-turbo-8bit"
            )
            asr = RuntimeChoice(
                "mlx",
                "mlx",
                "int8",
                settings.asr_model_id or mlx_model,
                "Apple Silicon detected; MLX avoids CT2/MPS incompatibility and keeps latency low.",
            )
            notes.append("Install mlx-audio to use Qwen3-ASR, which is much stronger on Chinese.")
        notes.append("For a native macOS rewrite, WhisperKit/CoreML is the more mature Apple-first ASR path.")
    elif hw.openvino_gpu_available:
        asr = RuntimeChoice(
            "openvino",
            "intel_gpu",
            "int8",
            settings.asr_model_id or "openai/whisper-large-v3-turbo",
            "Intel GPU is visible through OpenVINO.",
        )
    else:
        low_memory = hw.memory_gb is not None and hw.memory_gb < 8
        if profile == "quality" and not low_memory:
            cpu_model = "large-v3"
        elif profile == "speed" or low_memory:
            cpu_model = "small"
        else:
            cpu_model = "large-v3-turbo"
        asr = RuntimeChoice(
            "ct2",
            "cpu",
            "int8",
            settings.asr_model_id or cpu_model,
            "CPU fallback; use int8 and shorter segments to control latency.",
        )
        if low_memory:
            notes.append("Low-memory CPU devices should prefer cloud ASR/MT for real-time use.")

    mt_backend = settings.mt_backend
    if mt_backend == "cloud" or (cloud_ready and profile in ("quality", "cloud")):
        mt = RuntimeChoice("cloud", "cloud", "server", settings.cloud_mt_model, "Remote MT selected by profile/config.")
    else:
        mt = RuntimeChoice(
            "local",
            "auto",
            "gguf",
            settings.mt_model_id,
            "Local Hy-MT2 GGUF is the default compact offline translation path.",
        )
        if hw.memory_gb is not None and hw.memory_gb >= 24:
            notes.append("High-memory machines can evaluate Hunyuan-MT-7B or Qwen/Tower GGUF for quality, but latency will rise.")

    if cloud_ready:
        notes.append("Cloud endpoints are configured and can be used as fallback or quality tier.")

    return ModelPlan(
        profile=profile,
        hardware=hw.to_dict(),
        asr=asr,
        mt=mt,
        cloud_ready=cloud_ready,
        notes=notes,
    )

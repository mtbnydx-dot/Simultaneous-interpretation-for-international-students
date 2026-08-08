"""
ASR 后端注册表。
支持的后端:
- qwen3: Qwen3-ASR via mlx-audio (Apple Silicon, 默认)
- mlx: Apple Silicon MLX / Metal (Whisper)
- ct2: faster-whisper (CTranslate2)
- openvino: Intel OpenVINO
- transformers-distil: Distil-Whisper via HuggingFace Transformers
- transformers-whisper: 标准 Whisper via HuggingFace Transformers
- cloud: Remote ASR service
"""

from app.asr.backends.base import ASRBackend, TranscribeResult

__all__ = ["ASRBackend", "TranscribeResult", "create_backend"]


def create_backend(name: str) -> ASRBackend:
    """
    根据后端名称创建 ASR 后端实例。

    Args:
        name: 后端名称 ("mlx", "ct2", "openvino", "transformers-distil", "transformers-whisper", "cloud")

    Returns:
        ASRBackend 实例

    Raises:
        ValueError: 未知的后端名称
    """
    if name == "qwen3":
        from app.asr.backends.qwen3_backend import Qwen3Backend
        return Qwen3Backend()
    elif name == "mlx":
        from app.asr.backends.mlx_backend import MLXBackend
        return MLXBackend()
    elif name == "ct2":
        from app.asr.backends.ct2_backend import CT2Backend
        return CT2Backend()
    elif name == "openvino":
        from app.asr.backends.openvino_backend import OpenVINOBackend
        return OpenVINOBackend()
    elif name in ("transformers-distil", "transformers-whisper"):
        from app.asr.backends.transformers_backend import TransformersBackend
        return TransformersBackend()
    elif name == "cloud":
        from app.asr.backends.cloud_backend import CloudASRBackend
        return CloudASRBackend()
    else:
        raise ValueError(f"Unknown ASR backend: {name}")

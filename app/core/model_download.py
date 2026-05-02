import logging
import os
import subprocess
import sys
from pathlib import Path

from app.core.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def download_asr_model() -> None:
    """根据当前配置下载对应的 ASR 模型"""
    backend = settings.asr_backend
    if backend == "auto":
        from app.asr.engine import _detect_device
        backend, _device, _compute_type = _detect_device()

    if backend in ("transformers-distil", "transformers-whisper"):
        _download_transformers_asr()
    else:
        from faster_whisper import WhisperModel
        model_size = _resolve_ct2_model_size()
        logger.info("Downloading ASR model: %s", model_size)
        WhisperModel(model_size, device="cpu", compute_type=settings.asr_compute_type)
        logger.info("ASR model downloaded and cached")


def _resolve_ct2_model_size() -> str:
    raw = settings.asr_model_id or settings.asr_model_size
    if "/" in raw:
        raw = raw.split("/")[-1]
    if raw.startswith("whisper-"):
        raw = raw[len("whisper-"):]
    return raw


def _download_transformers_asr() -> None:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from app.asr.backends.transformers_backend import DISTIL_WHISPER_MODELS

    backend = settings.asr_backend

    if settings.asr_model_id:
        model_id = settings.asr_model_id
        if "/" not in model_id and not model_id.startswith("distil-"):
            model_id = f"openai/whisper-{model_id}"
    elif backend == "transformers-distil":
        model_id = DISTIL_WHISPER_MODELS.get(
            settings.asr_model_size, DISTIL_WHISPER_MODELS["large-v3"]
        )
    else:
        model_id = "openai/whisper-large-v3-turbo"

    logger.info("Downloading transformers ASR model: %s", model_id)
    AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    AutoProcessor.from_pretrained(model_id)
    logger.info("Transformers ASR model downloaded and cached: %s", model_id)


def download_mt_model(force: bool = False) -> str:
    """
    下载 HY-MT1.5 GGUF 翻译模型。
    优先从 models/ 目录查找，否则通过 huggingface-cli 下载。
    """
    model_id = settings.mt_model_id

    # 如果已经是 GGUF 文件路径
    if model_id.endswith(".gguf"):
        target_path = resolve_project_path(model_id)
        if target_path.exists():
            logger.info("MT GGUF model found at: %s", target_path)
            return str(target_path)

        # 尝试从 HF 下载 (优先 Python API，支持断点续传)
        hf_repo = "tencent/HY-MT1.5-1.8B-GGUF"
        filename = Path(model_id).name
        logger.info("Downloading MT GGUF: %s/%s → %s", hf_repo, filename, target_path)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        hf_error: Exception | None = None
        try:
            from huggingface_hub import hf_hub_download
            result = hf_hub_download(
                repo_id=hf_repo,
                filename=filename,
                local_dir=str(target_path.parent),
            )
            logger.info("MT model downloaded: %s", result)
            return str(Path(result).resolve())
        except ImportError as exc:
            hf_error = exc
            logger.warning("huggingface_hub is unavailable; trying CLI fallback")
        except Exception as exc:
            hf_error = exc
            logger.exception("hf_hub_download failed")

        if getattr(sys, "frozen", False):
            raise RuntimeError(
                f"Failed to download MT model ({hf_repo}/{filename}). "
                f"Check the log above for details, or download manually: "
                f"huggingface-cli download {hf_repo} {filename} --local-dir models/"
            ) from hf_error

        try:
            subprocess.run(
                [sys.executable, "-m", "huggingface_hub", "download", hf_repo, filename,
                 "--local-dir", str(target_path.parent)],
                check=True,
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1"},
            )
            logger.info("MT model downloaded: %s", target_path)
            return str(target_path)
        except Exception as exc2:
            raise RuntimeError(
                f"Failed to download MT model. "
                f"Run manually: hf download {hf_repo} {filename} --local-dir models/"
            ) from exc2

    # NLLB / opus-mt 旧路径 (留作回退)
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required. Run: pip install transformers sentencepiece sacremoses torch"
        ) from exc

    logger.info("Downloading MT model: %s (force=%s)", model_id, force)
    AutoTokenizer.from_pretrained(model_id, force_download=force, token=settings.hf_token)
    AutoModelForSeq2SeqLM.from_pretrained(model_id, force_download=force, token=settings.hf_token)
    logger.info("MT model ready: %s", model_id)
    return model_id


def ensure_mt_model_available() -> str:
    if not settings.auto_download_models:
        return settings.mt_model_id
    return download_mt_model()

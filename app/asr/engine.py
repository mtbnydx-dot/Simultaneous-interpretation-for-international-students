"""
ASR 引擎核心。
支持多后端: ct2, openvino, transformers-distil, transformers-whisper
"""

import logging
import platform
import threading
from pathlib import Path
import numpy as np

from app.core.config import settings, PROJECT_ROOT
from app.asr.backends.base import ASRBackend

logger = logging.getLogger(__name__)

OPENVINO_BLOCK_FILE = PROJECT_ROOT / ".openvino_blocked"


def _is_openvino_blocked() -> bool:
    """检查 OpenVINO 是否之前已失败并被缓存屏蔽"""
    return OPENVINO_BLOCK_FILE.exists()


def _block_openvino():
    """记录 OpenVINO 失败，下次启动跳过"""
    OPENVINO_BLOCK_FILE.write_text(
        "OpenVINO export failed on this machine. "
        "Delete this file to retry, or set TRANS_ASR_BACKEND=openvino to force."
    )
    logger.warning("OpenVINO blocked for future starts. Delete %s to retry.", OPENVINO_BLOCK_FILE)


def _is_mac_mps_available() -> bool:
    try:
        import torch
        return (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and platform.system() == "Darwin"
        )
    except ImportError:
        return False


def _is_cuda_available() -> tuple[bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0) or "NVIDIA GPU"
            return True, name
    except ImportError:
        pass
    return False, ""


def _detect_device() -> tuple[str, str, str]:
    """
    自动检测最佳设备和后端。

    检测优先级:
    1. NVIDIA CUDA → ct2 (float16)
    2. Apple MPS → transformers-whisper (float32)
    3. Intel GPU → openvino (int8)
    4. CPU → ct2 (int8)

    Returns:
        (backend, device, compute_type)
    """
    backend = settings.asr_backend
    device = settings.asr_device
    compute_type = settings.asr_compute_type

    if backend != "auto" and device != "auto" and compute_type != "default":
        return backend, device, compute_type

    detected_device = "cpu"
    detected_backend = "ct2"
    detected_compute = "int8"

    # 1. NVIDIA GPU
    cuda_ok, cuda_name = _is_cuda_available()
    if cuda_ok:
        detected_device = "cuda"
        detected_backend = "ct2"
        detected_compute = "float16"
        logger.info("Detected NVIDIA GPU: %s", cuda_name)

    # 2. Apple MPS. CTranslate2/faster-whisper does not support MPS, so the
    #    automatic Mac path uses Transformers directly.
    if detected_device == "cpu" and _is_mac_mps_available():
        detected_device = "mps"
        detected_backend = "transformers-whisper"
        detected_compute = "float32"
        logger.info("Detected Apple MPS (Metal)")

    # 3. Intel GPU via OpenVINO (如果之前失败过则跳过)
    if detected_device == "cpu":
        if _is_openvino_blocked():
            logger.info("OpenVINO previously blocked, skipping Intel GPU detection")
        else:
            try:
                import openvino as ov
                core = ov.Core()
                devices = core.available_devices
                gpu_devices = [d for d in devices if "GPU" in d]
                if gpu_devices:
                    detected_device = "intel_gpu"
                    detected_backend = "openvino"
                    detected_compute = "int8"
                    logger.info("Detected Intel GPU via OpenVINO: %s", gpu_devices)
            except ImportError:
                logger.info("OpenVINO not installed, Intel GPU support unavailable")
            except Exception as exc:
                logger.warning("OpenVINO probe failed: %s", exc)

    if detected_device == "cpu":
        logger.info("No GPU detected, using CPU (ct2/int8)")

    # 合并：显式 backend 时，根据该 backend 能力选择 auto device，避免 Mac 上
    # ct2/faster-whisper 被分配到不支持的 MPS。
    if backend == "auto":
        backend = detected_backend
    if device == "auto":
        if backend == "ct2":
            device = "cuda" if detected_device == "cuda" else "cpu"
        elif backend in ("transformers-distil", "transformers-whisper"):
            device = detected_device if detected_device in ("cuda", "mps") else "cpu"
        elif backend == "openvino":
            device = "intel_gpu" if detected_device == "intel_gpu" else "cpu"
        else:
            device = detected_device
    if compute_type == "default":
        if backend == "ct2":
            compute_type = "float16" if device == "cuda" else "int8"
        elif backend in ("transformers-distil", "transformers-whisper"):
            compute_type = "float16" if device == "cuda" else "float32"
        else:
            compute_type = detected_compute

    # 强制修正
    if device == "intel_gpu" and backend != "openvino" and settings.asr_backend == "auto":
        logger.warning("Intel GPU → switching to openvino backend")
        backend = "openvino"

    if backend == "ct2" and device == "mps":
        logger.warning("CT2/faster-whisper does not support Apple MPS; using CPU")
        device = "cpu"
        if settings.asr_compute_type == "default":
            compute_type = "int8"

    return backend, device, compute_type


class ASREngine:
    """ASR 引擎，委托给具体的后端实现。"""

    def __init__(self):
        self._backend: ASRBackend | None = None
        self._backend_name: str = "none"
        self._device: str = "unknown"
        self._compute_type: str = "unknown"
        self._lock = threading.RLock()

    @property
    def model(self):
        return self._backend if self._backend and self._backend.is_loaded else None

    def load_model(self):
        with self._lock:
            if self._backend is not None and self._backend.is_loaded:
                return

            backend_name, device, compute_type = _detect_device()
            model_id = settings.asr_model_id

            # 回退链：主后端 → transformers-whisper → ct2
            fallback_chain = [
                ("transformers-whisper", "cpu", "float32"),
                ("ct2", "cpu", "int8"),
            ]
            # 如果主后端已在回退链中，跳过
            fallback_chain = [(b, d, c) for b, d, c in fallback_chain if b != backend_name]

            try:
                self._try_load(backend_name, device, compute_type, model_id)
            except Exception as exc:
                logger.warning("Primary backend %s failed (%s), trying fallbacks...",
                               backend_name, exc)
                if backend_name == "openvino":
                    _block_openvino()
                loaded = False
                last_error = exc
                for fb_name, fb_device, fb_compute in fallback_chain:
                    try:
                        logger.info("Fallback: trying %s/%s", fb_name, fb_device)
                        self._try_load(fb_name, fb_device, fb_compute, model_id)
                        loaded = True
                        break
                    except Exception as fb_exc:
                        logger.warning("Fallback %s also failed: %s", fb_name, fb_exc)
                        last_error = fb_exc
                if not loaded:
                    raise RuntimeError(
                        f"All ASR backends failed. Last error: {last_error}. "
                        f"Check GPU drivers or set TRANS_ASR_BACKEND=transformers-whisper and TRANS_ASR_DEVICE=cpu."
                    ) from last_error

    def _try_load(self, backend_name: str, device: str, compute_type: str,
                  model_id: str | None):
        from app.asr.backends import create_backend

        self._backend_name = backend_name
        self._device = device
        self._compute_type = compute_type

        logger.info("Loading ASR: backend=%s device=%s compute=%s",
                     backend_name, device, compute_type)

        backend = create_backend(backend_name)
        backend.load(device=device, compute_type=compute_type, model_id=model_id)
        self._backend = backend

        logger.info("ASR model loaded (backend=%s)", backend_name)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        with self._lock:
            if self._backend is None or not self._backend.is_loaded:
                raise RuntimeError("ASR model not loaded")

            result = self._backend.transcribe(audio, language)
            return result.text

    def unload(self):
        with self._lock:
            if self._backend is not None:
                self._backend.unload()
                self._backend = None


asr_engine = ASREngine()

"""
ASR 引擎核心。
源码支持多后端；macOS 分发包固定使用 Qwen3-ASR。
"""

import logging
import platform
import threading
from typing import Callable
from importlib.util import find_spec
import numpy as np

from app.core.config import settings, PROJECT_ROOT
from app.core.memory import collect_model_memory
from app.core.model_profiles import select_model_plan
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


def _is_qwen3_available() -> bool:
    """Qwen3-ASR 需要 Apple Silicon + mlx-audio。"""
    return _is_apple_silicon_mlx() and find_spec("mlx_audio") is not None


def _is_apple_silicon_mlx() -> bool:
    """
    MLX 路线的可用性判断。故意不 import torch：Mac 上 ASR 走 mlx-whisper、
    MT 走 llama.cpp Metal、VAD 走 onnxruntime，torch 只会白白多占约 180MB
    常驻内存和 1 秒启动时间。
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    return find_spec("mlx.core") is not None


def _model_id_for_backend(backend_name: str, primary_model_id: str | None) -> str | None:
    """Keep model families isolated while walking the ASR fallback chain."""
    backend = (backend_name or "").strip().lower()
    candidate = (primary_model_id or "").strip() or None
    configured = (settings.asr_model_id or "").strip() or None

    if backend == "cloud":
        return settings.cloud_asr_model
    if backend == "qwen3":
        return settings.asr_qwen3_model_id
    if backend == "mlx":
        if candidate and "whisper" in candidate.casefold():
            return candidate
        return settings.asr_mlx_fallback_model_id

    # CT2/OpenVINO/Transformers are Whisper-family backends. Never feed a
    # Qwen3 repository to them after the primary MLX backend fails.
    for value in (candidate, configured):
        if value and "qwen" not in value.casefold():
            return value
    return None


def _backend_runtime_available(backend_name: str) -> bool:
    if backend_name == "qwen3":
        return _is_qwen3_available()
    if backend_name == "mlx":
        return _is_apple_silicon_mlx() and find_spec("mlx_whisper") is not None
    if backend_name.startswith("transformers"):
        return find_spec("torch") is not None and find_spec("transformers") is not None
    if backend_name == "ct2":
        return find_spec("faster_whisper") is not None
    if backend_name == "openvino":
        return find_spec("openvino") is not None
    return True


def _fallback_candidates(primary_backend: str) -> list[tuple[str, str, str]]:
    """Build the optional source-runtime fallback list.

    The signed macOS bundle disables this list and intentionally contains no
    Whisper inference runtime. Keeping the policy explicit prevents a failed
    Qwen load from silently switching models and changing latency or accuracy.
    """
    if not settings.asr_fallbacks_enabled:
        return []

    candidates: list[tuple[str, str, str]] = []
    if settings.cloud_enabled and settings.cloud_base_url:
        candidates.append(("cloud", "cloud", "server"))
    if _is_qwen3_available():
        candidates.append(("qwen3", "mlx", "8bit"))
    if _is_apple_silicon_mlx():
        candidates.append(("mlx", "mlx", "int8"))
    candidates += [
        ("transformers-whisper", "cpu", "float32"),
        ("ct2", "cpu", "int8"),
    ]
    return [
        (backend, device, compute)
        for backend, device, compute in candidates
        if backend != primary_backend and _backend_runtime_available(backend)
    ]


def _is_torch_mps_available() -> bool:
    """仅 transformers 后端需要（它本来就依赖 torch）。"""
    try:
        import torch

        return (
            platform.system() == "Darwin"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except Exception:
        return False


def _is_cuda_available() -> tuple[bool, str]:
    # macOS 永远没有 CUDA；不要为了这个探测导入 torch。
    if platform.system() == "Darwin":
        return False, ""
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
    2. Apple Silicon → mlx-whisper (Metal/MLX, quantized)
    3. Intel GPU → openvino (int8)
    4. CPU → ct2 (int8)

    Returns:
        (backend, device, compute_type)
    """
    backend = settings.asr_backend
    device = settings.asr_device
    compute_type = settings.asr_compute_type

    if backend == "cloud":
        return "cloud", "cloud", "server"

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

    # 2. Apple Silicon. CTranslate2/faster-whisper does not support Apple MPS,
    #    so the automatic Mac path uses MLX directly.
    if detected_device == "cpu" and _is_apple_silicon_mlx():
        detected_device = "mlx"
        # Qwen3-ASR 在中文上明显强于 Whisper（实测同一组音频 CER 0% vs 3.9%，
        # 加噪后差距更大），英文也略好，所以 Apple Silicon 上优先用它；
        # 没装 mlx-audio 时退回 MLX Whisper。
        detected_backend = "qwen3" if _is_qwen3_available() else "mlx"
        detected_compute = "int8"
        logger.info("Detected Apple Silicon MLX/Metal (ASR backend=%s)", detected_backend)

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
        elif backend in ("mlx", "qwen3"):
            device = "mlx" if detected_device == "mlx" else "cpu"
        elif backend in ("transformers-distil", "transformers-whisper"):
            if detected_device == "cuda":
                device = "cuda"
            elif _is_torch_mps_available():
                device = "mps"
            else:
                device = "cpu"
        elif backend == "openvino":
            device = "intel_gpu" if detected_device == "intel_gpu" else "cpu"
        else:
            device = detected_device
    if compute_type == "default":
        if backend == "ct2":
            compute_type = "float16" if device == "cuda" else "int8"
        elif backend == "mlx":
            compute_type = "int8"
        elif backend == "qwen3":
            compute_type = "8bit"   # mlx-audio 支持全套量化，见 qwen3_backend
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

    if backend in ("mlx", "qwen3") and platform.system() != "Darwin":
        logger.warning("%s ASR is macOS/Apple Silicon only; using CT2 CPU", backend)
        backend = "ct2"
        device = "cpu"
        compute_type = "int8"

    return backend, device, compute_type


class ASREngine:
    """ASR 引擎，委托给具体的后端实现。"""

    def __init__(self):
        self._backend: ASRBackend | None = None
        self._backend_name: str = "none"
        self._device: str = "unknown"
        self._compute_type: str = "unknown"
        self._model_id: str | None = None
        self._model_path: str | None = None
        self._lock = threading.RLock()

    @property
    def model(self):
        return self._backend if self._backend and self._backend.is_loaded else None

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def compute_type(self) -> str:
        return self._compute_type

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def model_path(self) -> str | None:
        return self._model_path

    @property
    def acceleration_info(self) -> dict[str, str | bool | None]:
        accelerated_devices = {"cuda", "mlx", "mps", "metal", "intel_gpu", "npu", "cloud"}
        accelerated_backends = {"mlx", "qwen3", "openvino", "cloud"}
        accelerated = self._device in accelerated_devices or self._backend_name in accelerated_backends
        if self._backend_name in ("mlx", "qwen3"):
            status = "mlx_metal"
            note = "Apple Silicon MLX/Metal ASR"
        elif self._device == "cuda":
            status = "cuda"
            note = "NVIDIA CUDA ASR"
        elif self._backend_name == "openvino":
            status = "openvino"
            note = "OpenVINO ASR"
        elif self._backend_name == "cloud":
            status = "cloud"
            note = "Remote ASR"
        elif self._device == "cpu":
            status = "cpu_only"
            note = "CPU ASR"
        else:
            status = self._device or self._backend_name
            note = None
        return {"accelerated": accelerated, "status": status, "note": note}

    def load_model(self):
        with self._lock:
            if self._backend is not None and self._backend.is_loaded:
                return

            backend_name, device, compute_type = _detect_device()
            model_plan = select_model_plan()
            model_id = settings.asr_model_id
            if model_id is None and model_plan.asr.backend == backend_name:
                model_id = model_plan.asr.model_id

            fallback_chain = _fallback_candidates(backend_name)

            try:
                self._try_load(
                    backend_name,
                    device,
                    compute_type,
                    _model_id_for_backend(backend_name, model_id),
                )
            except Exception as exc:
                error_text = str(exc)
                logger.warning("Primary backend %s failed: %s", backend_name, error_text)
                if backend_name == "openvino":
                    _block_openvino()
                loaded = False
                # Tracebacks retain every failed load() frame and can therefore retain
                # partially constructed model graphs. Keep the message, then sever the
                # frame chain before loading a fallback into the same unified memory.
                exc.__traceback__ = None
                exc.__context__ = None
                exc.__cause__ = None
                last_error = RuntimeError(error_text)
                collect_model_memory()
                if fallback_chain:
                    logger.info("Trying %d configured ASR fallback(s)", len(fallback_chain))
                for fb_name, fb_device, fb_compute in fallback_chain:
                    try:
                        logger.info("Fallback: trying %s/%s", fb_name, fb_device)
                        self._try_load(
                            fb_name,
                            fb_device,
                            fb_compute,
                            _model_id_for_backend(fb_name, model_id),
                        )
                        loaded = True
                        break
                    except Exception as fb_exc:
                        error_text = str(fb_exc)
                        logger.warning("Fallback %s also failed: %s", fb_name, error_text)
                        fb_exc.__traceback__ = None
                        fb_exc.__context__ = None
                        fb_exc.__cause__ = None
                        last_error = RuntimeError(error_text)
                        collect_model_memory()
                if not loaded:
                    if not settings.asr_fallbacks_enabled:
                        raise RuntimeError(
                            f"Qwen3-ASR failed to load: {last_error}. "
                            "Check that the model download is complete and macOS is 14.0 or newer."
                        ) from last_error
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
        try:
            backend.load(device=device, compute_type=compute_type, model_id=model_id)
        except Exception:
            # MLX/Transformers 可能在读完部分权重后才发现文件损坏。若直接进入
            # fallback，失败后端的统一内存会一直占到 GC，甚至导致回退模型也 OOM。
            try:
                backend.unload()
            except Exception:
                logger.debug("Failed to clean up partially loaded ASR backend", exc_info=True)
            raise
        self._backend = backend
        self._model_id = getattr(backend, "model_id", None) or model_id
        self._model_path = getattr(backend, "model_path", None)

        logger.info("ASR model loaded (backend=%s)", backend_name)

    def _transcribe_final_result(
        self,
        audio: np.ndarray,
        language: str | None = None,
        on_partial: Callable[[str, str | None], None] | None = None,
    ):
        """Run one authoritative transcription under the shared model lock."""
        with self._lock:
            if self._backend is None or not self._backend.is_loaded:
                raise RuntimeError("ASR model not loaded")
            if on_partial is not None:
                return self._backend.transcribe_stream(audio, language, on_partial)
            return self._backend.transcribe(audio, language)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        return self._transcribe_final_result(audio, language).text

    def transcribe_result(self, audio: np.ndarray, language: str | None = None):
        """
        和 transcribe 一样，但返回完整的 TranscribeResult。

        源语言选"自动检测"时，需要拿到后端识别出的语种再交给 MT，
        所以不能只返回文本。
        """
        return self._transcribe_final_result(audio, language)

    def transcribe_stream_result(
        self,
        audio: np.ndarray,
        language: str | None,
        on_partial: Callable[[str, str | None], None],
    ):
        return self._transcribe_final_result(audio, language, on_partial)

    def _detach_backend_locked(self) -> None:
        backend = self._backend
        self._backend = None
        if backend is not None:
            backend.unload()

    def unload(self):
        with self._lock:
            self._detach_backend_locked()
            self._backend_name = "none"
            self._device = "unknown"
            self._compute_type = "unknown"
            self._model_id = None
            self._model_path = None


asr_engine = ASREngine()


def unload_asr_engine(engine: ASREngine | None = None) -> None:
    """Unload ASR, then collect after the backend method frame has unwound."""
    target = engine or asr_engine
    target.unload()
    collect_model_memory()

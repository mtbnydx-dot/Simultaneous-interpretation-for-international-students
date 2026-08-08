"""Hardware capability detection used for model routing decisions."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib.util import find_spec
from dataclasses import asdict, dataclass
from functools import lru_cache


@dataclass(frozen=True)
class HardwareProfile:
    os_name: str
    os_version: str
    machine: str
    processor: str
    cpu_count: int
    memory_gb: float | None
    apple_silicon: bool
    mps_available: bool
    mlx_available: bool
    qwen3_asr_available: bool
    cuda_available: bool
    cuda_name: str | None
    cuda_vram_gb: float | None
    openvino_gpu_available: bool
    llama_gpu_offload_supported: bool | None

    def to_dict(self) -> dict:
        return asdict(self)


def _memory_gb() -> float | None:
    if platform.system() == "Darwin":
        try:
            output = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return round(int(output.strip()) / (1024**3), 1)
        except Exception:
            return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round((pages * page_size) / (1024**3), 1)
    except Exception:
        return None


def _torch_caps() -> tuple[bool, bool, str | None, float | None]:
    # Apple Silicon 上 Metal 一定可用，不需要为了报告这个字段而导入 torch
    # （macOS 路径已全面改用 mlx / llama.cpp Metal / onnxruntime）。
    mps_available = platform.system() == "Darwin" and platform.machine() == "arm64"
    cuda_available = False
    cuda_name = None
    cuda_vram_gb = None
    if "torch" not in sys.modules and platform.system() == "Darwin":
        return mps_available, cuda_available, cuda_name, cuda_vram_gb
    try:
        import torch

        if torch.cuda.is_available():
            cuda_available = True
            cuda_name = torch.cuda.get_device_name(0) or "NVIDIA GPU"
            props = torch.cuda.get_device_properties(0)
            cuda_vram_gb = round(props.total_memory / (1024**3), 1)
    except Exception:
        pass
    return mps_available, cuda_available, cuda_name, cuda_vram_gb


def _mlx_available() -> bool:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    return find_spec("mlx.core") is not None


def _qwen3_asr_available() -> bool:
    return _mlx_available() and find_spec("mlx_audio") is not None


def _openvino_gpu_available() -> bool:
    try:
        import openvino as ov

        return any("GPU" in device for device in ov.Core().available_devices)
    except Exception:
        return False


def _llama_gpu_offload_supported() -> bool | None:
    try:
        if "llama_cpp.llama_cpp" not in sys.modules:
            return None
        from llama_cpp import llama_cpp as _llc

        supports = getattr(_llc, "llama_supports_gpu_offload", None)
        if supports is None:
            return None
        return bool(supports())
    except Exception:
        return None


@lru_cache(maxsize=1)
def detect_hardware() -> HardwareProfile:
    mps_available, cuda_available, cuda_name, cuda_vram_gb = _torch_caps()
    machine = platform.machine()
    os_name = platform.system()
    return HardwareProfile(
        os_name=os_name,
        os_version=platform.platform(),
        machine=machine,
        processor=platform.processor(),
        cpu_count=os.cpu_count() or 1,
        memory_gb=_memory_gb(),
        apple_silicon=os_name == "Darwin" and machine == "arm64",
        mps_available=mps_available,
        mlx_available=_mlx_available(),
        qwen3_asr_available=_qwen3_asr_available(),
        cuda_available=cuda_available,
        cuda_name=cuda_name,
        cuda_vram_gb=cuda_vram_gb,
        openvino_gpu_available=_openvino_gpu_available(),
        llama_gpu_offload_supported=_llama_gpu_offload_supported(),
    )

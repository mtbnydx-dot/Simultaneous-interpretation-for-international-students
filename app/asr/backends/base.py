"""
ASR 后端抽象基类。
所有 ASR 后端必须实现此接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class TranscribeResult:
    """转录结果，包含文本和元数据"""
    text: str
    language: str | None = None
    duration: float = 0.0       # 音频时长（秒）
    process_time: float = 0.0   # 处理耗时（秒）

    @property
    def rtf(self) -> float:
        """Real-Time Factor = process_time / duration"""
        if self.duration <= 0:
            return 0.0
        return self.process_time / self.duration


class ASRBackend(ABC):
    """
    所有 ASR 后端的抽象基类。

    核心契约:
    - 输入: np.ndarray (float32, 16kHz 单声道, 范围 [-1, 1])
    - 输出: TranscribeResult (text 为纯文本，已 strip)
    - 线程安全: transcribe() 通过 run_in_executor 在线程池中调用
    """

    @abstractmethod
    def load(self, device: str, compute_type: str, model_id: str | None = None) -> None:
        """
        加载模型。

        Args:
            device: 设备类型 ("cpu", "cuda", "mps", "intel_gpu")
            compute_type: 计算精度 ("int8", "float16", "float32", "default")
            model_id: 可选的模型 ID (HuggingFace model ID 或模型大小)
        """
        ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray, language: str | None = None) -> TranscribeResult:
        """
        转录音频。

        Args:
            audio: float32 numpy array, 16kHz 单声道, 范围 [-1, 1]
            language: 语言代码 (e.g., "en"), None 则使用默认

        Returns:
            TranscribeResult with text and metadata
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """释放模型资源"""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """模型是否已加载"""
        ...

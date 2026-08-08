"""
音频预处理模块。
在 ASR 之前对原始音频进行降噪和增益控制。

功能:
- 高通滤波: 去除 80Hz 以下低频噪声（风扇、空调、电流声）
- AGC: 自动增益控制，让安静/大声说话者都能被正确识别
- 归一化: 确保音频幅度在 [-1, 1] 范围内
"""

import threading

import numpy as np

from app.core.config import settings


class AudioPreprocessor:
    """音频预处理器"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._highpass_sos = None
        self._highpass_initialized = False
        self._init_lock = threading.Lock()

    def _init_highpass(self) -> None:
        """初始化高通滤波器（Butterworth 二阶）"""
        from scipy.signal import butter

        nyq = self.sample_rate / 2.0
        cutoff = settings.audio_highpass_freq
        if cutoff >= nyq:
            return
        self._highpass_sos = butter(
            N=2,
            Wn=cutoff / nyq,
            btype="highpass",
            output="sos",
        )

    def _ensure_highpass(self) -> None:
        if self._highpass_initialized:
            return
        with self._init_lock:
            if self._highpass_initialized:
                return
            self._init_highpass()
            self._highpass_initialized = True

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        完整预处理管线。

        Args:
            audio: float32 numpy array, 16kHz 单声道

        Returns:
            处理后的音频
        """
        if not settings.audio_preprocess_enabled:
            return audio

        self._ensure_highpass()

        # Step 1: 高通滤波
        if self._highpass_sos is not None:
            from scipy.signal import sosfilt

            audio = sosfilt(self._highpass_sos, audio).astype(np.float32, copy=False)
        else:
            # 后续步骤原地运算，不能意外修改调用方持有的原始音频。
            audio = np.array(audio, dtype=np.float32, copy=True)

        # Step 2: AGC (自动增益控制)
        audio = self._agc(audio)

        # Step 3: 硬裁剪归一化
        np.clip(audio, -1.0, 1.0, out=audio)

        return audio

    def _agc(self, audio: np.ndarray, target_rms: float = 0.1,
             max_gain: float = 10.0) -> np.ndarray:
        """
        简单 AGC：按 RMS 缩放到目标电平。

        防止过度放大（max_gain 上限）和静音段放大。
        """
        if audio.size == 0:
            return audio
        rms = float(np.sqrt(np.dot(audio, audio) / audio.size))
        if rms < 1e-6:
            return audio  # 纯静音，不处理

        gain = target_rms / rms
        gain = min(gain, max_gain)
        audio *= gain
        return audio


# 模块级单例
audio_preprocessor = AudioPreprocessor()

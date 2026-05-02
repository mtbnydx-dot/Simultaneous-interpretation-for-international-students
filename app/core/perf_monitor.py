"""
性能监控模块。
跟踪 ASR + MT 各阶段耗时，计算 RTF（实时因子）。

RTF < 1.0 表示处理速度快于实时，可以做同声传译。
RTF > 1.0 表示处理速度跟不上，需要更轻的模型或更快的硬件。
"""

import time
import logging
from dataclasses import dataclass, field

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SegmentMetrics:
    """单个音频片段的性能指标"""
    audio_duration: float = 0.0
    preprocess_time: float = 0.0
    asr_time: float = 0.0
    mt_time: float = 0.0
    total_time: float = 0.0

    @property
    def rtf(self) -> float:
        """Real-Time Factor = processing_time / audio_duration"""
        if self.audio_duration <= 0:
            return 0.0
        return self.total_time / self.audio_duration

    def log(self) -> None:
        """记录性能日志"""
        logger.info(
            "Perf: audio=%.2fs | preprocess=%.3fs | asr=%.3fs | mt=%.3fs | "
            "total=%.3fs | RTF=%.3f %s",
            self.audio_duration,
            self.preprocess_time,
            self.asr_time,
            self.mt_time,
            self.total_time,
            self.rtf,
            "OK real-time" if self.rtf < 1.0 else "SLOW",
        )


class PerfMonitor:
    """性能监控器"""

    def __init__(self, max_history: int = 100):
        self._history: list[SegmentMetrics] = []
        self._max_history = max_history

    def create_timer(self) -> "PerfTimer":
        """创建性能计时器"""
        return PerfTimer(self)

    def record(self, metrics: SegmentMetrics) -> None:
        """记录性能指标"""
        if settings.perf_log_enabled:
            metrics.log()

        self._history.append(metrics)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_stats(self) -> dict:
        """返回最近 N 个片段的汇总统计"""
        if not self._history:
            return {}

        rtfs = [m.rtf for m in self._history if m.audio_duration > 0]
        asr_times = [m.asr_time for m in self._history]
        mt_times = [m.mt_time for m in self._history]

        return {
            "segments": len(self._history),
            "avg_rtf": sum(rtfs) / len(rtfs) if rtfs else 0,
            "max_rtf": max(rtfs) if rtfs else 0,
            "min_rtf": min(rtfs) if rtfs else 0,
            "avg_asr_ms": sum(asr_times) / len(asr_times) * 1000 if asr_times else 0,
            "avg_mt_ms": sum(mt_times) / len(mt_times) * 1000 if mt_times else 0,
        }

    def clear(self) -> None:
        """清空历史记录"""
        self._history.clear()


class PerfTimer:
    """性能计时器，用于跟踪单个音频片段的处理时间"""

    def __init__(self, monitor: PerfMonitor):
        self._monitor = monitor
        self._metrics = SegmentMetrics()
        self._t_start = 0.0
        self._t_last = 0.0

    def start(self, audio_duration: float) -> None:
        """开始计时"""
        self._metrics.audio_duration = audio_duration
        self._t_start = time.perf_counter()
        self._t_last = self._t_start

    def mark_preprocess_done(self) -> None:
        """标记预处理完成"""
        now = time.perf_counter()
        self._metrics.preprocess_time = now - self._t_last
        self._t_last = now

    def mark_asr_done(self) -> None:
        """标记 ASR 完成"""
        now = time.perf_counter()
        self._metrics.asr_time = now - self._t_last
        self._t_last = now

    def mark_mt_done(self) -> None:
        """标记 MT 完成并记录总时间"""
        now = time.perf_counter()
        self._metrics.mt_time = now - self._t_last
        self._metrics.total_time = now - self._t_start
        self._monitor.record(self._metrics)


# 模块级单例
perf_monitor = PerfMonitor()

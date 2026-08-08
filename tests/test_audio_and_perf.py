from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.core.audio_preprocess import AudioPreprocessor
from app.core.perf_monitor import PerfMonitor, SegmentMetrics


def test_audio_preprocessor_import_is_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import app.core.audio_preprocess; "
            "raise SystemExit('scipy.signal' in sys.modules)",
        ],
        check=False,
    )

    assert result.returncode == 0


def test_audio_preprocessor_returns_finite_float32_audio() -> None:
    preprocessor = AudioPreprocessor()
    audio = np.linspace(-4.0, 4.0, 16_000, dtype=np.float32)

    result = preprocessor.process(audio)

    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert float(np.max(np.abs(result))) <= 1.0


def test_perf_monitor_keeps_bounded_thread_safe_history() -> None:
    monitor = PerfMonitor(max_history=8)

    def record(index: int) -> None:
        monitor.record(
            SegmentMetrics(
                audio_duration=1.0,
                asr_time=index / 1000,
                total_time=index / 1000,
            )
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(record, range(64)))

    stats = monitor.get_stats()
    assert stats["segments"] == 8

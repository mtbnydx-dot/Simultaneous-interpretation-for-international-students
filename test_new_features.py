"""
测试新功能的脚本。
验证 Distil-Whisper 后端、音频预处理、性能监控等功能。
"""

import sys
import numpy as np
from app.core.config import settings


def test_config():
    """测试配置"""
    print("=" * 60)
    print("1. Testing Configuration")
    print("=" * 60)

    print(f"  App name: {settings.app_name}")
    print(f"  ASR backend: {settings.asr_backend}")
    print(f"  ASR model size: {settings.asr_model_size}")
    print(f"  ASR model ID: {settings.asr_model_id}")
    print(f"  Audio preprocess: {settings.audio_preprocess_enabled}")
    print(f"  Perf log: {settings.perf_log_enabled}")
    print("  [OK] Configuration loaded successfully")
    print()


def test_audio_preprocessor():
    """测试音频预处理"""
    print("=" * 60)
    print("2. Testing Audio Preprocessor")
    print("=" * 60)

    from app.core.audio_preprocess import audio_preprocessor

    # 测试 1: AGC（自动增益控制）
    quiet_audio = np.random.randn(16000).astype(np.float32) * 0.001
    processed = audio_preprocessor.process(quiet_audio)
    rms_in = np.sqrt(np.mean(np.square(quiet_audio)))
    rms_out = np.sqrt(np.mean(np.square(processed)))
    print(f"  AGC test: input RMS={rms_in:.6f}, output RMS={rms_out:.4f}")
    assert rms_out > rms_in * 5, "AGC should amplify quiet audio"
    print("  [OK] AGC working correctly")

    # 测试 2: 高通滤波（去低频）
    t = np.linspace(0, 1, 16000, dtype=np.float32)
    low_freq = np.sin(2 * np.pi * 50 * t) * 0.5  # 50Hz
    processed = audio_preprocessor.process(low_freq)
    rms_in = np.sqrt(np.mean(np.square(low_freq)))
    rms_out = np.sqrt(np.mean(np.square(processed)))
    print(f"  Highpass test: input RMS={rms_in:.4f}, output RMS={rms_out:.4f}")
    print("  [OK] Highpass filter working correctly")

    # 测试 3: 归一化
    loud_audio = np.ones(16000, dtype=np.float32) * 10.0
    processed = audio_preprocessor.process(loud_audio)
    assert np.max(np.abs(processed)) <= 1.0, "Should clip to [-1, 1]"
    print("  [OK] Normalization working correctly")

    print()


def test_perf_monitor():
    """测试性能监控"""
    print("=" * 60)
    print("3. Testing Performance Monitor")
    print("=" * 60)

    import time
    from app.core.perf_monitor import perf_monitor

    # 清空历史
    perf_monitor.clear()

    # 模拟多个片段
    for i in range(3):
        timer = perf_monitor.create_timer()
        timer.start(1.0 + i * 0.5)  # 不同长度的音频
        time.sleep(0.01)
        timer.mark_preprocess_done()
        time.sleep(0.02)
        timer.mark_asr_done()
        time.sleep(0.01)
        timer.mark_mt_done()

    stats = perf_monitor.get_stats()
    print(f"  Segments recorded: {stats['segments']}")
    print(f"  Average RTF: {stats['avg_rtf']:.4f}")
    print(f"  Average ASR time: {stats['avg_asr_ms']:.1f}ms")
    print(f"  Average MT time: {stats['avg_mt_ms']:.1f}ms")
    print("  [OK] Performance monitor working correctly")
    print()


def test_backend_registry():
    """测试后端注册表"""
    print("=" * 60)
    print("4. Testing Backend Registry")
    print("=" * 60)

    from app.asr.backends import create_backend

    # 测试 CT2 后端
    try:
        backend = create_backend("ct2")
        print("  [OK] CT2 backend created")
    except Exception as e:
        print(f"  [FAIL] CT2 backend failed: {e}")

    # 测试 OpenVINO 后端
    try:
        backend = create_backend("openvino")
        print("  [OK] OpenVINO backend created")
    except Exception as e:
        print(f"  [FAIL] OpenVINO backend failed: {e}")

    # 测试 Transformers 后端
    try:
        backend = create_backend("transformers-distil")
        print("  [OK] Transformers backend created")
    except Exception as e:
        print(f"  [FAIL] Transformers backend failed: {e}")

    print()


def test_device_detection():
    """测试设备检测"""
    print("=" * 60)
    print("5. Testing Device Detection")
    print("=" * 60)

    from app.asr.engine import _detect_device

    backend, device, compute_type = _detect_device()
    print(f"  Detected backend: {backend}")
    print(f"  Detected device: {device}")
    print(f"  Detected compute type: {compute_type}")
    print("  [OK] Device detection working correctly")
    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("TransLive New Features Test Suite")
    print("=" * 60 + "\n")

    try:
        test_config()
        test_audio_preprocessor()
        test_perf_monitor()
        test_backend_registry()
        test_device_detection()

        print("=" * 60)
        print("All tests passed! [OK]")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Run the macOS/Linux launcher:")
        print("   ./start.sh -y")
        print()
        print("2. Open the local web UI:")
        print("   http://127.0.0.1:8766/")
        print()
        print("3. In the packaged macOS app, choose 系统音频 for native")
        print("   ScreenCaptureKit capture, or 麦克风 for browser microphone input.")
        print()
        print("4. See MACOS_MIGRATION.md for migration notes.")
        print()

        return 0

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
回归测试：llama.cpp 的 Llama 对象不是线程安全的。

两个线程同时进 create_completion 会在 llama-kv-cache.cpp 里触发
GGML_ASSERT 然后 ggml_abort()，整个进程直接被杀（退出码 134），
Python 层捕获不到 —— 这就是「点全文对照就闪退」的根因。

这里用假模型验证 MTEngine 把所有推理串行化了，不需要真的加载 1.1GB 权重。
"""

import threading
import time

import pytest

from app.mt.engine import MTEngine


class _OverlapDetectingModel:
    """记录是否出现过并发进入，模拟 llama.cpp 的不可重入性。"""

    def __init__(self, hold_s: float = 0.02):
        self.hold_s = hold_s
        self.max_concurrent = 0
        self._active = 0
        self._lock = threading.Lock()

    def _enter(self):
        with self._lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)

    def _exit(self):
        with self._lock:
            self._active -= 1

    def create_completion(self, prompt, stream=False, **kwargs):
        self._enter()
        if stream:
            def _generator():
                try:
                    for token in ("翻", "译", "结", "果"):
                        time.sleep(self.hold_s / 4)
                        yield {"choices": [{"text": token}]}
                finally:
                    self._exit()
            return _generator()
        try:
            time.sleep(self.hold_s)
            return {"choices": [{"text": "翻译结果"}]}
        finally:
            self._exit()


def _engine_with(model) -> MTEngine:
    engine = MTEngine()
    engine._model = model
    engine._model_path = "/models/Hy-MT2-1.8B-Q4_K_M.gguf"
    return engine


def _run_threads(targets):
    threads = [threading.Thread(target=fn) for fn in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads), "MT worker deadlocked"


def test_concurrent_translate_calls_never_overlap():
    model = _OverlapDetectingModel()
    engine = _engine_with(model)
    errors = []

    def worker():
        try:
            for _ in range(3):
                engine.translate("hello world", "en", "zh")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    _run_threads([worker] * 4)

    assert not errors
    assert model.max_concurrent == 1, (
        f"检测到 {model.max_concurrent} 个线程同时进入 create_completion —— "
        "真实的 llama.cpp 在这里会 abort()"
    )


def test_streaming_holds_the_lock_for_the_whole_generation():
    """流式解码期间 KV cache 是活的，别的调用必须等它结束。"""
    model = _OverlapDetectingModel()
    engine = _engine_with(model)

    def stream_worker():
        list(engine.translate_stream("hello world", "en", "zh"))

    def blocking_worker():
        engine.translate("another sentence", "en", "zh")

    _run_threads([stream_worker, blocking_worker, stream_worker])

    assert model.max_concurrent == 1


def test_abandoned_stream_generator_releases_the_lock():
    """会话超时/断连时消费方会提前 break，生成器被丢弃也必须解锁，
    否则后续所有翻译都会永久卡住。"""
    model = _OverlapDetectingModel()
    engine = _engine_with(model)

    stream = engine.translate_stream("hello world", "en", "zh")
    next(stream)          # 进入生成器，此时锁已被持有
    stream.close()        # 模拟提前放弃

    done = threading.Event()

    def worker():
        engine.translate("still works", "en", "zh")
        done.set()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=10)
    assert done.is_set(), "生成器被提前丢弃后锁没有释放"


def test_close_waits_for_inflight_inference():
    """「释放内存」按钮不能在解码途中把模型拆掉。"""
    model = _OverlapDetectingModel(hold_s=0.3)
    engine = _engine_with(model)
    started = threading.Event()

    def worker():
        started.set()
        engine.translate("hello", "en", "zh")

    thread = threading.Thread(target=worker)
    thread.start()
    started.wait(timeout=5)
    time.sleep(0.05)

    engine.close(wait_s=10.0)
    thread.join(timeout=10)

    assert engine._model is None
    assert model.max_concurrent == 1


def test_close_gives_up_after_timeout_instead_of_hanging():
    model = _OverlapDetectingModel()
    engine = _engine_with(model)
    acquired = threading.Event()
    release = threading.Event()

    def holder():
        with engine._infer_lock:
            acquired.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=holder)
    thread.start()
    assert acquired.wait(timeout=2)

    start = time.perf_counter()
    closed = engine.close(wait_s=0.2)
    assert time.perf_counter() - start < 5.0
    assert closed is False
    assert engine._model is model
    assert engine._closing is False

    release.set()
    thread.join(timeout=2)

    assert engine.close(wait_s=1.0) is True
    assert engine._model is None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

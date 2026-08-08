"""
VAD (Voice Activity Detection) 模块。

Silero VAD 走 ONNX Runtime（纯 numpy 前后处理，不依赖 PyTorch），RMS 作为回退。

关键约束：Silero v5/v6 在 16kHz 下只接受**恰好 512 个采样点**的帧，并且要携带
上一帧尾部 64 个采样点作为 context。喂任何其它长度都会抛
`ValueError: Provided number of samples is N`。调用方必须按帧对齐再调过来，
`app.core.session.StreamSession` 负责做这个缓冲。
"""

import logging
import shutil
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np

from app.core.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

# 16kHz 下的 Silero 帧参数，写死是因为模型图本身就是按这个尺寸导出的。
FRAME_SAMPLES = 512
CONTEXT_SAMPLES = 64
SUPPORTED_SAMPLE_RATE = 16000
_ONNX_FILENAME = "silero_vad.onnx"


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int = 16000) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly

        gcd = int(np.gcd(source_rate, target_rate))
        return resample_poly(audio, target_rate // gcd, source_rate // gcd).astype(np.float32)
    except Exception:
        ratio = target_rate / max(1, source_rate)
        new_len = max(1, int(round(audio.size * ratio)))
        x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


def _model_cache_dir() -> Path:
    """桌面版模型放在 Application Support，开发模式放 models/。"""
    configured = Path(settings.mt_model_id or "").expanduser()
    if configured.is_absolute():
        return configured.parent
    return PROJECT_ROOT / "models"


def _silero_repo_ref() -> tuple[str, str]:
    """把 'snakers4/silero-vad:v6.2.1' 拆成 (repo, tag)。"""
    raw = (settings.vad_silero_repo or "snakers4/silero-vad:v6.2.1").strip()
    repo, _, tag = raw.partition(":")
    return repo or "snakers4/silero-vad", tag or "master"


def _candidate_onnx_paths() -> list[Path]:
    candidates: list[Path] = []

    explicit = (settings.vad_silero_onnx_path or "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())

    # 打包后的 App 会把权重放进 bundle 资源目录，免掉首次运行的网络依赖
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / _ONNX_FILENAME)

    candidates.append(_model_cache_dir() / _ONNX_FILENAME)

    # pip 安装的 silero-vad 包自带 onnx 权重
    try:
        import silero_vad  # type: ignore

        candidates.append(Path(silero_vad.__file__).parent / "data" / _ONNX_FILENAME)
    except Exception:
        pass

    # 之前用 torch.hub 跑过的机器，缓存里已经有同一份文件，直接复用免下载。
    # 优先匹配配置里钉住的 tag，避免误用 master 之类的旧缓存目录。
    hub_root = Path.home() / ".cache" / "torch" / "hub"
    if hub_root.is_dir():
        repo, tag = _silero_repo_ref()
        repo_slug = repo.replace("/", "_")
        try:
            pinned = hub_root / f"{repo_slug}_{tag}" / "src" / "silero_vad" / "data" / _ONNX_FILENAME
            candidates.append(pinned)
            candidates.extend(
                path
                for path in sorted(hub_root.glob(f"*silero-vad*/src/silero_vad/data/{_ONNX_FILENAME}"))
                if path != pinned
            )
        except OSError:
            pass

    return candidates


def _resolve_onnx_path() -> Path:
    for candidate in _candidate_onnx_paths():
        if not candidate.is_file() or candidate.stat().st_size <= 512 * 1024:
            continue
        try:
            _validate_onnx_hash(candidate)
            return candidate
        except Exception as exc:
            logger.warning("Skipping unverified Silero VAD model %s: %s", candidate, exc)
    return _download_onnx()


def _validate_onnx_hash(path: Path) -> None:
    expected = (settings.vad_silero_sha256 or "").strip()
    if not expected:
        return
    from app.core.model_download import sha256_file

    actual = sha256_file(path)
    if actual.casefold() != expected.casefold():
        raise RuntimeError(f"Silero VAD SHA-256 mismatch (expected {expected}, got {actual})")


def _download_onnx() -> Path:
    repo, tag = _silero_repo_ref()
    url = f"https://raw.githubusercontent.com/{repo}/{tag}/src/silero_vad/data/{_ONNX_FILENAME}"
    target = _model_cache_dir() / _ONNX_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".onnx.part")

    logger.info("Downloading Silero VAD ONNX weights: %s", url)
    with urllib.request.urlopen(url, timeout=30) as response:
        if getattr(response, "status", 200) != 200:
            raise RuntimeError(f"Silero VAD download failed with HTTP {response.status}")
        with tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    if tmp.stat().st_size < 512 * 1024:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded Silero VAD weights look truncated: {url}")

    try:
        _validate_onnx_hash(tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    tmp.replace(target)
    logger.info("Silero VAD ONNX weights cached at %s", target)
    return target


class SileroVAD:
    """Silero VAD (ONNX Runtime)。有状态：帧之间共享 RNN state 和 context。"""

    def __init__(self):
        self._session = None
        self._model_path: Path | None = None
        self._lock = threading.Lock()
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
        self._input = np.empty((1, CONTEXT_SAMPLES + FRAME_SAMPLES), dtype=np.float32)
        self._sample_rate_input = np.array(SUPPORTED_SAMPLE_RATE, dtype=np.int64)

    @property
    def model_path(self) -> str | None:
        return str(self._model_path) if self._model_path else None

    def _ensure_loaded(self):
        if self._session is not None:
            return
        try:
            import onnxruntime
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD 需要 onnxruntime。安装：pip install onnxruntime，"
                "或设置 TRANS_VAD_ENGINE=rms 使用回退方案。"
            ) from exc

        try:
            model_path = _resolve_onnx_path()
            options = onnxruntime.SessionOptions()
            # VAD 每帧只有 512 个采样点，多线程只会带来调度开销和额外唤醒。
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = onnxruntime.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._model_path = model_path
            self.reset_states()
            logger.info("Silero VAD (onnxruntime) loaded from %s", model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Silero VAD: {exc}. 设置 TRANS_VAD_ENGINE=rms 可回退到 RMS 检测。"
            ) from exc

    def reset_states(self) -> None:
        """段落之间必须重置：Silero 是 RNN，跨段沿用状态会污染判断。"""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def _frame_probability(self, frame: np.ndarray) -> float:
        self._input[:, :CONTEXT_SAMPLES] = self._context
        self._input[:, CONTEXT_SAMPLES:] = frame.reshape(1, -1)
        outputs = self._session.run(
            None,
            {
                "input": self._input,
                "state": self._state,
                "sr": self._sample_rate_input,
            },
        )
        probability, state = outputs[0], outputs[1]
        self._state = state.astype(np.float32, copy=False)
        self._context[...] = self._input[:, -CONTEXT_SAMPLES:]
        return float(np.asarray(probability).reshape(-1)[0])

    def speech_probability(self, audio: np.ndarray, sr: int = SUPPORTED_SAMPLE_RATE) -> float:
        """返回该段音频中所有完整帧的最大语音概率。"""
        self._ensure_loaded()

        if sr != SUPPORTED_SAMPLE_RATE:
            audio = _resample_audio(audio, sr, SUPPORTED_SAMPLE_RATE)

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return 0.0

        # 尾巴不足一帧时补零：模型图对输入长度是硬约束，不能直接喂进去。
        remainder = audio.size % FRAME_SAMPLES
        if remainder:
            audio = np.concatenate((audio, np.zeros(FRAME_SAMPLES - remainder, dtype=np.float32)))

        best = 0.0
        with self._lock:
            for start in range(0, audio.size, FRAME_SAMPLES):
                best = max(best, self._frame_probability(audio[start:start + FRAME_SAMPLES]))
        return best

    def detect_speech(self, audio: np.ndarray, sr: int = SUPPORTED_SAMPLE_RATE) -> bool:
        return self.speech_probability(audio, sr) >= settings.vad_silero_threshold


class RMSVAD:
    """简单 RMS 阈值 VAD (回退方案)"""

    def reset_states(self) -> None:
        return None

    def speech_probability(self, audio: np.ndarray, sr: int = SUPPORTED_SAMPLE_RATE) -> float:
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.dot(audio, audio) / audio.size))

    def detect_speech(self, audio: np.ndarray, sr: int = SUPPORTED_SAMPLE_RATE) -> bool:
        if len(audio) == 0:
            return False
        rms = float(np.sqrt(np.dot(audio, audio) / audio.size))
        return rms >= settings.audio_vad_rms_threshold


_vad_instance: SileroVAD | RMSVAD | None = None


def get_vad():
    global _vad_instance
    if _vad_instance is None:
        if settings.vad_engine == "silero":
            try:
                instance = SileroVAD()
                instance._ensure_loaded()
                _vad_instance = instance
            except Exception as exc:
                logger.warning("Silero VAD unavailable, falling back to RMS: %s", exc)
                _vad_instance = RMSVAD()
        else:
            _vad_instance = RMSVAD()
    return _vad_instance

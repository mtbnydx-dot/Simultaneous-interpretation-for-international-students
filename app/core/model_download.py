import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from app.core.config import PROJECT_ROOT, settings
from app.core.model_profiles import select_model_plan

logger = logging.getLogger(__name__)


class ModelIntegrityError(RuntimeError):
    pass


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _min_free_bytes() -> int:
    gb = max(0.0, float(settings.model_min_free_disk_gb or 0.0))
    return int(gb * 1024 * 1024 * 1024)


def ensure_free_space(target_dir: Path, min_bytes: int | None = None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    required = _min_free_bytes() if min_bytes is None else max(0, int(min_bytes))
    if required <= 0:
        return
    usage = shutil.disk_usage(target_dir)
    if usage.free < required:
        raise RuntimeError(
            f"磁盘空间不足：{target_dir} 仅剩 {usage.free / (1024**3):.1f} GB，"
            f"至少需要 {required / (1024**3):.1f} GB。"
        )


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integrity_cache_path(model_path: Path) -> Path:
    return model_path.with_name(f".{model_path.name}.translive-integrity.json")


def _integrity_fingerprint(model_path: Path) -> dict[str, int]:
    stat = model_path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "inode": stat.st_ino,
        "device": stat.st_dev,
    }


def _cached_hash_matches(model_path: Path, expected_sha256: str) -> bool:
    try:
        payload = json.loads(_integrity_cache_path(model_path).read_text(encoding="utf-8"))
        return (
            payload.get("version") == 1
            and payload.get("sha256") == expected_sha256
            and payload.get("fingerprint") == _integrity_fingerprint(model_path)
        )
    except (OSError, ValueError, TypeError):
        return False


def _write_hash_cache(model_path: Path, sha256: str) -> None:
    cache_path = _integrity_cache_path(model_path)
    payload = {
        "version": 1,
        "sha256": sha256,
        "fingerprint": _integrity_fingerprint(model_path),
    }
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        temp_path.replace(cache_path)
    except OSError:
        logger.debug("Unable to persist model integrity cache for %s", model_path, exc_info=True)
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def validate_gguf_file(
    path: str | Path,
    min_size_bytes: int = 128 * 1024 * 1024,
    expected_sha256: str | None = None,
    allow_cached_hash: bool = False,
) -> Path:
    model_path = Path(path).expanduser()
    if not model_path.is_file():
        raise ModelIntegrityError(f"GGUF model file does not exist: {model_path}")
    size = model_path.stat().st_size
    if size < min_size_bytes:
        raise ModelIntegrityError(
            f"GGUF model looks incomplete: {model_path} is only {size / (1024**2):.1f} MB"
        )
    with model_path.open("rb") as f:
        header = f.read(4)
    if header != b"GGUF":
        raise ModelIntegrityError(
            f"GGUF model header is invalid: {model_path}. "
            "The file may be a failed download or the wrong model format."
        )
    if expected_sha256:
        expected = expected_sha256.strip().casefold()
        if allow_cached_hash and _cached_hash_matches(model_path, expected):
            return model_path
        actual_sha256 = sha256_file(model_path).casefold()
        if actual_sha256 != expected:
            _integrity_cache_path(model_path).unlink(missing_ok=True)
            raise ModelIntegrityError(
                f"GGUF model SHA-256 mismatch: {model_path} "
                f"(expected {expected}, got {actual_sha256})"
            )
        _write_hash_cache(model_path, actual_sha256)
    return model_path


def expected_mt_sha256(path: str | Path) -> str | None:
    return (settings.mt_model_sha256 or {}).get(Path(path).name)


def download_hf_snapshot(repo_id: str, *, force: bool = False) -> str:
    """Download a pinned HF snapshot after enforcing a declared size ceiling."""
    from huggingface_hub import HfApi, snapshot_download

    revision = (settings.hf_model_revisions or {}).get(repo_id)
    if not force:
        try:
            return snapshot_download(
                repo_id=repo_id,
                revision=revision,
                token=settings.hf_token,
                local_files_only=True,
            )
        except Exception:
            pass

    max_bytes = int(max(0.1, float(settings.asr_model_max_download_gb)) * 1024**3)
    info = HfApi(token=settings.hf_token).model_info(
        repo_id=repo_id,
        revision=revision,
        files_metadata=True,
    )
    declared_size = sum(int(getattr(item, "size", 0) or 0) for item in (info.siblings or []))
    if declared_size > max_bytes:
        raise RuntimeError(
            f"ASR snapshot is too large: {declared_size / 1024**3:.1f} GB "
            f"exceeds the {max_bytes / 1024**3:.1f} GB limit"
        )
    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=settings.hf_token,
        force_download=force,
    )


def quarantine_bad_model(path: str | Path, reason: str) -> Path | None:
    model_path = Path(path).expanduser()
    if not model_path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = model_path.with_name(f"{model_path.name}.bad-{stamp}")
    logger.warning("Quarantining bad model %s -> %s (%s)", model_path, target, reason)
    _integrity_cache_path(model_path).unlink(missing_ok=True)
    model_path.replace(target)
    # 一个 GGUF 往往超过 1GB。只保留最新一份损坏文件供排查，避免多次修复后
    # Application Support 被历史隔离文件吃满。
    for stale in model_path.parent.glob(f"{model_path.name}.bad-*"):
        if stale != target:
            try:
                stale.unlink()
            except OSError:
                logger.debug("Unable to remove stale quarantined model %s", stale, exc_info=True)
    return target


def remove_model_for_redownload(path: str | Path) -> None:
    model_path = Path(path).expanduser()
    _integrity_cache_path(model_path).unlink(missing_ok=True)
    if model_path.exists():
        logger.info("Removing model before explicit redownload: %s", model_path)
        model_path.unlink()


def download_asr_model() -> None:
    """根据当前配置下载对应的 ASR 模型"""
    backend = settings.asr_backend
    if backend == "auto":
        from app.asr.engine import _detect_device
        backend, _device, _compute_type = _detect_device()

    if backend == "cloud":
        logger.info("ASR backend is cloud; no local ASR model download required")
    elif backend == "qwen3":
        from app.asr.backends.qwen3_backend import resolve_qwen3_model_id

        model_id = resolve_qwen3_model_id(settings.asr_model_id)
        logger.info("Downloading Qwen3-ASR model: %s", model_id)
        path = download_hf_snapshot(model_id)
        logger.info("Qwen3-ASR model downloaded and cached: %s", path)
    elif backend == "mlx":
        from app.asr.backends.mlx_backend import prepare_mlx_model_path, resolve_mlx_model_id

        model_id = resolve_mlx_model_id(settings.asr_model_id)
        logger.info("Downloading MLX ASR model: %s", model_id)
        model_path = prepare_mlx_model_path(model_id)
        logger.info("MLX ASR model downloaded and cached: %s", model_path)
    elif backend in ("transformers-distil", "transformers-whisper"):
        _download_transformers_asr()
    else:
        from faster_whisper import WhisperModel
        model_size = _resolve_ct2_model_size()
        logger.info("Downloading ASR model: %s", model_size)
        WhisperModel(model_size, device="cpu", compute_type=settings.asr_compute_type)
        logger.info("ASR model downloaded and cached")


def _resolve_ct2_model_size() -> str:
    plan = select_model_plan()
    raw = settings.asr_model_id or (
        plan.asr.model_id if plan.asr.backend == "ct2" and plan.asr.model_id else None
    ) or settings.asr_model_size
    if "mlx-community/whisper-large-v3-turbo" in raw:
        return "large-v3-turbo"
    if "/" in raw:
        raw = raw.split("/")[-1]
    if raw.startswith("whisper-"):
        raw = raw[len("whisper-"):]
    return raw


def _download_transformers_asr() -> None:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from app.asr.backends.transformers_backend import DISTIL_WHISPER_MODELS

    backend = settings.asr_backend

    if settings.asr_model_id:
        model_id = settings.asr_model_id
        if "/" not in model_id and not model_id.startswith("distil-"):
            model_id = f"openai/whisper-{model_id}"
    elif backend == "transformers-distil":
        model_id = DISTIL_WHISPER_MODELS.get(
            settings.asr_model_size, DISTIL_WHISPER_MODELS["large-v3"]
        )
    else:
        model_id = "openai/whisper-large-v3-turbo"

    logger.info("Downloading transformers ASR model: %s", model_id)
    AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    AutoProcessor.from_pretrained(model_id)
    logger.info("Transformers ASR model downloaded and cached: %s", model_id)


MT_GGUF_REPOS: tuple[tuple[str, str], ...] = (
    ("hy-mt2", "tencent/Hy-MT2-1.8B-GGUF"),
    ("hunyuan-mt2", "tencent/Hy-MT2-1.8B-GGUF"),
    ("hy-mt1.5", "tencent/HY-MT1.5-1.8B-GGUF"),
    ("hunyuan-mt1.5", "tencent/HY-MT1.5-1.8B-GGUF"),
)
DEFAULT_MT_GGUF_REPO = "tencent/Hy-MT2-1.8B-GGUF"


def resolve_mt_repo(filename: str) -> str:
    """由 GGUF 文件名推断它属于哪个 HF 仓库。"""
    name = Path(filename).name.lower()
    for prefix, repo in MT_GGUF_REPOS:
        if name.startswith(prefix):
            return repo
    return DEFAULT_MT_GGUF_REPO


def download_mt_model(force: bool = False) -> str:
    """
    下载 Hy-MT GGUF 翻译模型（默认 Hy-MT2-1.8B，Apache-2.0）。
    优先从 models/ 目录查找，否则通过 huggingface_hub 下载。
    """
    if settings.mt_backend == "cloud":
        model_id = settings.cloud_mt_model or "cloud"
        logger.info("MT backend is cloud; no local MT model download required")
        return model_id

    model_id = settings.mt_model_id

    # 如果已经是 GGUF 文件路径
    if model_id.endswith(".gguf"):
        target_path = resolve_project_path(model_id)
        if target_path.exists():
            if force:
                remove_model_for_redownload(target_path)
            else:
                try:
                    validate_gguf_file(
                        target_path,
                        expected_sha256=expected_mt_sha256(target_path),
                        allow_cached_hash=True,
                    )
                    logger.info("MT GGUF model found at: %s", target_path)
                    return str(target_path)
                except ModelIntegrityError as exc:
                    quarantine_bad_model(target_path, str(exc))

        # 尝试从 HF 下载 (优先 Python API，支持断点续传)
        filename = Path(model_id).name
        hf_repo = resolve_mt_repo(filename)
        logger.info("Downloading MT GGUF: %s/%s → %s", hf_repo, filename, target_path)

        ensure_free_space(target_path.parent)
        hf_error: Exception | None = None
        try:
            from huggingface_hub import HfApi, hf_hub_download

            info = HfApi(token=settings.hf_token).model_info(hf_repo, files_metadata=True)
            sibling = next(
                (item for item in (info.siblings or []) if getattr(item, "rfilename", None) == filename),
                None,
            )
            declared_size = int(getattr(sibling, "size", 0) or 0)
            max_download = int(max(0.1, float(settings.mt_model_max_download_gb)) * 1024**3)
            if declared_size <= 0:
                raise RuntimeError(f"Hugging Face did not report a size for {hf_repo}/{filename}")
            if declared_size > max_download:
                raise RuntimeError(
                    f"MT model is too large: {declared_size / 1024**3:.1f} GB "
                    f"exceeds the {max_download / 1024**3:.1f} GB limit"
                )
            ensure_free_space(target_path.parent, declared_size + _min_free_bytes())
            result = hf_hub_download(
                repo_id=hf_repo,
                filename=filename,
                local_dir=str(target_path.parent),
                force_download=force,
                token=settings.hf_token,
            )
            resolved = validate_gguf_file(
                Path(result).resolve(),
                expected_sha256=expected_mt_sha256(filename),
            )
            logger.info("MT model downloaded: %s", resolved)
            return str(resolved)
        except ImportError as exc:
            hf_error = exc
            logger.warning("huggingface_hub is unavailable; trying CLI fallback")
        except Exception as exc:
            hf_error = exc
            logger.exception("hf_hub_download failed")

        if getattr(sys, "frozen", False):
            raise RuntimeError(
                f"Failed to download MT model ({hf_repo}/{filename}). "
                f"Check the log above for details, or download manually: "
                f"huggingface-cli download {hf_repo} {filename} --local-dir models/"
            ) from hf_error

        try:
            subprocess.run(
                [sys.executable, "-m", "huggingface_hub", "download", hf_repo, filename,
                 "--local-dir", str(target_path.parent)] + (["--force-download"] if force else []),
                check=True,
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1"},
            )
            validate_gguf_file(target_path, expected_sha256=expected_mt_sha256(target_path))
            logger.info("MT model downloaded: %s", target_path)
            return str(target_path)
        except Exception as exc2:
            raise RuntimeError(
                f"Failed to download MT model. "
                f"Run manually: hf download {hf_repo} {filename} --local-dir models/"
            ) from exc2

    # NLLB / opus-mt 旧路径 (留作回退)
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required. Run: pip install transformers sentencepiece sacremoses torch"
        ) from exc

    logger.info("Downloading MT model: %s (force=%s)", model_id, force)
    AutoTokenizer.from_pretrained(model_id, force_download=force, token=settings.hf_token)
    AutoModelForSeq2SeqLM.from_pretrained(model_id, force_download=force, token=settings.hf_token)
    logger.info("MT model ready: %s", model_id)
    return model_id


def ensure_mt_model_available() -> str:
    if not settings.auto_download_models:
        return settings.mt_model_id
    return download_mt_model()

"""
MT 引擎 —— 本地 Tencent Hy-MT GGUF + llama-cpp-python，或云端 MT 服务。
本地路径统一跨平台：CUDA / Metal / Intel SYCL/Vulkan / CPU。
支持流式 token-by-token 输出。
"""

import atexit
import logging
import os
import platform
import re
import threading
from pathlib import Path
from typing import Any, Generator

from app.core.config import settings
from app.core.languages import (
    AUTO_LANGUAGE_CODE,
    language_name,
    language_name_zh,
    normalize_language_code,
    normalize_source_language,
)

logger = logging.getLogger(__name__)
_GPU_OFFLOAD_SUPPORTED: bool | None = None


class MTError(RuntimeError):
    """Base class for MT engine failures."""


class MTUnavailableError(MTError):
    """Raised when the MT model cannot be loaded or is unavailable."""


class MTTranslationError(MTError):
    """Raised when llama.cpp fails during translation."""


_TEXT_COMPARE_RE = re.compile(r"[\s\.,!?;:，。！？、；：\"'`“”‘’\[\]\(\){}<>《》…~\-–—_]+")
_OUTER_QUOTE_PAIRS = {
    '"': '"',
    "'": "'",
    "`": "`",
    "“": "”",
    "‘": "’",
    "「": "」",
    "『": "』",
}
_PROMPT_ECHO_PREFIXES = (
    "translate the following",
    "output only",
    "use the following terminology",
    "reference translation",
    "source text",
    "target text",
    "请将以下",
    "将以下",
    "只输出",
    "参考下面",
    "源文",
    "原文",
)


def _compare_text(value: str) -> str:
    return _TEXT_COMPARE_RE.sub("", value.casefold())


def _strip_outer_quotes(value: str) -> str:
    text = value.strip()
    changed = True
    while changed and len(text) >= 2:
        changed = False
        for left, right in _OUTER_QUOTE_PAIRS.items():
            if text.startswith(left) and text.endswith(right):
                text = text[1:-1].strip()
                changed = True
                break
    return text


def _strip_label_prefix(value: str, source_lang: str, target_lang: str) -> str:
    src_labels = {language_name(source_lang), language_name_zh(source_lang)}
    tgt_labels = {language_name(target_lang), language_name_zh(target_lang)}
    labels = {
        "translation",
        "translated text",
        "result",
        "output",
        "译文",
        "翻译",
        "结果",
        *src_labels,
        *tgt_labels,
    }
    label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True) if label)
    match = re.match(rf"^\s*(?:{label_pattern})\s*[:：\-]\s*(.+?)\s*$", value, flags=re.IGNORECASE)
    return match.group(1).strip() if match else value.strip()


def clean_translation_output(
    output: str,
    source_text: str = "",
    source_lang: str | None = None,
    target_lang: str | None = None,
) -> str:
    """Conservatively remove labels, stop tokens, prompt echoes, and wrapper quotes."""
    text = (output or "").strip()
    if not text or not settings.mt_output_cleanup_enabled:
        return text

    for stop in settings.mt_stop_tokens or []:
        if isinstance(stop, str) and stop and stop in text:
            text = text.split(stop, 1)[0].strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text).strip()
        text = re.sub(r"\s*```$", "", text).strip()

    src = normalize_language_code(source_lang, settings.source_lang)
    tgt = normalize_language_code(target_lang, settings.target_lang)
    source_norm = _compare_text(source_text or "")
    raw_lines = [line.strip() for line in text.splitlines()]
    non_empty_count = sum(1 for line in raw_lines if line)
    cleaned_lines: list[str] = []

    for line in raw_lines:
        if not line:
            continue
        lowered = line.casefold()
        if any(lowered.startswith(prefix) for prefix in _PROMPT_ECHO_PREFIXES):
            continue
        candidate = _strip_outer_quotes(_strip_label_prefix(line, src, tgt))
        if source_norm and non_empty_count > 1 and _compare_text(candidate) == source_norm:
            continue
        if candidate:
            cleaned_lines.append(candidate)

    cleaned = "\n".join(cleaned_lines).strip() if cleaned_lines else text
    return _strip_outer_quotes(cleaned)


def _llama_supports_gpu() -> bool:
    """检测安装的 llama-cpp-python 是否带 GPU 后端 (CUDA/Metal/SYCL/Vulkan)"""
    global _GPU_OFFLOAD_SUPPORTED
    if _GPU_OFFLOAD_SUPPORTED is not None:
        return _GPU_OFFLOAD_SUPPORTED
    try:
        from llama_cpp import llama_cpp as _llc
        if hasattr(_llc, "llama_supports_gpu_offload"):
            result = bool(_llc.llama_supports_gpu_offload())
            _GPU_OFFLOAD_SUPPORTED = result
            logger.info("llama-cpp-python GPU offload supported: %s", result)
            return result
        logger.info("llama-cpp-python missing llama_supports_gpu_offload attribute")
    except Exception as exc:
        logger.warning("llama-cpp-python GPU detection failed: %s", exc)
    _GPU_OFFLOAD_SUPPORTED = False
    return False


def _intel_gpu_present() -> bool:
    """通过 OpenVINO 探测 Intel GPU 是否在线（仅作为存在性检查，不实际加载）"""
    try:
        import openvino as ov
        return any("GPU" in d for d in ov.Core().available_devices)
    except Exception:
        return False


def _detect_llama_device() -> tuple[str, int, dict]:
    """
    检测最佳 llama.cpp 设备配置。
    Returns: (device_label, n_gpu_layers, extra_kwargs)
    """
    explicit = settings.mt_device
    if explicit != "auto":
        return _device_config(explicit)

    gpu_offload = _llama_supports_gpu()

    # 1) NVIDIA CUDA —— macOS 上不可能有 CUDA，跳过以免仅为这次探测就把 torch
    #    整个装进内存（约 180MB / 1 秒）。
    if platform.system() != "Darwin":
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if gpu_offload:
                    layers = -1 if vram_gb >= 4 else 24
                    logger.info("MT: CUDA available (%.1f GB VRAM), n_gpu_layers=%d", vram_gb, layers)
                    return "cuda", layers, {}
                logger.info("MT: CUDA detected, but llama-cpp-python has no GPU offload; using CPU")
        except ImportError:
            pass

    # 2) Apple Metal —— llama.cpp 的 Metal 后端和 torch 没有关系，
    #    用 platform 判断即可，不要为了一句 is_available() 把 torch 拉进来。
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if gpu_offload:
            logger.info("MT: Apple Metal available")
            return "metal", -1, {}
        logger.info("MT: Apple Silicon detected, but llama-cpp-python has no GPU offload; using CPU")

    # 3) Intel GPU —— 仅当 llama-cpp-python 是带 GPU 后端编译版才启用
    if _intel_gpu_present() and gpu_offload:
        logger.info("MT: Intel GPU available and llama-cpp-python supports GPU offload")
        return "intel_gpu", -1, {}

    logger.info("MT: using CPU backend")
    return "cpu", 0, {}


def _device_config(device: str) -> tuple[str, int, dict]:
    device = (device or "cpu").lower()
    if device in ("cuda", "nvidia"):
        if _llama_supports_gpu():
            return "cuda", -1, {}
        logger.info("MT: CUDA requested, but llama-cpp-python has no GPU offload; using CPU")
        return "cpu", 0, {}
    if device in ("mps", "metal"):
        if _llama_supports_gpu():
            return "metal", -1, {}
        logger.info("MT: Metal requested, but llama-cpp-python has no GPU offload; using CPU")
        return "cpu", 0, {}
    if device == "intel_gpu":
        if _llama_supports_gpu():
            logger.info("MT: explicit Intel GPU; llama-cpp-python supports GPU offload")
            return "intel_gpu", -1, {}
        # 静默降级为 CPU —— 避免向终端用户输出环境配置噪音
        logger.info("MT: Intel GPU requested, llama-cpp-python is CPU-only; using CPU")
        return "cpu", 0, {}
    return "cpu", 0, {}


def _resolve_n_threads(device_label: str) -> int | None:
    """计算 llama.cpp 的 n_threads（仅 CPU 路径有意义）"""
    if device_label != "cpu":
        return None  # GPU 路径 llama.cpp 自行管理
    explicit = settings.mt_n_threads
    if isinstance(explicit, int) and explicit > 0:
        return explicit
    cores = os.cpu_count() or 8
    # 留一半给 ASR/VAD/网络。Core Ultra 这类大小核架构上「物理 P 核数」往往最优，
    # 此处用 cpu_count // 2 做一个保守但合理的默认。
    return max(1, cores // 2)


def _resolve_n_ctx() -> int:
    """计算 n_ctx；为 chat template 预留缓冲"""
    explicit = settings.mt_n_ctx
    if isinstance(explicit, int) and explicit > 0:
        return explicit
    # input + output + 模板/系统提示开销
    realtime_ctx = settings.mt_max_input_tokens + settings.mt_max_new_tokens + 256
    # CJK text can approach one token per character. Using chars // 2 let a
    # 6000-character transcript overflow a 4K context before generation began.
    full_context_input_tokens = max(
        int(settings.mt_max_input_tokens or 0),
        int(settings.mt_full_context_max_chars or 0),
    )
    full_context_ctx = full_context_input_tokens + int(settings.mt_full_context_max_output_tokens or 0) + 512
    return min(8192, max(realtime_ctx, full_context_ctx))


class MTEngine:
    """Hy-MT GGUF 翻译引擎，默认 Hy-MT2，兼容 HY-MT1.5。"""

    def __init__(self):
        self._model: Any = None
        self._device_label: str = "cpu"
        self._n_gpu_layers: int = 0
        self._device_configured = False
        self._n_threads: int | None = None
        self._lock = threading.Lock()
        # llama.cpp 的 Llama 对象不是线程安全的：两个线程同时进 create_completion
        # 会在 llama-kv-cache.cpp 里触发 GGML_ASSERT → ggml_abort()，直接杀掉进程
        # （Python 捕获不到）。所有推理必须串行化。
        # 锁顺序固定为 _infer_lock → _lock，避免死锁。
        self._infer_lock = threading.RLock()
        self._loading = False
        self._closing = False
        self.unavailable_reason: str | None = None
        self._model_path: str | None = None
        self._n_ctx = _resolve_n_ctx()

    @property
    def is_loaded(self) -> bool:
        if settings.mt_backend == "cloud":
            return bool(settings.cloud_enabled and settings.cloud_base_url)
        return self._model is not None

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def device(self) -> str:
        return self._device_label

    @property
    def loaded_pairs(self) -> list[str]:
        if settings.mt_backend == "cloud" and self.is_loaded:
            return [settings.cloud_mt_model or "cloud MT"]
        if self._model:
            # 以文件名为准，别写死型号：现在两代模型都可能被加载
            name = Path(self._model_path).stem if self._model_path else "Hy-MT"
            return [f"{name} (multilingual)"]
        return []

    @property
    def acceleration_info(self) -> dict[str, Any]:
        if settings.mt_backend == "cloud":
            return {
                "status": "cloud",
                "device": "cloud",
                "accelerated": True,
                "gpu_offload_supported": None,
                "n_gpu_layers": 0,
                "note": "翻译由云端服务执行，本地不加载 GGUF。",
            }
        if not self.is_loaded and not self.is_loading:
            return {
                "status": "not_loaded",
                "device": self._device_label,
                "accelerated": False,
                "gpu_offload_supported": None,
                "n_gpu_layers": self._n_gpu_layers,
                "note": None,
            }

        gpu_supported = _llama_supports_gpu()
        accelerated = self._device_label in ("cuda", "metal", "intel_gpu") and self._n_gpu_layers != 0
        status = "accelerated" if accelerated else ("gpu_available" if gpu_supported else "cpu_only")
        note = None
        if status == "cpu_only":
            if platform.system() == "Darwin":
                note = (
                    "当前 llama-cpp-python 不支持 Metal/GPU offload，翻译会使用 CPU，"
                    "延迟可能明显升高。"
                )
            else:
                note = "当前 llama-cpp-python 不支持 GPU offload，翻译会使用 CPU。"
        elif status == "gpu_available" and self._device_label == "cpu":
            note = "llama-cpp-python 支持 GPU offload，但当前配置使用 CPU。"
        return {
            "status": status,
            "device": self._device_label,
            "accelerated": accelerated,
            "gpu_offload_supported": gpu_supported,
            "n_gpu_layers": self._n_gpu_layers,
            "note": note,
        }

    def load_model(self):
        self._loading = True
        self.unavailable_reason = None
        try:
            if settings.mt_backend == "cloud":
                if not settings.cloud_enabled or not settings.cloud_base_url:
                    self.unavailable_reason = (
                        "cloud MT requires TRANS_CLOUD_ENABLED=true and TRANS_CLOUD_BASE_URL"
                    )
                    raise RuntimeError(self.unavailable_reason)
                self._device_label = "cloud"
                self._device_configured = True
                logger.info("MT cloud backend ready: %s", settings.cloud_base_url)
                return

            try:
                self._device_label, self._n_gpu_layers, _extra = _detect_llama_device()
                self._device_configured = True
            except Exception as exc:
                self.unavailable_reason = f"device detection failed: {exc}"
                logger.exception("MT device detection failed")
                raise RuntimeError(self.unavailable_reason) from exc

            try:
                self._ensure_loaded()
            except Exception as exc:
                self.unavailable_reason = f"failed to load MT model: {exc}"
                logger.exception("MT model load failed")
                raise RuntimeError(self.unavailable_reason) from exc

            logger.info(
                "MT model ready (%s, device=%s)",
                Path(self._model_path).name if self._model_path else "Hy-MT",
                self._device_label,
            )
        finally:
            self._loading = False

    def _find_model_file(self) -> str:
        """定位 GGUF 模型文件，不存在则委托 model_download 自动下载"""
        from app.core.config import PROJECT_ROOT
        from app.core.model_download import (
            ModelIntegrityError,
            expected_mt_sha256,
            quarantine_bad_model,
            resolve_project_path,
            validate_gguf_file,
        )

        # 优先使用配置的路径（统一走 resolve_project_path 与 model_download 一致）
        model_id = settings.mt_model_id
        if model_id and model_id.endswith(".gguf"):
            resolved = resolve_project_path(model_id)
            if resolved.exists():
                try:
                    return str(
                        validate_gguf_file(
                            resolved,
                            expected_sha256=expected_mt_sha256(resolved),
                            allow_cached_hash=True,
                        )
                    )
                except ModelIntegrityError as exc:
                    if settings.auto_download_models:
                        quarantine_bad_model(resolved, str(exc))
                    else:
                        raise FileNotFoundError(
                            f"Configured Hy-MT GGUF model is damaged or incomplete: {resolved}. "
                            "Delete it or use the in-app download button to repair it."
                        ) from exc
            if Path(model_id).expanduser().is_absolute():
                if settings.auto_download_models:
                    download_path = self._auto_download_gguf(force=True)
                    if download_path:
                        return download_path
                raise FileNotFoundError(
                    "Configured Hy-MT GGUF model not found: "
                    f"{resolved}. Use the in-app download button or set "
                    "TRANS_MT_MODEL_ID to an existing GGUF file."
                )

        # 扫描 models/ 目录寻找 GGUF 文件 —— 白名单匹配 hy-mt 前缀，避免命中其它 LLM。
        # 排序偏好：Hy-MT2 优先于 HY-MT1.5（新版语言更多、许可证更宽松），
        # 同代之间再按体积从大到小（体积大 = 量化损失小）。
        models_dir = PROJECT_ROOT / "models"
        if models_dir.is_dir():
            def _preference(path: Path) -> tuple[int, int]:
                name = path.name.lower()
                generation = 0 if name.startswith(("hy-mt2", "hunyuan-mt2")) else 1
                return generation, -path.stat().st_size

            candidates = sorted(
                (p for p in models_dir.glob("*.gguf") if p.name.lower().startswith("hy-mt")),
                key=_preference,
            )
            if candidates:
                for candidate in candidates:
                    try:
                        return str(
                            validate_gguf_file(
                                candidate,
                                expected_sha256=expected_mt_sha256(candidate),
                                allow_cached_hash=True,
                            )
                        )
                    except ModelIntegrityError as exc:
                        logger.warning("Skipping invalid GGUF candidate %s: %s", candidate, exc)
                if settings.auto_download_models:
                    logger.warning("All local HY-MT GGUF candidates were invalid; downloading a fresh copy")
                    download_path = self._auto_download_gguf(force=True)
                    if download_path:
                        return download_path

        # ── 自动下载（统一委托到 model_download） ──
        if settings.auto_download_models:
            download_path = self._auto_download_gguf()
            if download_path:
                return download_path

        raise FileNotFoundError(
            "No valid Hy-MT GGUF model found. Use the in-app model download button "
            "or set TRANS_MT_MODEL_ID to an existing GGUF file."
        )

    def _auto_download_gguf(self, force: bool = False) -> str | None:
        """委托给 model_download.download_mt_model（单一下载实现）"""
        try:
            from app.core.model_download import download_mt_model
            return download_mt_model(force=force)
        except Exception as exc:
            logger.warning("Auto-download failed: %s", exc)
            return None

    def _ensure_loaded(self):
        if settings.mt_backend == "cloud":
            if not settings.cloud_enabled or not settings.cloud_base_url:
                raise RuntimeError("Cloud MT is not configured")
            return

        with self._lock:
            if self._model is not None:
                return

            # Translation endpoints can lazy-load without going through load_model().
            # Configure the device here too, otherwise Metal-capable Macs use CPU defaults.
            if not self._device_configured:
                self._device_label, self._n_gpu_layers, _extra = _detect_llama_device()
                self._device_configured = True

            model_path = self._find_model_file()

            try:
                from llama_cpp import Llama
            except ImportError:
                raise RuntimeError(
                    "llama-cpp-python is required for local Hy-MT translation. "
                    "Install: pip install llama-cpp-python"
                )

            self._n_threads = _resolve_n_threads(self._device_label)
            n_ctx = _resolve_n_ctx()
            self._n_ctx = n_ctx
            logger.info(
                "Loading Hy-MT GGUF: %s (device=%s, n_gpu_layers=%d, n_threads=%s, n_ctx=%d)",
                model_path, self._device_label, self._n_gpu_layers, self._n_threads, n_ctx,
            )

            self._model = Llama(
                model_path=model_path,
                n_gpu_layers=self._n_gpu_layers,
                n_ctx=n_ctx,
                n_threads=self._n_threads,
                verbose=False,
            )
            self._model_path = model_path
            self.unavailable_reason = None
            logger.info("Hy-MT GGUF loaded successfully")

    def _model_generation(self) -> str:
        """
        判断当前加载的是哪一代 Hy-MT。两代的官方 prompt 模板不同：
        HY-MT1.5 的模板同时点明源语言和目标语言，Hy-MT2 的官方模板只写目标语言。
        用错模板不会报错，但会明显影响译文质量，所以按文件名分发。
        """
        name = Path(self._model_path or settings.mt_model_id or "").name.lower()
        return "v2" if name.startswith(("hy-mt2", "hunyuan-mt2")) else "v1"

    @staticmethod
    def _glossary_pairs(glossary: dict[str, str] | None) -> list[tuple[str, str]]:
        effective = glossary if glossary is not None else getattr(settings, "mt_glossary", None)
        pairs: list[tuple[str, str]] = []
        if effective:
            for k, v in effective.items():
                if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                    pairs.append((k.strip(), v.strip()))
        return pairs

    @staticmethod
    def _glossary_prefix(pairs: list[tuple[str, str]], chinese: bool) -> str:
        if not pairs:
            return ""
        if chinese:
            body = "\n".join(f"{source} 翻译成 {target}" for source, target in pairs)
            return f"参考下面的翻译：\n{body}\n\n"
        body = "\n".join(f"{source} -> {target}" for source, target in pairs)
        return f"Use the following terminology exactly:\n{body}\n\n"

    def _build_prompt(
        self, text: str, src: str, tgt: str,
        glossary: dict[str, str] | None = None,
    ) -> str:
        """按当前模型这一代的官方模板构建裸 prompt。"""
        src = normalize_language_code(src, settings.source_lang)
        tgt = normalize_language_code(tgt, settings.target_lang)
        src_name = language_name(src)
        tgt_name = language_name(tgt)
        src_name_zh = language_name_zh(src)
        tgt_name_zh = language_name_zh(tgt)

        pairs = self._glossary_pairs(glossary)
        chinese = src == "zh" or tgt == "zh"
        glossary_text = self._glossary_prefix(pairs, chinese)

        if self._model_generation() == "v2":
            # Hy-MT2 官方模板：只指定目标语言，不提源语言。
            if chinese:
                return (
                    f"{glossary_text}"
                    f"把下面的文本翻译成{tgt_name_zh}，"
                    f"注意只输出翻译结果，不要输出任何额外解释：\n\n"
                    f"{text}"
                )
            return (
                f"{glossary_text}"
                f"Translate the following text into {tgt_name}. "
                f"Note that you should only output the translated result "
                f"without any additional explanation:\n\n"
                f"{text}"
            )

        if chinese:
            return (
                f"{glossary_text}"
                f"请将以下{src_name_zh}文本翻译为{tgt_name_zh}。"
                f"只输出{tgt_name_zh}译文，不要额外解释，不要保留原文：\n\n"
                f"{text}"
            )
        return (
            f"{glossary_text}"
            f"Translate the following {src_name} segment into {tgt_name}. "
            f"Output only the {tgt_name} translation, without explanations or the source text.\n\n"
            f"{text}"
        )

    def _build_transcript_prompt(
        self,
        segments: list[dict[str, Any]],
        src: str,
        tgt: str,
        glossary: dict[str, str] | None = None,
    ) -> str:
        """Build a full-context prompt that asks the model to preserve line IDs."""
        src = normalize_source_language(src, settings.source_lang)
        tgt = normalize_language_code(tgt, settings.target_lang)
        mixed_source = src == AUTO_LANGUAGE_CODE
        src_name = "mixed languages" if mixed_source else language_name(src)
        tgt_name = language_name(tgt)
        src_name_zh = "多种语言" if mixed_source else language_name_zh(src)
        tgt_name_zh = language_name_zh(tgt)
        body_lines = []
        for item in segments:
            item_lang = normalize_source_language(item.get("source_lang"), AUTO_LANGUAGE_CODE)
            lang_hint = ""
            if mixed_source and item_lang != AUTO_LANGUAGE_CODE:
                lang_hint = f"<{language_name(item_lang)}> "
            body_lines.append(f"[{item['index']}] {lang_hint}{item['text']}")
        body = "\n".join(body_lines)

        effective_glossary = glossary if glossary is not None else getattr(settings, "mt_glossary", None)
        glossary_pairs: list[tuple[str, str]] = []
        if effective_glossary:
            for k, v in effective_glossary.items():
                if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                    glossary_pairs.append((k.strip(), v.strip()))

        if src == "zh" or tgt == "zh":
            glossary_text = ""
            if glossary_pairs:
                glossary_text = (
                    "参考下面的翻译：\n"
                    + "\n".join(
                        f"{source_term} 翻译成 {target_term}"
                        for source_term, target_term in glossary_pairs
                    )
                    + "\n\n"
                )
            return (
                f"{glossary_text}"
                f"请将下面完整转写从{src_name_zh}翻译为{tgt_name_zh}。"
                f"请利用全文上下文理解代词、术语和前后句关系。"
                f"保持输入行数、顺序和编号，每行只输出对应编号和{tgt_name_zh}译文；"
                f"不要解释，不要输出原文。\n\n"
                f"{body}"
            )

        glossary_text = ""
        if glossary_pairs:
            glossary_text = (
                "Use the following terminology exactly:\n"
                + "\n".join(f"{source_term} -> {target_term}" for source_term, target_term in glossary_pairs)
                + "\n\n"
            )
        return (
            f"{glossary_text}"
            f"Translate the complete transcript from {src_name} to {tgt_name}. "
            f"Use the entire transcript as context for pronouns, terminology, and continuity. "
            f"Preserve the input line count, order, and line IDs. "
            f"Output one line per input line as \"[1] translation\". "
            f"Do not explain and do not output the source text.\n\n"
            f"{body}"
        )

    def _prompt_token_count(self, prompt: str) -> int:
        tokenizer = getattr(self._model, "tokenize", None)
        if callable(tokenizer):
            payload = prompt.encode("utf-8")
            try:
                return len(tokenizer(payload, add_bos=True, special=True))
            except TypeError:
                return len(tokenizer(payload, add_bos=True))
        # Conservative fallback for test doubles and unusual llama bindings.
        return len(prompt)

    def _fit_transcript_prompt(
        self,
        segments: list[dict[str, Any]],
        src: str,
        tgt: str,
        glossary: dict[str, str] | None,
        max_output_tokens: int,
    ) -> tuple[str, list[dict[str, Any]], bool]:
        budget = max(128, int(self._n_ctx) - max_output_tokens - 64)
        fitted = [dict(item) for item in segments]
        truncated = False

        while fitted:
            prompt = self._build_transcript_prompt(fitted, src, tgt, glossary=glossary)
            if self._prompt_token_count(prompt) <= budget:
                return prompt, fitted, truncated
            truncated = True
            tail = fitted[-1]
            if len(tail["text"]) > 96:
                tail["text"] = tail["text"][: max(64, len(tail["text"]) // 2)].rstrip()
            else:
                fitted.pop()

        raise MTTranslationError("Transcript is too large for the configured MT context window")

    def _fit_document_prompt(
        self,
        text: str,
        src: str,
        tgt: str,
        glossary: dict[str, str] | None,
        max_output_tokens: int,
    ) -> tuple[str, str, bool]:
        """Fit one continuous document into the model context without line alignment."""
        source = (text or "").strip()
        if not source:
            return self._build_prompt("", src, tgt, glossary=glossary), "", False

        budget = max(128, int(self._n_ctx) - max_output_tokens - 64)
        prompt = self._build_prompt(source, src, tgt, glossary=glossary)
        if self._prompt_token_count(prompt) <= budget:
            return prompt, source, False

        low = 1
        high = len(source)
        best_text = ""
        best_prompt = ""
        while low <= high:
            midpoint = (low + high) // 2
            candidate = source[:midpoint].rstrip()
            candidate_prompt = self._build_prompt(candidate, src, tgt, glossary=glossary)
            if candidate and self._prompt_token_count(candidate_prompt) <= budget:
                best_text = candidate
                best_prompt = candidate_prompt
                low = midpoint + 1
            else:
                high = midpoint - 1

        if not best_text:
            raise MTTranslationError("Document is too large for the configured MT context window")
        return best_prompt, best_text, True

    def _document_output_token_limit(self, source: str) -> int:
        """Size document output to its input instead of always decoding to the global cap."""
        configured = max(
            1,
            int(settings.mt_full_context_max_output_tokens or settings.mt_max_new_tokens),
        )
        source_tokens = max(1, self._prompt_token_count(source))
        return min(configured, max(128, source_tokens * 2 + 64))

    def _raise_if_closing(self) -> None:
        if self._closing:
            raise MTUnavailableError("MT engine is releasing model resources")

    def _resolve_stop(self) -> list[str] | None:
        """额外停止符；GGUF 中的 EOS token 由 llama.cpp 自动处理"""
        stops = settings.mt_stop_tokens
        if stops:
            return list(stops)
        return None

    def clean_translation_output(
        self,
        output: str,
        source_text: str = "",
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> str:
        return clean_translation_output(output, source_text, source_lang, target_lang)

    def translate(
        self, text: str,
        source_lang: str | None = None,
        target_lang: str | None = None,
        glossary: dict[str, str] | None = None,
    ) -> str:
        """同步翻译，返回完整译文"""
        text = (text or "").strip()
        if not text:
            return ""

        src = normalize_language_code(source_lang, settings.source_lang)
        tgt = normalize_language_code(target_lang, settings.target_lang)
        if src == tgt:
            return text

        if settings.mt_backend == "cloud":
            try:
                from app.core import cloud_client

                return clean_translation_output(
                    cloud_client.translate(text, src, tgt, glossary=glossary),
                    source_text=text,
                    source_lang=src,
                    target_lang=tgt,
                )
            except Exception as exc:
                reason = str(exc)
                self.unavailable_reason = reason
                logger.exception("Cloud MT failed for %s->%s", src, tgt)
                raise MTTranslationError(reason) from exc

        with self._infer_lock:
            self._raise_if_closing()
            try:
                self._ensure_loaded()
            except Exception as exc:
                reason = str(exc)
                self.unavailable_reason = reason
                logger.error("MT unavailable for %s->%s: %s", src, tgt, reason)
                raise MTUnavailableError(reason) from exc

            try:
                prompt = self._build_prompt(text, src, tgt, glossary=glossary)
                response = self._model.create_completion(
                    prompt=prompt,
                    max_tokens=settings.mt_max_new_tokens,
                    temperature=settings.mt_temperature,
                    stop=self._resolve_stop(),
                )
                translated = response["choices"][0]["text"].strip()
                return clean_translation_output(
                    translated,
                    source_text=text,
                    source_lang=src,
                    target_lang=tgt,
                )
            except Exception as exc:
                logger.exception("MT translation failed for %s->%s", src, tgt)
                raise MTTranslationError(str(exc)) from exc

    def translate_transcript(
        self,
        segments: list[dict[str, Any]],
        source_lang: str | None = None,
        target_lang: str | None = None,
        glossary: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Translate a whole transcript with full context while preserving per-line comparison."""
        normalized_segments = [
            {
                "index": int(item["index"]),
                "segment_id": item.get("segment_id"),
                "text": str(item.get("text") or "").strip(),
                "source_lang": normalize_source_language(item.get("source_lang"), AUTO_LANGUAGE_CODE),
            }
            for item in segments
            if str(item.get("text") or "").strip()
        ]
        if not normalized_segments:
            return {
                "items": [],
                "full_original_text": "",
                "full_translated_text": "",
                "aligned": True,
                "warning": None,
            }

        src = normalize_source_language(source_lang, settings.source_lang)
        tgt = normalize_language_code(target_lang, settings.target_lang)
        full_source = "\n".join(item["text"] for item in normalized_segments)
        if src != AUTO_LANGUAGE_CODE and src == tgt:
            return self._format_transcript_translation(normalized_segments, full_source, full_source, None)

        if settings.mt_backend == "cloud":
            try:
                from app.core import cloud_client

                translated = cloud_client.translate(full_source, src, tgt, glossary=glossary)
                cleaned = clean_translation_output(
                    translated,
                    source_text=full_source,
                    source_lang=src,
                    target_lang=tgt,
                )
                return self._format_transcript_translation(normalized_segments, full_source, cleaned, None)
            except Exception as exc:
                reason = str(exc)
                self.unavailable_reason = reason
                logger.exception("Cloud transcript MT failed for %s->%s", src, tgt)
                raise MTTranslationError(reason) from exc

        with self._infer_lock:
            self._raise_if_closing()
            try:
                self._ensure_loaded()
            except Exception as exc:
                reason = str(exc)
                self.unavailable_reason = reason
                logger.error("MT unavailable for transcript %s->%s: %s", src, tgt, reason)
                raise MTUnavailableError(reason) from exc

            try:
                max_output_tokens = max(
                    1,
                    int(settings.mt_full_context_max_output_tokens or settings.mt_max_new_tokens),
                )
                prompt, fitted_segments, context_truncated = self._fit_transcript_prompt(
                    normalized_segments,
                    src,
                    tgt,
                    glossary,
                    max_output_tokens,
                )
                full_source = "\n".join(item["text"] for item in fitted_segments)
                response = self._model.create_completion(
                    prompt=prompt,
                    max_tokens=max_output_tokens,
                    temperature=settings.mt_temperature,
                    stop=self._resolve_stop(),
                )
                translated = response["choices"][0]["text"].strip()
                cleaned = clean_translation_output(
                    translated,
                    source_text=full_source,
                    source_lang=src,
                    target_lang=tgt,
                )
                result = self._format_transcript_translation(
                    fitted_segments,
                    full_source,
                    cleaned,
                    "context_truncated" if context_truncated else None,
                )
                result["context_truncated"] = context_truncated
                return result
            except Exception as exc:
                logger.exception("Transcript MT failed for %s->%s", src, tgt)
                raise MTTranslationError(str(exc)) from exc

    def translate_document(
        self,
        text: str,
        source_lang: str | None = None,
        target_lang: str | None = None,
        glossary: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Translate a transcript as one continuous document, without per-segment alignment."""
        source = (text or "").strip()
        src = normalize_source_language(source_lang, settings.source_lang)
        tgt = normalize_language_code(target_lang, settings.target_lang)
        empty_result = {
            "full_original_text": source,
            "full_translated_text": "",
            "warning": "empty_output" if source else None,
            "context_truncated": False,
        }
        if not source:
            return empty_result
        if src != AUTO_LANGUAGE_CODE and src == tgt:
            return {
                **empty_result,
                "full_translated_text": source,
                "warning": None,
            }

        if settings.mt_backend == "cloud":
            try:
                from app.core import cloud_client

                translated = cloud_client.translate(source, src, tgt, glossary=glossary)
                cleaned = clean_translation_output(
                    translated,
                    source_text=source,
                    source_lang=src,
                    target_lang=tgt,
                )
                return {
                    **empty_result,
                    "full_translated_text": cleaned,
                    "warning": None if cleaned else "empty_output",
                }
            except Exception as exc:
                reason = str(exc)
                self.unavailable_reason = reason
                logger.exception("Cloud document MT failed for %s->%s", src, tgt)
                raise MTTranslationError(reason) from exc

        with self._infer_lock:
            self._raise_if_closing()
            try:
                self._ensure_loaded()
            except Exception as exc:
                reason = str(exc)
                self.unavailable_reason = reason
                logger.error("MT unavailable for document %s->%s: %s", src, tgt, reason)
                raise MTUnavailableError(reason) from exc

            try:
                max_output_tokens = self._document_output_token_limit(source)
                prompt, fitted_source, context_truncated = self._fit_document_prompt(
                    source,
                    src,
                    tgt,
                    glossary,
                    max_output_tokens,
                )
                response = self._model.create_completion(
                    prompt=prompt,
                    max_tokens=max_output_tokens,
                    temperature=settings.mt_temperature,
                    stop=self._resolve_stop(),
                )
                translated = response["choices"][0]["text"].strip()
                cleaned = clean_translation_output(
                    translated,
                    source_text=fitted_source,
                    source_lang=src,
                    target_lang=tgt,
                )
                return {
                    "full_original_text": fitted_source,
                    "full_translated_text": cleaned,
                    "warning": "empty_output" if not cleaned else (
                        "context_truncated" if context_truncated else None
                    ),
                    "context_truncated": context_truncated,
                }
            except Exception as exc:
                logger.exception("Document MT failed for %s->%s", src, tgt)
                raise MTTranslationError(str(exc)) from exc

    def _format_transcript_translation(
        self,
        segments: list[dict[str, Any]],
        full_source: str,
        translated_output: str,
        initial_warning: str | None,
    ) -> dict[str, Any]:
        translations, warning = self._parse_transcript_output(translated_output, len(segments))
        warning = initial_warning or warning
        items = []
        for pos, item in enumerate(segments):
            translated_text = translations[pos] if pos < len(translations) else ""
            items.append({
                "index": item["index"],
                "segment_id": item.get("segment_id"),
                "original_text": item["text"],
                "translated_text": translated_text,
            })
        return {
            "items": items,
            "full_original_text": full_source,
            "full_translated_text": translated_output,
            "aligned": warning is None,
            "warning": warning,
        }

    def _parse_transcript_output(self, output: str, expected_count: int) -> tuple[list[str], str | None]:
        lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
        if expected_count <= 0:
            return [], None
        if not lines:
            return [""] * expected_count, "empty_output"

        numbered: dict[int, str] = {}
        loose: list[str] = []
        for line in lines:
            match = re.match(
                r"^\s*[\[\(【]?\s*(\d{1,4})\s*[\]\)】]?\s*[\.:：、\-\s]\s*(.+?)\s*$",
                line,
            )
            if match:
                index = int(match.group(1))
                text = match.group(2).strip()
                if 1 <= index <= expected_count:
                    numbered[index] = text
                    continue
            loose.append(line)

        if numbered:
            translations = [numbered.get(index, "") for index in range(1, expected_count + 1)]
            warning = None if all(translations) else "alignment_partial"
            return translations, warning

        if len(loose) == expected_count:
            return loose, None

        translations = [""] * expected_count
        if loose:
            for idx, text in enumerate(loose[:expected_count]):
                translations[idx] = text
            if len(loose) > expected_count:
                translations[-1] = "\n".join([translations[-1], *loose[expected_count:]]).strip()
        return translations, "alignment_mismatch"

    def translate_stream(
        self, text: str,
        source_lang: str | None = None,
        target_lang: str | None = None,
        glossary: dict[str, str] | None = None,
    ) -> Generator[str, None, None]:
        """流式翻译，逐 token 产出"""
        text = (text or "").strip()
        if not text:
            return

        src = normalize_language_code(source_lang, settings.source_lang)
        tgt = normalize_language_code(target_lang, settings.target_lang)
        if src == tgt:
            yield text
            return

        if settings.mt_backend == "cloud":
            try:
                from app.core import cloud_client

                yield from cloud_client.translate_stream(text, src, tgt, glossary=glossary)
                return
            except Exception as exc:
                reason = str(exc)
                self.unavailable_reason = reason
                logger.exception("Cloud MT stream failed for %s->%s", src, tgt)
                raise MTTranslationError(reason) from exc

        # 整个生成过程都必须持锁：llama.cpp 的 KV cache 在流式解码期间是活的。
        # 生成器被提前丢弃时 CPython 会抛 GeneratorExit，with 语句照样会释放锁。
        with self._infer_lock:
            self._raise_if_closing()
            try:
                self._ensure_loaded()
            except Exception as exc:
                reason = str(exc)
                self.unavailable_reason = reason
                logger.error("MT unavailable for %s->%s: %s", src, tgt, reason)
                raise MTUnavailableError(reason) from exc

            try:
                prompt = self._build_prompt(text, src, tgt, glossary=glossary)
                stream = self._model.create_completion(
                    prompt=prompt,
                    max_tokens=settings.mt_max_new_tokens,
                    temperature=settings.mt_temperature,
                    stream=True,
                    stop=self._resolve_stop(),
                )
                for chunk in stream:
                    choices = chunk.get("choices", [])
                    if choices:
                        token = choices[0].get("text", "")
                        if token:
                            yield token
            except GeneratorExit:
                raise
            except Exception as exc:
                logger.exception("MT streaming failed for %s->%s", src, tgt)
                raise MTTranslationError(str(exc)) from exc

    def close(self, wait_s: float = 60.0) -> bool:
        """释放 GGUF 模型。会先等待进行中的推理结束，避免在解码途中拆掉模型。"""
        with self._lock:
            self._closing = True
        acquired = self._infer_lock.acquire(timeout=max(0.0, wait_s))
        if not acquired:
            logger.warning("MT close(): inference still running after %.1fs; model kept intact", wait_s)
            with self._lock:
                self._closing = False
            return False
        try:
            with self._lock:
                if self._model is not None:
                    del self._model
                    self._model = None
                self._model_path = None
                self._device_label = "cpu"
                self._n_gpu_layers = 0
                self._device_configured = False
                self._n_threads = None
                self._closing = False
        finally:
            self._infer_lock.release()
        return True


mt_engine = MTEngine()


@atexit.register
def _close_mt_engine_at_exit() -> None:
    """
    llama.cpp 的 Metal 后端有静态析构函数，如果 Llama 对象活到 C 运行时
    退出阶段，ggml-metal-device.m 会断言 [rsets->data count] == 0 失败并
    abort()，表现为退出码 134。atexit 在 Py_Finalize 期间跑，早于静态析构，
    所以在这里兜底释放即可。实测：不释放 → 134，释放 → 0。
    """
    try:
        mt_engine.close(wait_s=5.0)
    except Exception:
        pass

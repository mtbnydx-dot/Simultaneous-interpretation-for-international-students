"""
MT 引擎 —— 基于 Tencent HY-MT1.5 GGUF + llama-cpp-python。
统一跨平台：CUDA / Metal / Intel SYCL/Vulkan / CPU，一个模型文件覆盖所有平台。
支持流式 token-by-token 输出。
"""

import logging
import os
import platform
import threading
from typing import Any, Generator

from app.core.config import settings

logger = logging.getLogger(__name__)
_GPU_OFFLOAD_SUPPORTED: bool | None = None


class MTError(RuntimeError):
    """Base class for MT engine failures."""


class MTUnavailableError(MTError):
    """Raised when the MT model cannot be loaded or is unavailable."""


class MTTranslationError(MTError):
    """Raised when llama.cpp fails during translation."""


# HY-MT1.5 支持的语种 → 模型可理解的语言名称
_LANG_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "ar": "Arabic",
    "pt": "Portuguese",
    "it": "Italian",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "cs": "Czech",
    "nl": "Dutch",
}

_LANG_NAMES_ZH: dict[str, str] = {
    "en": "英语",
    "zh": "中文",
    "ja": "日语",
    "ko": "韩语",
    "de": "德语",
    "fr": "法语",
    "es": "西班牙语",
    "ru": "俄语",
    "ar": "阿拉伯语",
    "pt": "葡萄牙语",
    "it": "意大利语",
    "th": "泰语",
    "vi": "越南语",
    "id": "印尼语",
    "hi": "印地语",
    "tr": "土耳其语",
    "pl": "波兰语",
    "cs": "捷克语",
    "nl": "荷兰语",
}


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

    # 1) NVIDIA CUDA
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            if gpu_offload:
                layers = -1 if vram_gb >= 4 else 24
                logger.info("MT: CUDA available (%.1f GB VRAM), n_gpu_layers=%d", vram_gb, layers)
                return "cuda", layers, {}
            logger.info("MT: CUDA detected, but llama-cpp-python has no GPU offload; using CPU")
    except ImportError:
        pass

    # 2) Apple Metal
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if gpu_offload:
                logger.info("MT: Apple Metal (MPS) available")
                return "metal", -1, {}
            logger.info("MT: Apple Metal detected, but llama-cpp-python has no GPU offload; using CPU")
    except ImportError:
        pass

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
    return settings.mt_max_input_tokens + settings.mt_max_new_tokens + 256


class MTEngine:
    """HY-MT1.5 GGUF 翻译引擎"""

    def __init__(self):
        self._model: Any = None
        self._device_label: str = "cpu"
        self._n_gpu_layers: int = 0
        self._n_threads: int | None = None
        self._lock = threading.Lock()
        self._loading = False
        self.unavailable_reason: str | None = None
        self._model_path: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def _device(self) -> str:
        return self._device_label

    @property
    def loaded_pairs(self) -> list[str]:
        if self._model:
            return ["HY-MT1.5-1.8B (multilingual)"]
        return []

    @property
    def acceleration_info(self) -> dict[str, Any]:
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
            try:
                self._device_label, self._n_gpu_layers, _extra = _detect_llama_device()
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

            logger.info("MT model ready (HY-MT1.5-1.8B, device=%s)", self._device_label)
        finally:
            self._loading = False

    def _find_model_file(self) -> str:
        """定位 GGUF 模型文件，不存在则委托 model_download 自动下载"""
        from pathlib import Path
        from app.core.config import PROJECT_ROOT
        from app.core.model_download import resolve_project_path

        # 优先使用配置的路径（统一走 resolve_project_path 与 model_download 一致）
        model_id = settings.mt_model_id
        if model_id and model_id.endswith(".gguf"):
            resolved = resolve_project_path(model_id)
            if resolved.exists():
                return str(resolved)
            if Path(model_id).expanduser().is_absolute():
                if settings.auto_download_models:
                    download_path = self._auto_download_gguf()
                    if download_path:
                        return download_path
                raise FileNotFoundError(
                    "Configured HY-MT1.5 GGUF model not found: "
                    f"{resolved}. Use the in-app download button or set "
                    "TRANS_MT_MODEL_ID to an existing GGUF file."
                )

        # 扫描 models/ 目录寻找 GGUF 文件 —— 白名单匹配 hy-mt 前缀，避免命中其它 LLM
        models_dir = PROJECT_ROOT / "models"
        if models_dir.is_dir():
            candidates = sorted(
                (p for p in models_dir.glob("*.gguf") if p.name.lower().startswith("hy-mt")),
                key=lambda p: -p.stat().st_size,
            )
            if candidates:
                return str(candidates[0])

        # ── 自动下载（统一委托到 model_download） ──
        if settings.auto_download_models:
            download_path = self._auto_download_gguf()
            if download_path:
                return download_path

        raise FileNotFoundError(
            "No HY-MT1.5 GGUF model found. Download it:\n"
            "  huggingface-cli download tencent/HY-MT1.5-1.8B-GGUF \\\n"
            "    HY-MT1.5-1.8B-Q4_K_M.gguf --local-dir models/\n"
            "Or set TRANS_MT_MODEL_ID to the GGUF file path."
        )

    def _auto_download_gguf(self) -> str | None:
        """委托给 model_download.download_mt_model（单一下载实现）"""
        try:
            from app.core.model_download import download_mt_model
            return download_mt_model()
        except Exception as exc:
            logger.warning("Auto-download failed: %s", exc)
            return None

    def _ensure_loaded(self):
        with self._lock:
            if self._model is not None:
                return

            model_path = self._find_model_file()

            try:
                from llama_cpp import Llama
            except ImportError:
                raise RuntimeError(
                    "llama-cpp-python is required for HY-MT1.5. "
                    "Install: pip install llama-cpp-python"
                )

            self._n_threads = _resolve_n_threads(self._device_label)
            n_ctx = _resolve_n_ctx()
            logger.info(
                "Loading HY-MT1.5 GGUF: %s (device=%s, n_gpu_layers=%d, n_threads=%s, n_ctx=%d)",
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
            logger.info("HY-MT1.5 GGUF loaded successfully")

    def _build_prompt(
        self, text: str, src: str, tgt: str,
        glossary: dict[str, str] | None = None,
    ) -> str:
        """按 HY-MT1.5 官方模板构建裸 prompt。"""
        tgt_name = _LANG_NAMES.get(tgt, tgt)
        tgt_name_zh = _LANG_NAMES_ZH.get(tgt, tgt_name)

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
                f"将以下文本翻译为{tgt_name_zh}，注意只需要输出翻译后的结果，不要额外解释：\n\n"
                f"{text}"
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
            f"Translate the following segment into {tgt_name}, without additional explanation.\n\n"
            f"{text}"
        )

    def _resolve_stop(self) -> list[str] | None:
        """额外停止符；GGUF 中的 EOS token 由 llama.cpp 自动处理"""
        stops = settings.mt_stop_tokens
        if stops:
            return list(stops)
        return None

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

        src = source_lang or settings.source_lang
        tgt = target_lang or settings.target_lang
        if src == tgt:
            return text

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
                temperature=0.0,
                stop=self._resolve_stop(),
            )
            translated = response["choices"][0]["text"].strip()
            return translated
        except Exception as exc:
            logger.exception("MT translation failed for %s->%s", src, tgt)
            raise MTTranslationError(str(exc)) from exc

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

        src = source_lang or settings.source_lang
        tgt = target_lang or settings.target_lang
        if src == tgt:
            yield text
            return

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
                temperature=0.0,
                stream=True,
                stop=self._resolve_stop(),
            )
            for chunk in stream:
                choices = chunk.get("choices", [])
                if choices:
                    token = choices[0].get("text", "")
                    if token:
                        yield token
        except Exception as exc:
            logger.exception("MT streaming failed for %s->%s", src, tgt)
            raise MTTranslationError(str(exc)) from exc

    def close(self):
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
                self._model_path = None


mt_engine = MTEngine()

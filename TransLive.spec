# -*- mode: python ; coding: utf-8 -*-

import json
import os
from pathlib import Path
from datetime import date

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

import llama_cpp

from app.version import APP_CREDIT, APP_NAME, APP_VERSION, APP_VERSION_NUMERIC

entitlements = Path("scripts/entitlements.plist")
codesign_identity = os.environ.get("TRANS_CODESIGN_IDENTITY") or None
entitlements_file = str(entitlements) if entitlements.exists() else None
build_datas = [
    ("web", "web"),
    ("MACOS_MIGRATION.md", "."),
    ("LICENSE", "."),
    ("NOTICE", "."),
    ("MODEL_LICENSES.md", "."),
    (str(Path(llama_cpp.__file__).resolve().parent / "lib"), "llama_cpp/lib"),
]
build_datas += collect_data_files("mlx")
build_binaries = collect_dynamic_libs("mlx")


def _find_silero_onnx():
    """
    Silero VAD 权重只有 2.3MB，打进包里可以省掉首次运行的网络依赖。
    直接复用 app.core.vad 的解析逻辑，保证打进包的文件和运行时会选的
    是同一个版本（它认的是 TRANS_VAD_SILERO_REPO 里 pin 住的 tag）。
    本地找不到时会按该 tag 下载一次。
    """
    try:
        from app.core.vad import _resolve_onnx_path

        candidate = _resolve_onnx_path()
        if candidate.is_file() and candidate.stat().st_size > 512 * 1024:
            return candidate
    except Exception as exc:
        print(f"WARNING: could not resolve silero_vad.onnx for bundling: {exc}")
    return None


_silero_onnx = _find_silero_onnx()
if _silero_onnx is not None:
    build_datas.append((str(_silero_onnx), "."))
else:
    # 不是致命错误：app.core.vad 会在首次使用时按 pin 住的 tag 自行下载。
    print("WARNING: silero_vad.onnx not found locally; the app will download it on first run")

app_expiry_date = os.environ.get("TRANS_APP_EXPIRY_DATE", "").strip()
if app_expiry_date:
    date.fromisoformat(app_expiry_date)
    build_config = Path("build/translive_build_config.json")
    build_config.parent.mkdir(parents=True, exist_ok=True)
    build_config.write_text(
        json.dumps({"app_expiry_date": app_expiry_date}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    build_datas.append((str(build_config), "."))

hiddenimports = []
for package in (
    "app",
    "uvicorn",
    "fastapi",
    "websockets",
    "mlx",
    # mlx-audio supports many unrelated TTS/STT families. The desktop app only
    # needs Qwen3-ASR, so collect that dynamically imported model family alone.
    "mlx_audio.stt.models.qwen3_asr",
    # transformers 用惰性模块加载（_import_structure），PyInstaller 静态分析
    # 看不到这些子模块。Qwen3-ASR 的 AutoTokenizer/AutoConfig 会按 model_type
    # 动态 import transformers.models.qwen3_asr —— 少了它打包后会报
    # `No module named 'transformers.models.qwen3_asr'`，然后静默退回 Whisper。
    # 只收需要的这几个（约 30 个模块）；collect_submodules("transformers")
    # 会拉进 2556 个模块，把包撑大几百 MB。
    "transformers.models.qwen3_asr",
    "transformers.models.auto",
):
    hiddenimports += collect_submodules(package)

# torch 仍然排除（244MB，macOS 上没有代码路径用得到）。
# 但 transformers 不能再排：mlx-audio 的 Qwen3-ASR 需要它的 AutoTokenizer 和
# WhisperFeatureExtractor（qwen3_asr.py 里在加载时 import）。
# transformers 在没有 torch 时会走纯 numpy/tokenizers 路径，is_torch_available()
# 返回 False 即可，不会把 torch 拉回来。
hiddenimports += [
    "transformers",
    "transformers.models.qwen2.tokenization_qwen2",
    "transformers.models.qwen2.tokenization_qwen2_fast",
    # Qwen3-ASR calls this feature extractor. It is preprocessing code, not a
    # bundled Whisper model or Whisper inference fallback.
    "transformers.models.whisper.feature_extraction_whisper",
    "mlx_audio.stt.generate",
    "mlx_audio.stt.utils",
    "mlx_lm.generate",
    "mlx_lm.models.base",
    "mlx_lm.models.cache",
    "mlx_lm.sample_utils",
    "AVFoundation",
    "CoreAudio",
    "CoreMedia",
    "Quartz",
    "ScreenCaptureKit",
    "tokenizers",
    "tiktoken",
    "onnxruntime",
    "llama_cpp",
    "huggingface_hub",
    "huggingface_hub.file_download",
    "huggingface_hub.hf_api",
    "huggingface_hub.utils",
]

a = Analysis(
    ["desktop_launcher.py"],
    pathex=[],
    binaries=build_binaries,
    datas=build_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["scripts/pyi_rth_shutdown.py"],
    excludes=[
        "openvino",
        "optimum",
        "tensorflow",
        "matplotlib",
        "notebook",
        "IPython",
        "triton",
        # ── macOS 运行时用不到的传递依赖 ────────────────────────────
        # macOS 上 ASR 走 Qwen3/MLX、MT 走 llama.cpp Metal、VAD 走
        # onnxruntime，torch 没有任何代码路径会用到（见 app/core/vad.py、
        # app/asr/engine.py 的 _is_apple_silicon_mlx）。torch 单独就占 244MB。
        "torch",
        "torchaudio",
        "torchgen",
        "accelerate",
        # 下面这些是被 transformers / accelerate 拖进来的，本应用一个都没引用。
        # 每一项都用「屏蔽该模块后跑通 MLX ASR + Silero VAD」验证过。
        "cv2",          # 108MB
        "pyarrow",      # 108MB
        "pandas",       #  18MB
        "spacy",        #  14MB
        "sklearn",      #  13MB
        "scikit_learn",
        "PIL",          #  11MB
        "datasets",
        "sympy",
        "networkx",
        # These may exist in a developer environment and are discovered by
        # optional Transformers hooks, but none is used by the frozen app.
        "librosa",
        "soundfile",
        "pydub",
        "pytest",
        "py",
        "aiohttp",
        "orjson",
        "babel",
        "dateutil",
        "pytz",
        "jsonschema",
        "rdflib",
        "pycountry",
        # Qwen-only 分发版不含任何 Whisper 推理后端及其传递依赖。
        "mlx_whisper",
        "faster_whisper",
        "ctranslate2",
        "transformers.models.whisper.configuration_whisper",
        "transformers.models.whisper.english_normalizer",
        "transformers.models.whisper.generation_whisper",
        "transformers.models.whisper.modeling_whisper",
        "transformers.models.whisper.processing_whisper",
        "transformers.models.whisper.tokenization_whisper",
        "numba",
        "llvmlite",
        "av",
        "sounddevice",
    ],
    noarchive=False,
    optimize=0,
)

# Transformers' PyInstaller hook also copies dynamically discoverable model
# source files as data. Qwen3-ASR only needs WhisperFeatureExtractor for log-mel
# preprocessing, so keep that tiny compatibility component and its package file.
_whisper_preprocessor_data = {
    "transformers/models/whisper/__init__.py",
    "transformers/models/whisper/feature_extraction_whisper.py",
}
a.datas = [
    entry
    for entry in a.datas
    if not entry[0].startswith("transformers/models/whisper/")
    or entry[0] in _whisper_preprocessor_data
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TransLive",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=codesign_identity,
    entitlements_file=entitlements_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TransLive",
)

app = BUNDLE(
    coll,
    name="TransLive.app",
    icon=None,
    bundle_identifier="com.translive.desktop",
    info_plist={
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleShortVersionString": APP_VERSION_NUMERIC,
        "CFBundleVersion": APP_VERSION_NUMERIC,
        "CFBundleGetInfoString": f"{APP_NAME} {APP_VERSION} · {APP_CREDIT}",
        "NSHumanReadableCopyright": APP_CREDIT,
        "NSMicrophoneUsageDescription": "TransLive needs microphone access to transcribe speech for live translation.",
        "NSScreenCaptureUsageDescription": "TransLive can capture system audio on macOS for live translation.",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": os.environ.get("MACOSX_DEPLOYMENT_TARGET", "14.0"),
    },
)

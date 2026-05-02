# -*- mode: python ; coding: utf-8 -*-

import json
import os
from pathlib import Path
from datetime import date

from PyInstaller.utils.hooks import collect_submodules

import llama_cpp

entitlements = Path("scripts/entitlements.plist")
codesign_identity = os.environ.get("TRANS_CODESIGN_IDENTITY") or None
entitlements_file = str(entitlements) if entitlements.exists() else None
build_datas = [
    ("web", "web"),
    ("MACOS_MIGRATION.md", "."),
    (str(Path(llama_cpp.__file__).resolve().parent / "lib"), "llama_cpp/lib"),
]

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
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "llama_cpp",
    "huggingface_hub",
):
    hiddenimports += collect_submodules(package)

hiddenimports += [
    "AVFoundation",
    "CoreAudio",
    "CoreMedia",
    "Quartz",
    "ScreenCaptureKit",
    "sentencepiece",
    "tokenizers",
    "torch",
    "torch.testing",
    "torchaudio",
    "transformers",
    "huggingface_hub.file_download",
    "huggingface_hub.hf_api",
    "huggingface_hub.utils",
    "transformers.models.auto",
    "transformers.models.whisper",
    "transformers.models.whisper.configuration_whisper",
    "transformers.models.whisper.feature_extraction_whisper",
    "transformers.models.whisper.modeling_whisper",
    "transformers.models.whisper.processing_whisper",
    "transformers.models.whisper.tokenization_whisper",
]

a = Analysis(
    ["desktop_launcher.py"],
    pathex=[],
    binaries=[],
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
        "torch.utils.tensorboard",
        "triton",
    ],
    noarchive=False,
    optimize=0,
)

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
    target_arch=None,
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
        "CFBundleName": "TransLive",
        "CFBundleDisplayName": "TransLive",
        "CFBundleShortVersionString": "0.88",
        "CFBundleVersion": "0.88",
        "CFBundleGetInfoString": "TransLive 0.88 (test) · 薛定谔的帮你偶",
        "NSHumanReadableCopyright": "薛定谔的帮你偶",
        "NSMicrophoneUsageDescription": "TransLive needs microphone access to transcribe speech for live translation.",
        "NSScreenCaptureUsageDescription": "TransLive can capture system audio on macOS for live translation.",
        "NSHighResolutionCapable": True,
    },
)

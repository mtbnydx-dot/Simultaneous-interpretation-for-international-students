#!/usr/bin/env python
"""
TransLive 一键启动器

做四件事：
  1. 检测 Python 与运行环境
  2. 检查必需依赖，缺失的自动 pip 安装（带进度）
  3. 启动 FastAPI / uvicorn 服务
  4. 等服务就绪后自动打开浏览器

适合分发给最终用户：双击 start.bat（Windows）/ ./start.sh（macOS/Linux）即可。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── 必须最早设置，防止 Rich/NNCF/HF 在 Windows GBK 控制台写入崩溃 ──
os.environ.setdefault("NNCF_PROGRESS_BAR", "false")
os.environ.setdefault("RICH_FORCE_TERMINAL", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

import argparse
import importlib
import importlib.util
import shutil
import subprocess
import threading
import time
import webbrowser

# (import_name, pip_spec, label)
REQUIRED_PACKAGES: list[tuple[str, str, str]] = [
    ("fastapi",           "fastapi>=0.115.0",          "FastAPI"),
    ("uvicorn",           "uvicorn[standard]>=0.34.0", "uvicorn"),
    ("websockets",        "websockets>=14.0",          "websockets"),
    ("huggingface_hub",   "huggingface-hub>=0.26.0",   "huggingface-hub"),
    ("numpy",             "numpy>=1.26.0",             "numpy"),
    ("scipy",             "scipy>=1.11.0",             "scipy"),
    ("sounddevice",       "sounddevice>=0.5.0",        "sounddevice"),
    ("pydantic_settings", "pydantic-settings>=2.7.0",  "pydantic-settings"),
    ("torch",             "torch>=2.2.0",              "PyTorch"),
    ("torchaudio",        "torchaudio>=2.2.0",         "torchaudio"),
    ("transformers",      "transformers>=4.40.0",      "transformers"),
    ("sentencepiece",     "sentencepiece>=0.2.0",      "sentencepiece"),
    ("accelerate",        "accelerate>=0.30.0",        "accelerate"),
    ("faster_whisper",    "faster-whisper>=1.1.0",     "faster-whisper"),
    ("llama_cpp",         "llama-cpp-python>=0.3.0",   "llama-cpp-python"),
]

OPTIONAL_PACKAGES: list[tuple[str, str]] = [
    ("openvino",  "OpenVINO (Intel iGPU/NPU ASR 加速)"),
]

BANNER = (
    "+------------------------------------------------------------+\n"
    "|                  TransLive · AI 同声传译                   |\n"
    "|                 v0.88 (test) · 薛定谔的帮你偶              |\n"
    "|                    一键启动 / 自动配置                     |\n"
    "+------------------------------------------------------------+"
)


def _print(text: str = "") -> None:
    print(text, flush=True)


def _hr(char: str = "-", width: int = 60) -> None:
    _print(char * width)


# ────────────────────────────────────────────────────────────────────
#  Stage 1: Python / 环境检测
# ────────────────────────────────────────────────────────────────────

def stage_check_runtime() -> None:
    _print(">> [1/4] 运行环境")
    py = sys.version_info
    _print(f"  Python   : {py.major}.{py.minor}.{py.micro}   ({sys.executable})")
    _print(f"  平台     : {sys.platform}")

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    _print(f"  虚拟环境 : {'是' if in_venv else '否（建议在 venv 中运行）'}")

    if (py.major, py.minor) < (3, 10):
        _print("  ! 警告：Python 版本过低，推荐 3.10+。继续运行可能出错。")
    _print("")


# ────────────────────────────────────────────────────────────────────
#  Stage 2: 依赖检测 + 自动安装
# ────────────────────────────────────────────────────────────────────

def _is_installed(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


def _pip_install(spec: str) -> tuple[bool, str]:
    """同步执行 pip install。失败时返回 (False, stderr 摘要)"""
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--disable-pip-version-check", "--no-input", spec,
    ]
    try:
        # 直接把 pip 输出透传到当前控制台，让用户看见进度
        subprocess.check_call(cmd, env={**os.environ})
        return True, ""
    except subprocess.CalledProcessError as exc:
        return False, f"exit={exc.returncode}"
    except FileNotFoundError:
        return False, "pip 不可用 (请确认 Python 安装是否完整)"


def stage_check_dependencies() -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """返回 (installed, missing)"""
    _print(">> [2/4] 必需依赖检测")
    installed: list[tuple[str, str, str]] = []
    missing: list[tuple[str, str, str]] = []
    for import_name, spec, label in REQUIRED_PACKAGES:
        if _is_installed(import_name):
            _print(f"  [OK]    {label}")
            installed.append((import_name, spec, label))
        else:
            _print(f"  [缺失]  {label}    ->  {spec}")
            missing.append((import_name, spec, label))
    _print("")

    _print(">> 可选依赖（仅检测，不自动安装）")
    for import_name, label in OPTIONAL_PACKAGES:
        flag = "OK" if _is_installed(import_name) else "--"
        _print(f"  [{flag}]    {label}")
    _print("")
    return installed, missing


def stage_install_missing(missing: list[tuple[str, str, str]], yes: bool) -> bool:
    """安装缺失依赖；全部成功返回 True"""
    if not missing:
        _print("  所有必需依赖已就绪。")
        _print("")
        return True

    _print(f">> [3/4] 安装 {len(missing)} 个缺失依赖")
    if not yes and sys.stdin.isatty():
        try:
            ans = input(f"  即将自动安装上述依赖到 {sys.executable}\n  按回车继续 / 输入 n 取消: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        if ans in ("n", "no"):
            _print("  已取消。")
            return False

    failed: list[str] = []
    for i, (_imp, spec, label) in enumerate(missing, 1):
        _print(f"  [{i}/{len(missing)}] pip install {spec}")
        ok, err = _pip_install(spec)
        if ok:
            _print(f"          [完成] {label}")
        else:
            _print(f"          [失败] {label}: {err}")
            failed.append(label)
        _print("")

    importlib.invalidate_caches()

    if failed:
        _print("  以下依赖安装失败，请手动重试:")
        for name in failed:
            _print(f"    - {name}")
        _print(f"  或执行: {sys.executable} -m pip install -r requirements.txt")
        return False

    _print("  所有依赖安装完成。")
    _print("")
    return True


# ────────────────────────────────────────────────────────────────────
#  Stage 3: 模型文件检测（仅显示，不下载 —— 服务启动后按需下载）
# ────────────────────────────────────────────────────────────────────

def stage_detect_models() -> None:
    _print(">> 模型文件")
    models_dir = SCRIPT_DIR / "models"
    if not models_dir.exists():
        _print(f"  [--]   {models_dir} 不存在；服务首次启动时会自动创建并下载")
        _print("")
        return
    target = next(
        (p for p in models_dir.glob("*.gguf") if p.name.lower().startswith("hy-mt")),
        None,
    )
    if target:
        size_mb = target.stat().st_size / (1024 * 1024)
        _print(f"  [OK]   MT  : {target.name}  ({size_mb:.0f} MB)")
    else:
        _print("  [--]   MT  : HY-MT1.5 GGUF 未找到（服务启动后自动下载约 1.13 GB）")
    _print("")


# ────────────────────────────────────────────────────────────────────
#  Stage 4: 启动服务 + 自动开浏览器
# ────────────────────────────────────────────────────────────────────

def _open_browser_when_ready(url: str, health_url: str, max_wait: float = 120.0) -> None:
    """在后台线程里轮询 /api/health；服务起来后打开浏览器"""
    import urllib.request
    import urllib.error

    def _wait() -> None:
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(health_url, timeout=1) as resp:
                    if resp.status == 200:
                        try:
                            webbrowser.open(url)
                        except Exception:
                            pass
                        return
            except (urllib.error.URLError, ConnectionRefusedError, OSError, TimeoutError):
                pass
            time.sleep(0.4)
        # 兜底：超时也尝试开一次浏览器
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_wait, daemon=True).start()


def stage_launch(no_browser: bool = False) -> None:
    # 此时所有依赖已就位，安全 import
    from app.core.config import settings  # noqa: WPS433
    import uvicorn  # noqa: WPS433

    host = settings.host
    port = settings.port
    browse_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    url = f"http://{browse_host}:{port}/"
    health = f"http://{browse_host}:{port}/api/health"

    _print(">> [4/4] 启动服务")
    _print(f"  绑定地址 : http://{host}:{port}")
    _print(f"  访问地址 : {url}")
    if no_browser:
        _print("  浏览器   : 已禁用 (--no-browser)")
    else:
        _print("  浏览器   : 服务就绪后自动打开")
    _print("")
    _print("  按 Ctrl+C 退出")
    _hr("=", 60)
    _print("")

    if not no_browser:
        _open_browser_when_ready(url, health)

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
        ws_max_size=settings.websocket_max_message_bytes,
    )


# ────────────────────────────────────────────────────────────────────
#  入口
# ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="TransLive 一键启动 —— 自动检测环境 / 自动安装依赖 / 自动开浏览器",
    )
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--no-install", action="store_true", help="只检测，不自动安装缺失依赖")
    parser.add_argument("--check-only", action="store_true", help="只做环境检测，不启动服务")
    parser.add_argument("-y", "--yes", action="store_true", help="非交互模式（自动同意安装）")
    args = parser.parse_args()

    _print(BANNER)
    _print("")

    stage_check_runtime()
    _installed, missing = stage_check_dependencies()

    if args.no_install:
        if missing:
            _print(f"  ! 检测到 {len(missing)} 个缺失依赖，但 --no-install 已启用。")
            sys.exit(2)
    else:
        if not stage_install_missing(missing, yes=args.yes):
            sys.exit(1)

    stage_detect_models()

    if args.check_only:
        _print(">> 检测完成 (--check-only)")
        return

    stage_launch(no_browser=args.no_browser)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _print("")
        _print("已退出。")

# TransLive macOS 迁移说明

Windows 启动脚本、旧虚拟环境和 Windows 版 llama.cpp 二进制已统一归档到
`windows_legacy/`。macOS 主流程只使用 `.venv-macos`、`llama-cpp-python`
和 `dist/TransLive.app`，这些 Windows 文件不会参与运行或打包。

## 快速启动

```bash
cd /Users/bbzlk/Desktop/code/TRANS
chmod +x start.sh
./start.sh -y
```

`start.sh` 会优先使用本机虚拟环境：

- 如果存在 `.venv/bin/python`，直接使用它。
- 否则使用 `.venv-macos`。
- 如果 `.venv-macos` 不存在，会自动创建，然后交给 `run.py` 安装依赖并启动服务。

服务启动后访问：

```text
http://127.0.0.1:8766/
```

## Apple Silicon / Metal

ASR 默认会在 Apple Silicon 上走 `faster-whisper / CTranslate2 + CPU int8`，
模型为 `large-v3-turbo`，优先保证实时性。若你更看重准确率且能接受更高延迟，
可在 `.env` 中切到 `TRANS_ASR_BACKEND=transformers-whisper` 和
`TRANS_ASR_MODEL_ID=openai/whisper-large-v3-turbo`。MT 默认走
`llama-cpp-python`，只有在当前安装的 `llama-cpp-python` 确认支持 GPU offload 时
才启用 Metal；否则自动降级 CPU，避免启动失败。

如需强制重装带 Metal 的 llama.cpp Python 包：

```bash
.venv-macos/bin/python -m pip uninstall -y llama-cpp-python
CMAKE_ARGS="-DGGML_METAL=on" FORCE_CMAKE=1 \
  .venv-macos/bin/python -m pip install --no-cache-dir llama-cpp-python
```

## 系统声音输入

桌面版会优先使用 macOS 原生系统音频采集。首次使用需要在系统设置里授予屏幕录制
权限；如果当前系统或依赖不支持该能力，仍可退回麦克风输入或 BlackHole/Loopback。

## 常用检查

```bash
./start.sh --check-only --no-install
.venv-macos/bin/python test_new_features.py
```

如果需要重新下载翻译模型：

```bash
.venv-macos/bin/python scripts/download_models.py --mt
```

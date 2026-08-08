# TransLive macOS 迁移说明

Windows 启动脚本、旧虚拟环境和 Windows 版 llama.cpp 二进制已统一归档到
`windows_legacy/`。macOS 主流程只使用 `.venv-macos`、`llama-cpp-python`
和 `dist/TransLive.app`，这些 Windows 文件不会参与运行或打包。

当前分发包支持 Apple Silicon 上的 macOS 14.0 及以上版本，包括 macOS 14 和
macOS 15；MLX 不支持 Intel Mac。ASR 固定为 Qwen3-ASR 8-bit，不打包 Whisper
模型或 Whisper 推理后端。

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

Apple Silicon 默认使用 `mlx-community/Qwen3-ASR-1.7B-8bit`，经
`mlx-audio` 在 MLX/Metal 上运行。该模型是 8bit 量化版本；`mlx-audio`
不可用或加载失败时，程序才回退到 MLX Whisper。ASR 与 MT 会顺序加载，避免两个
大模型同时从磁盘读入造成峰值内存和 I/O 抖动。

MT 默认使用 Hy-MT2 GGUF，并通过带 Metal 的 `llama-cpp-python` 将模型层卸载到
GPU。桌面状态栏和 `/api/health` 会显示实际后端与加速状态；若 Metal 不可用，会
明确显示 CPU 降级，而不是静默伪装成 GPU 模式。

如需强制重装带 Metal 的 llama.cpp Python 包：

```bash
.venv-macos/bin/python -m pip uninstall -y llama-cpp-python
CMAKE_ARGS="-DGGML_METAL=on" FORCE_CMAKE=1 \
  .venv-macos/bin/python -m pip install --no-cache-dir llama-cpp-python
```

正式构建使用带哈希锁定的 macOS 依赖和 Metal 编译步骤：

```bash
./scripts/build_macos_app.sh
```

不要在 `.env` 的通用 `TRANS_ASR_MODEL_ID` 中填入另一种后端的模型。指定 Qwen3
权重时使用 `TRANS_ASR_QWEN3_MODEL_ID`；当前默认值已经适合大多数 Apple Silicon
设备。

## 系统声音输入

桌面版会优先使用 macOS 原生系统音频采集。首次使用需要在系统设置里授予屏幕录制
权限；如果当前系统或依赖不支持该能力，仍可退回麦克风输入或 BlackHole/Loopback。

## 常用检查

```bash
./start.sh --check-only --no-install
.venv-macos/bin/python -m pytest
```

如果需要重新下载翻译模型：

```bash
.venv-macos/bin/python scripts/download_models.py --mt
```

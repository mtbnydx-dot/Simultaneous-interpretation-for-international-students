# Whisper / Distil-Whisper 历史后端说明

> 状态：历史兼容文档。当前 macOS Apple Silicon 分发版使用 Qwen3-ASR/MLX，不打包 Whisper 权重，也不会在 Qwen3 加载失败时静默回退到 Whisper。

本文件保留是因为源码仍包含面向 Windows、Linux、NVIDIA CUDA、Intel OpenVINO 和 CPU 的可选 ASR 后端。它不是当前 macOS App 的安装指南；macOS 用户请从 [README.md](README.md) 和 [MACOS_MIGRATION.md](MACOS_MIGRATION.md) 开始。

## 可选源码后端

| `TRANS_ASR_BACKEND` | 典型环境 | 说明 |
| --- | --- | --- |
| `ct2` | Windows/Linux、NVIDIA 或 CPU | 使用 faster-whisper/CTranslate2 |
| `openvino` | Intel CPU/GPU | 使用 OpenVINO 导出的 Whisper 模型 |
| `transformers-distil` | 支持 Transformers 的开发环境 | Distil-Whisper 兼容路径 |
| `transformers-whisper` | 支持 Transformers 的开发环境 | 标准 Whisper 兼容路径 |
| `mlx` | Apple Silicon 源码实验 | 旧 MLX Whisper 路径，不进入正式 Qwen-only App |

这些后端属于源码兼容能力，不等同于本项目对所有硬件组合提供正式支持或预编译安装包。

## 配置示例

CTranslate2：

```dotenv
TRANS_ASR_BACKEND=ct2
TRANS_ASR_MODEL_SIZE=large-v3-turbo
TRANS_ASR_DEVICE=auto
TRANS_ASR_COMPUTE_TYPE=default
```

Transformers Whisper：

```dotenv
TRANS_ASR_BACKEND=transformers-whisper
TRANS_ASR_MODEL_ID=openai/whisper-large-v3-turbo
TRANS_ASR_DEVICE=auto
```

Distil-Whisper：

```dotenv
TRANS_ASR_BACKEND=transformers-distil
TRANS_ASR_MODEL_ID=distil-whisper/distil-large-v3
TRANS_ASR_DEVICE=auto
```

具体可用性取决于平台、驱动、Python 包和模型本身。请从 `.env.example` 复制完整配置，不要把不同后端的模型 ID 混写到 Qwen3 专用变量中。

## 安装与验证

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python run.py
```

启动后检查 `/api/health` 或界面顶部状态，确认实际加载的后端与设备。不能只根据配置推断 GPU 已生效；如果目标后端加载失败，应查看日志中的原始异常。

## 性能说明

仓库不再给出跨硬件的固定“快多少”或 WER 表。ASR 延迟通常同时受以下因素影响：

- VAD 静音等待和最大分段时长
- 音频片段长度
- 模型大小与量化方式
- GPU/CPU 后端是否真正启用
- 首次加载、模型缓存和系统内存压力

比较后端时应使用同一组真实中文、英文和混合语言音频，分别记录分段等待、纯 ASR 推理时间、错误率和峰值内存。

## macOS 分发边界

当前 macOS App 构建固定：

```text
TRANS_ASR_BACKEND=qwen3
TRANS_ASR_FALLBACKS_ENABLED=false
```

因此不要按本文件给正式 macOS 包添加 Whisper 权重。需要重新启用某个历史后端时，应同时更新依赖锁、PyInstaller hidden imports、许可证文档、模型下载流程和完整冻结包测试。

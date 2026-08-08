# 模型与第三方许可证说明

更新日期：2026-08-08

本文用于说明 TransLive 代码中默认或可选的模型来源，方便源码发布和二进制分发时核对归属。它不是法律意见；模型卡和许可证可能更新，正式发布前应再次查看上游原文。

## 仓库不包含模型

GitHub 仓库只包含源码、脚本和文档。以下内容不应提交：

- `models/` 与下载后的 GGUF、Safetensors、ONNX、PyTorch 权重
- Hugging Face 本地缓存
- `.env`、访问令牌、签名证书和公证凭据
- `.venv*`、`dist/`、`build/`、日志和用户翻译历史

模型由用户在首次启动时单独下载。即使某个模型允许再分发，本项目也默认不把权重放进源码仓库或 App 压缩包。

## 当前桌面版使用的模型

| 用途 | 模型 | 上游标示许可证 | 本项目处理方式 |
| --- | --- | --- | --- |
| ASR | `mlx-community/Qwen3-ASR-1.7B-8bit` | Apache-2.0 | Apple Silicon 默认模型；固定 revision，按需下载，不随 App 分发 |
| MT | `tencent/Hy-MT2-1.8B-GGUF` | Apache-2.0 | 默认使用 `Hy-MT2-1.8B-Q4_K_M.gguf`；固定 SHA-256，按需下载，不随 App 分发 |
| VAD | `snakers4/silero-vad` | MIT | 使用固定 tag 的 ONNX 权重；可能由程序按需下载 |

Qwen3-ASR 的音频预处理会使用 Transformers 中的 `WhisperFeatureExtractor`。这里的 “Whisper” 是特征提取器类名，不代表 App 打包或运行了 Whisper 模型权重。

## 兼容或实验模型

| 用途 | 模型 | 上游标示许可证 | 状态 |
| --- | --- | --- | --- |
| 旧版 MT | `tencent/HY-MT1.5-1.8B-GGUF` | Tencent HY Community License Agreement | 源码仍兼容，但不再默认使用；启用前必须单独审阅许可证 |
| 旧版 Apple ASR | `mlx-community/whisper-large-v3-turbo` | MIT | 仅源码兼容路径，不进入当前 Qwen-only macOS 分发包 |
| CTranslate2 ASR | `dropbox-dash/faster-whisper-large-v3-turbo` | MIT | Windows/Linux 源码模式可选 |
| Transformers ASR | `openai/whisper-large-v3-turbo` | MIT | 源码模式可选，不进入当前 macOS 分发包 |
| 云端候选 ASR | `nvidia/parakeet-tdt-0.6b-v3` | CC-BY-4.0 | 仅路线图候选，桌面版不下载或打包 |
| 云端候选 ASR | `nvidia/canary-1b-v2` | CC-BY-4.0 | 仅路线图候选，桌面版不下载或打包 |
| 高质量 MT 候选 | `tencent/Hunyuan-MT-7B` | 以模型卡为准 | 仅路线图候选，启用前重新核对许可证 |

## HY-MT1.5 特别说明

HY-MT1.5 使用 Tencent HY Community License Agreement，不应被描述为 Apache-2.0 或 MIT。其许可证包含适用地域、可接受使用、分发和后续用户通知等条款。

如果通过 `TRANS_MT_MODEL_ID` 切回 HY-MT1.5：

- 在下载、使用或分发前阅读完整许可证原文。
- 在产品和文档中明确模型名称与来源。
- 不要暗示 Tencent 提供、认可或赞助 TransLive。
- 对公开服务、商业收费、模型再分发和跨地区使用进行单独法律审查。

默认 Hy-MT2 模型页当前标示 Apache-2.0，因此项目优先使用 Hy-MT2，避免把 HY-MT1.5 的特殊条款带入默认发行流程。

## 主要运行库

| 组件 | 上游标示许可证 | 用途 |
| --- | --- | --- |
| `mlx` | MIT | Apple Silicon 数组与 Metal 运行时 |
| `mlx-audio` | MIT | Qwen3-ASR 的 MLX 音频推理 |
| `llama-cpp-python` | MIT | 本地 GGUF 翻译推理 |
| `onnxruntime` | MIT | Silero VAD ONNX 推理 |
| `faster-whisper` | MIT | 非 Apple Silicon 的可选源码后端 |
| `CTranslate2` | MIT | 非 Apple Silicon 的可选源码后端 |

完整 Python 依赖见 `requirements.txt`、`requirements-app.txt` 和 `requirements-macos-arm64.lock`。发布二进制时还需要检查实际打入 App 的 wheel、framework 和动态库，不能只检查顶层 Python 包。

## 发布检查清单

### 只发布源码

- 保留根目录 `LICENSE`、`NOTICE` 和本文件。
- 运行 `./scripts/check_secrets.sh`。
- 确认 Git 中没有模型权重、缓存、真实 `.env` 或证书。
- 保持 README 中“模型由用户另行下载”的说明。

### 发布 App

- 列出当前默认下载的模型、来源和许可证链接。
- 在首次下载提示中让用户知道模型来自第三方仓库。
- 保留打包依赖要求的许可证文本和归属信息。
- 对 App 中实际包含的第三方二进制重新生成依赖清单。
- 若改用或捆绑其他模型，先更新本文件再发布。

### 商业收费或公共服务

- 区分 TransLive 源码许可证、模型许可证和依赖许可证。
- 不要因为权重未打包，就假定模型许可证与服务使用无关。
- 对账户、支付、试用、远程推理和用户音频处理另行准备服务条款与隐私政策。
- 正式上线前由熟悉目标司法辖区的专业人士复核。

## 官方来源

- [Tencent Hy-MT2 1.8B GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF)
- [Tencent Hy-MT2 项目](https://github.com/Tencent-Hunyuan/Hy-MT2)
- [Tencent HY-MT1.5 1.8B GGUF](https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF)
- [Tencent HY-MT1.5 License.txt](https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF/blob/main/License.txt)
- [Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [MLX Qwen3-ASR 1.7B 8-bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit)
- [OpenAI Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
- [CTranslate2 Whisper large-v3-turbo conversion](https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [MLX](https://github.com/ml-explore/mlx)
- [mlx-audio](https://github.com/Blaizzy/mlx-audio)

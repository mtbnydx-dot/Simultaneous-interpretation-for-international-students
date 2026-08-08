# TransLive

TransLive 是一个本地运行的实时同声翻译工具：浏览器/桌面壳采集音频，FastAPI 后端做 ASR、VAD、翻译和 WebSocket 推流，前端显示双语结果与悬浮字幕。

版本与署名统一定义在 [`app/version.py`](app/version.py)，构建脚本、桌面壳和网页
都会从这里读取。

## 功能

- 实时音频转写与翻译
- 双语/单译文悬浮字幕窗口
- macOS 桌面壳，首次启动可提示用户自行下载模型
- macOS 原生系统音频采集入口，基于 ScreenCaptureKit
- 支持术语表、导出、性能信息和模型健康检查
- 硬件自动选型：macOS 分发版固定使用 Qwen3/MLX；其他源码环境可选 CTranslate2/CUDA
- 预留云端 ASR/MT 接口，便于以后接 paid inference / 私有服务

## 仓库内容

这个仓库只适合放源码、脚本和文档，不应提交本机模型、虚拟环境、打包产物、日志或真实密钥。项目源码使用 Apache-2.0，见 [LICENSE](LICENSE)；第三方模型和运行库仍分别遵循各自许可证。

默认不会包含：

- `models/`
- `.env`
- `.venv*`、`venv/`
- `dist/`、`build/`
- `logs/`
- `windows_legacy/tools/`

请用 `.env.example` 作为配置模板，不要把真实 `TRANS_HF_TOKEN` 提交到 GitHub。

## 本地运行

```bash
python3 -m venv .venv-macos
source .venv-macos/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run.py
```

启动器会输出并自动打开带本次随机本地令牌的地址，例如：

```text
http://127.0.0.1:8766/#token=...
```

## macOS App 打包

当前 arm64 打包基线为 Apple Silicon + macOS 14.0 或更高版本（包括 macOS 14、15）。构建脚本使用带哈希的 `requirements-macos-arm64.lock`，不会在构建时升级 pip 或漂移依赖：

```bash
./scripts/build_macos_app.sh
```

默认输出：

- `dist/TransLive.app`
- `dist/TransLive-macOS-arm64.zip`

模型不会被打进 App。桌面版会把模型放在：

```text
~/Library/Application Support/TransLive/models/
```

## 模型与许可证

本项目当前默认翻译模型是 `tencent/Hy-MT2-1.8B-GGUF`（Apache-2.0，33 种语言），macOS Apple Silicon 分发版的 ASR 只使用 `mlx-community/Qwen3-ASR-1.7B-8bit`（Apache-2.0）并通过 `mlx-audio` 运行。VAD 使用 Silero VAD 的 ONNX 权重经 onnxruntime 推理（不依赖 PyTorch）。默认模型按 revision/SHA-256 固定，且不随源码或 App 分发。Qwen 所需的 Transformers `WhisperFeatureExtractor` 仅负责音频特征预处理，不包含 Whisper 权重或推理后端。

旧的 `tencent/HY-MT1.5-1.8B-GGUF` 仍然兼容：把 `TRANS_MT_MODEL_ID` 指向本地的 1.5 文件即可，prompt 模板和下载仓库会按文件名自动切换。

硬件兼容性审计、模型路线和云端接口契约见 [docs/PROJECT_AUDIT_AND_ROADMAP.md](docs/PROJECT_AUDIT_AND_ROADMAP.md)。

上传 GitHub 前必须保留模型来源和许可证说明。详细清单见 [MODEL_LICENSES.md](MODEL_LICENSES.md)。

注意：如果你切回 HY-MT1.5，它使用的是 Tencent HY Community License Agreement，不是 MIT/Apache 这类宽松许可证，包含地域、用途、分发声明和服务披露要求。公开发布、商业收费或面向第三方分发前，请先确认你的使用方式满足该许可证。Hy-MT2 是 Apache-2.0，没有这些限制。

## 项目代码许可证

TransLive 自有源码使用 Apache License 2.0，第三方模型与依赖不因此改变许可证；对应归属见 `NOTICE` 和 `MODEL_LICENSES.md`。

## 安全检查

上传前建议运行：

```bash
./scripts/check_secrets.sh
```

开发回归测试：

```bash
pip install -r requirements-dev.txt
pytest
```

如果要生成干净的 GitHub 源码目录：

```bash
./scripts/prepare_github_upload.sh
```

脚本只复制源码和文档，不复制模型、venv、dist、build、日志、旧上传包和真实 `.env`。生成后请上传输出目录，而不是直接上传整个工作目录。

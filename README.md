# TransLive

TransLive 是一款面向课堂、会议和视频内容的本地实时同声翻译工具。它在本机完成语音活动检测、语音识别和机器翻译，并通过桌面窗口、网页界面与悬浮字幕显示结果。

- 当前版本：`v.1.1`
- 署名：薛定谔的帮你偶
- 源码许可证：[Apache License 2.0](LICENSE)

> 本仓库只发布源码、构建脚本和文档，不包含模型权重、用户记录、密钥或已打包的 App。

## 当前能力

- 麦克风实时识别与翻译
- macOS 桌面版原生系统音频采集（ScreenCaptureKit）
- 双语或仅译文悬浮字幕，可调字体、宽度、高度和透明度
- 最新内容置顶，并支持历史记录、TXT/SRT 导出和术语表
- 将本次全部原文按时间顺序合并后进行一次全文翻译
- Qwen3-ASR token 流式预览实验功能；默认关闭，不影响最终字幕
- 本地 API 令牌、会话数限制、WebSocket 消息大小限制和模型健康检查
- 首次启动按需下载模型，分发包不携带模型权重

## 默认技术栈

```text
麦克风 / 系统音频
        |
        v
Silero VAD (ONNX) -> Qwen3-ASR (MLX/Metal) -> Hy-MT2 (GGUF/Metal)
        |
        v
FastAPI + WebSocket -> 主界面 / 悬浮字幕 / 导出 / 全文翻译
```

| 模块 | macOS 桌面版默认实现 |
| --- | --- |
| ASR | `mlx-community/Qwen3-ASR-1.7B-8bit`，通过 `mlx-audio` 使用 Apple Metal |
| MT | `tencent/Hy-MT2-1.8B-GGUF` 的 `Q4_K_M` 量化，通过 `llama-cpp-python` 使用 Metal |
| VAD | Silero VAD ONNX，通过 `onnxruntime` 推理 |
| 服务 | FastAPI + Uvicorn + WebSocket |
| 桌面壳 | PyWebView + PyObjC |

正式 macOS 分发配置固定使用 Qwen3-ASR，并关闭静默回退。模型不可用时会明确报错，不会悄悄切换到 Whisper。源码模式仍保留 CTranslate2、OpenVINO、Transformers 和云端接口，供其他硬件或开发场景使用。

## 平台支持

| 平台 | 支持状态 | 说明 |
| --- | --- | --- |
| Apple Silicon，macOS 14 或更高版本 | 主要支持 | 当前 App 构建、MLX ASR、Metal MT 和系统音频采集目标平台 |
| Intel Mac | 不支持当前 App | MLX/Qwen3 分发配置需要 Apple Silicon |
| Windows / Linux | 源码模式，尽力支持 | 可使用 CTranslate2/CUDA、OpenVINO 或 CPU；仓库不提供当前平台安装包 |

macOS 兼容性声明只适用于构建脚本实际检查通过的 `arm64` 二进制。修改依赖或重新生成锁文件后，应重新执行完整构建和最低系统版本检查。

## 源码运行

建议使用 Python 3.12；启动脚本最低要求 Python 3.10。

```bash
git clone https://github.com/mtbnydx-dot/Simultaneous-interpretation-for-international-students.git
cd Simultaneous-interpretation-for-international-students

python3 -m venv .venv-macos
source .venv-macos/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python run.py
```

启动器会绑定 `127.0.0.1`，生成本次运行的随机 API 令牌，并打开类似下面的本地地址：

```text
http://127.0.0.1:8766/#token=...
```

网页源码模式可以使用麦克风；原生系统音频采集仅在 macOS 桌面壳中提供：

```bash
python -m pip install -r requirements-app.txt
python desktop_launcher.py
```

## 模型下载

模型不在 Git 仓库或 App 内。桌面版首次启动会提示下载；源码模式也可以手动准备：

```bash
python scripts/download_models.py --all
```

源码模式默认把 MT 模型放在项目的 `models/` 目录，Hugging Face 模型使用本机缓存。打包后的桌面版使用：

```text
~/Library/Application Support/TransLive/models/
```

下载前请阅读 [MODEL_LICENSES.md](MODEL_LICENSES.md)。如果改用其他模型，发布者需要自行核对并补充相应许可证、来源和归属。

## 基本使用

1. 选择源语言和目标语言；源语言可以设为自动检测。
2. 选择麦克风或系统音频，然后开始同传。
3. 需要跨应用显示时打开字幕窗，并选择双语或仅译文。
4. 需要上下文一致的结果时使用全文翻译。该功能会把本次原文按时间顺序合并，并作为一个整体重新翻译，而不是逐条重放实时译文。

首次使用麦克风或系统音频时，请在“系统设置 -> 隐私与安全性”中授予相应权限。系统音频入口不可用时，先确认正在运行桌面 App，而不是直接打开 `web/index.html`。

## 配置

复制 `.env.example` 后按需修改。常用配置包括：

| 变量 | 用途 |
| --- | --- |
| `TRANS_ASR_QWEN3_MODEL_ID` | Qwen3-ASR 模型或量化档位 |
| `TRANS_MT_MODEL_ID` | 本地 Hy-MT GGUF 文件路径 |
| `TRANS_ASR_STREAMING_PREVIEW_ENABLED` | 是否默认启用实验性流式预览 |
| `TRANS_SESSION_LATENCY_MODE` | `low`、`balanced` 或 `stable` 分段策略 |
| `TRANS_MAX_STREAM_SESSIONS` | 同时运行的音频会话上限 |
| `TRANS_TRANSCRIPT_HISTORY_ENABLED` | 是否保存本地翻译历史 |

不要提交真实 `.env`、`TRANS_HF_TOKEN`、云端 API Key 或公证凭据。

## 测试

```bash
python -m pip install -r requirements-dev.txt
pytest
```

模型、麦克风和 ScreenCaptureKit 依赖真实硬件与系统权限，发布前还应在目标 Mac 上手动验证：

- Qwen3-ASR 和 Hy-MT2 均按预期后端加载
- 中文、英文和自动语言选择结果正确
- 麦克风与系统音频均可启动、停止和再次启动
- 悬浮字幕、全文翻译、导出和历史记录正常
- 关闭窗口和从 Dock 退出时进程能完整结束

## macOS 打包

详细步骤见 [MACOS_MIGRATION.md](MACOS_MIGRATION.md)。本机测试包：

```bash
./scripts/build_macos_app.sh
```

默认输出：

```text
dist/TransLive.app
dist/TransLive-macOS-arm64.zip
```

未设置证书时脚本使用 ad-hoc 签名，只适合本机测试。Developer ID 签名和 Apple 公证是两个独立步骤；面向其他用户分发时，两者都应完成并分别验证。

## 隐私与安全

- 默认服务仅监听 `127.0.0.1`，不向局域网公开。
- 默认 ASR、VAD 和 MT 均在本机运行；只有显式启用云端配置时才会发送数据。
- 翻译历史默认保存在本机应用数据目录，可通过配置关闭。
- 模型下载来自第三方模型仓库，项目对默认模型固定 revision 或 SHA-256 以降低上游漂移风险。
- 不要把本项目当作需要绝对准确性的医疗、法律或安全关键转录工具。

## 仓库边界

`.gitignore` 会排除以下本机内容：

- `models/` 和模型权重
- `.env`、证书、密钥和公证凭据
- `.venv*`、缓存和日志
- `dist/`、`build/` 和其他打包产物
- 本地翻译历史与临时上传目录

提交前可运行：

```bash
./scripts/check_secrets.sh
./scripts/prepare_github_upload.sh
```

## 文档

- [macOS 安装、构建与分发](MACOS_MIGRATION.md)
- [模型与第三方许可证](MODEL_LICENSES.md)
- [架构审计与路线图](docs/PROJECT_AUDIT_AND_ROADMAP.md)
- [历史 Whisper 后端说明](DISTIL_WHISPER_GUIDE.md)
- [菜单栏模式提案](MACOS_MENUBAR_PLAN.md)

## 许可证与归属

TransLive 自有源码使用 Apache License 2.0。第三方模型、Python 包和动态库继续遵循各自许可证，不会因为本仓库的源码许可证而改变。发布源码或二进制前，请同时保留 [NOTICE](NOTICE) 和 [MODEL_LICENSES.md](MODEL_LICENSES.md) 中的来源说明。

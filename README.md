# TransLive

TransLive 是一个本地运行的实时同声翻译工具：浏览器/桌面壳采集音频，FastAPI 后端做 ASR、VAD、翻译和 WebSocket 推流，前端显示双语结果与悬浮字幕。

当前版本：`0.88 (test)`  
署名：薛定谔的帮你偶

## 功能

- 实时音频转写与翻译
- 双语/单译文悬浮字幕窗口
- macOS 桌面壳，首次启动可提示用户自行下载模型
- macOS 原生系统音频采集入口，基于 ScreenCaptureKit
- 支持术语表、导出、性能信息和模型健康检查

## 仓库内容

这个仓库只适合放源码、脚本和文档，不应提交本机模型、虚拟环境、打包产物、日志或真实密钥。

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

然后打开：

```text
http://127.0.0.1:8766
```

## macOS App 打包

桌面壳依赖：

```bash
source .venv-macos/bin/activate
pip install -r requirements-app.txt
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

本项目当前默认翻译模型是 `tencent/HY-MT1.5-1.8B-GGUF`，ASR 默认走 Whisper large-v3-turbo / faster-whisper / CTranslate2 路线，VAD 默认用 Silero VAD。

上传 GitHub 前必须保留模型来源和许可证说明。详细清单见 [MODEL_LICENSES.md](MODEL_LICENSES.md)。

特别注意：HY-MT1.5 使用 Tencent HY Community License Agreement，不是 MIT/Apache 这类宽松开源许可证。它包含地域、用途、分发声明和服务披露要求。公开发布、商业收费或面向第三方分发前，请先确认你的使用方式满足该许可证。

## 项目代码许可证

当前仓库还没有为你自己的代码选择开源许可证。公开到 GitHub 前建议补一个根目录 `LICENSE` 文件；否则别人可以看代码，但默认不获得明确的复制、修改和再分发授权。

## 安全检查

上传前建议运行：

```bash
./scripts/check_secrets.sh
```

如果要重新生成干净上传包，可参考本次生成的 `github_upload/` 目录规则：只包含源码和文档，不包含模型、venv、dist、build、日志和真实 `.env`。

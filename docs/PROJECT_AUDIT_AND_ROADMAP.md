# TransLive 架构审计与路线图

更新日期：2026-08-08

## 结论

TransLive 当前适合作为 Apple Silicon 上的本地单用户同传工具：音频采集、VAD、ASR、MT、字幕与导出形成了完整链路，默认模型均可在本机运行，源码仓库也已把模型、密钥和打包产物排除在外。

当前最成熟的发布目标是 macOS 14+、Apple Silicon、Developer ID 站外分发。Windows/Linux 仍是源码兼容路径；云端推理是接口预留；菜单栏模式和 Mac App Store 上架尚未完成。

## 当前发布配置

| 项目 | 当前选择 | 发布边界 |
| --- | --- | --- |
| 平台 | macOS 14+ / Apple Silicon | 构建脚本检查 arm64 与 Mach-O 最低系统版本 |
| ASR | Qwen3-ASR 1.7B 8-bit / MLX | 正式 App 固定为 `qwen3`，关闭静默 fallback |
| MT | Hy-MT2 1.8B Q4_K_M / llama.cpp Metal | 模型按需下载，不随 App 分发 |
| VAD | Silero VAD ONNX | 固定 tag，避免执行远程仓库代码 |
| 并发 | 单机单活动流为默认 | 会话数、消息大小和队列均有限制 |
| UI | PyWebView + 原生 JS | 主窗口、悬浮字幕和全文翻译 |
| 系统音频 | ScreenCaptureKit | 仅 macOS 桌面壳，依赖系统权限 |

## 运行链路

```text
音频输入
  -> 预处理 / VAD / 分段
  -> Qwen3-ASR 最终识别
  -> Hy-MT2 流式翻译
  -> WebSocket 事件
  -> 主界面 / 字幕窗 / 历史记录 / 导出
```

实验性流式识别预览使用正式 Qwen 解码过程中的 token 回调，默认关闭。预览内容不进入历史、导出或全文翻译，最终识别结果拥有更高优先级。

全文翻译与实时逐段翻译是两条不同路径：实时路径优先低延迟；全文路径会把本次原文按时间顺序合并，再作为整体重新翻译，以获得更完整的上下文。

## 已完成的工程改进

### 稳定性

- ASR 和 MT 共享模型调用受到并发控制，避免同一模型被多个线程交错推理。
- WebSocket 单条消息、活动会话数、分段队列和 MT token 队列都有上限。
- 客户端断开后会停止继续发送和不必要的流式翻译工作。
- 桌面退出路径会停止音频采集、ASR/MT、Uvicorn 和后台线程。
- Qwen3 模型缓存会检查配置与权重完整性，损坏时给出可恢复的错误。

### 性能与资源

- VAD 使用 ONNX Runtime，不再为语音活动检测加载完整 PyTorch。
- ASR 默认使用 8-bit Qwen3，MT 默认使用 Q4_K_M GGUF。
- 翻译 partial 事件和流式队列做了节流与限长。
- 实验性预览只保留当前任务，过期草稿不会阻塞正式识别。
- 空闲和活动状态分开管理 App Nap；用户可主动释放模型内存。

### 用户体验

- 最新识别内容显示在列表顶部，并使用深色滚动条。
- 悬浮字幕支持双语/仅译文、尺寸、字体和透明度调整。
- 系统音频通过桌面壳原生采集，不要求用户安装虚拟声卡。
- 语言选择统一由后端语言表驱动，避免目标语言被固定为英语。
- 全文翻译按一次完整文档处理，不复用逐条实时译文。

### 安全与发布

- 默认只监听 `127.0.0.1`，并为本地 API 生成运行期随机令牌。
- 模型下载使用固定 revision 或文件 SHA-256。
- 真实 `.env`、模型、证书、历史记录、日志和构建产物均被 Git 忽略。
- macOS 构建区分 ad-hoc、Developer ID 签名和 Apple 公证，不把三者混为一谈。

## 兼容性路线

| 硬件 | 建议 ASR | 建议 MT | 状态 |
| --- | --- | --- | --- |
| Apple Silicon | Qwen3-ASR / MLX Metal | Hy-MT2 GGUF / Metal | 当前主要支持 |
| NVIDIA CUDA | faster-whisper / CTranslate2 CUDA | Hy-MT2 或云端 | 源码兼容，需独立验证 |
| Intel CPU/GPU | OpenVINO 或 CTranslate2 int8 | Hy-MT2 CPU 或云端 | 源码兼容，实时性取决于硬件 |
| 纯 CPU 低内存 | 较小 CT2 模型 | 本地小模型或云端 | 不建议承诺稳定实时 |
| 云端配置 | 服务端选择 | 服务端选择 | 仅客户端桥接与契约 |

自动硬件路由适合开发环境，但正式发行包应固定并验证唯一后端。否则“应用能启动”可能掩盖主模型加载失败和静默降级。

## 云端推理契约

云端能力默认关闭。设置 `TRANS_MODEL_PROFILE=cloud`，或单独选择 `TRANS_ASR_BACKEND=cloud` / `TRANS_MT_BACKEND=cloud` 后，客户端使用简单 JSON/NDJSON 协议。

### 鉴权

配置 `TRANS_CLOUD_API_KEY` 后发送：

```http
Authorization: Bearer <key>
```

额外固定请求头可以通过 `TRANS_CLOUD_HEADERS` JSON 配置。

### ASR

`POST /v1/asr/transcribe`

```json
{
  "audio_pcm16_b64": "<base64 little-endian pcm16>",
  "sample_rate": 16000,
  "language": "en",
  "model": "server-model-name",
  "provider": "generic"
}
```

最小响应：

```json
{
  "text": "recognized text",
  "language": "en",
  "duration": 3.2,
  "segments": []
}
```

### MT

`POST /v1/mt/translate`

```json
{
  "text": "hello",
  "source_lang": "en",
  "target_lang": "zh",
  "glossary": {"TransLive": "TransLive"},
  "model": "server-model-name",
  "provider": "generic",
  "stream": false
}
```

最小响应：

```json
{"translated": "你好"}
```

### 流式 MT

`POST /v1/mt/translate-stream` 返回 `application/x-ndjson`：

```json
{"token": "你"}
{"token": "好"}
```

流式端点返回 HTTP 404 时，客户端会退回非流式 MT 请求。正式接入收费服务前，还需要补充重试幂等、用量计费、错误码、隐私政策和服务端速率限制。

## 仍需改进

### P0：发布可信度

- 建立 CI，至少运行单元测试、secret scan、Markdown 链接检查和基础静态检查。
- 为发布流程增加冻结 App 冒烟测试，确认 Qwen3 与 Hy-MT2 实际加载，而不是只检查 PyInstaller 退出码。
- 在干净的 macOS 14 和 15 机器上保留一套发布验收记录。
- 正式发行时自动保存签名、公证、staple 和 Gatekeeper 验证结果。

### P1：可观测性与性能

- 建立固定中英文真实音频集，分别记录 VAD 等待、ASR、MT、端到端延迟和峰值内存。
- 在界面中区分“等待分段”和“模型推理”，避免用户把静音等待误认为 ASR 很慢。
- 持续测试 8 GB、16 GB 和 32 GB 机器上的内存压力与模型释放行为。
- 为系统音频采集增加权限诊断和可理解的恢复提示。

### P2：产品与平台

- 实现菜单栏状态机，并验证空闲时资源占用与退出一致性。
- 把体积较大的 Python 桌面壳与模型服务边界进一步明确，再决定是否迁移原生 Swift 壳。
- 为 Windows/Linux 建立独立锁文件、安装说明和 CI 后，再宣称正式跨平台支持。
- 如需 Mac App Store，上线前单独完成沙盒、模型下载、收据验证和审核政策评估。

## 当前已知限制

- 本地 ASR 与 MT 同时常驻会占用数 GB 内存，8 GB 机器可能出现明显内存压力。
- 首次模型下载较大，并受 Hugging Face 网络和磁盘空间影响。
- 系统音频权限由 macOS 管理，授权变化后通常需要重启应用。
- 单机架构不是多用户服务器；不要把本地 WebSocket 端点直接暴露到公网。
- 自动识别和机器翻译都可能出错，不适合作为医疗、法律或安全关键记录的唯一依据。
- App Store 兼容性尚未完成；Developer ID 可分发不等于通过商店审核。

## 发布原则

1. 源码、签名测试包、带到期日试用包和正式公证包使用清晰的独立命名。
2. 模型权重、密钥和用户数据不进入 GitHub。
3. 每个发布声明都以实际后端健康信息、目标系统测试和签名/公证结果为证据。
4. 新模型先做真实音频准确率、延迟、能耗和冻结包测试，再替换默认模型。

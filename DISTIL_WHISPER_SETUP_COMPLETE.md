# 旧版 Distil-Whisper 迁移记录

> 已归档：这个文件名来自早期 Windows/Distil-Whisper 迁移阶段，不代表当前机器或当前发行包已经安装、加载或验证了 Distil-Whisper。

## 当前结论

- macOS Apple Silicon 正式分发版已改为 Qwen3-ASR 1.7B 8-bit。
- 正式 App 不包含 Whisper 权重，并关闭 ASR 静默回退。
- Windows/Linux 源码模式仍可选择 CTranslate2、OpenVINO 或 Transformers 后端。
- 是否成功启用某个后端，必须以运行时健康信息和真实转写测试为准。

## 为什么保留

旧仓库或外部链接可能仍引用本文件，因此不直接删除。原先的 Windows 本机路径、安装完成声明、固定速度对比和旧 fallback 描述均已移除，避免被当作当前发布状态。

## 当前入口

- 项目安装与使用：[README.md](README.md)
- macOS 构建与分发：[MACOS_MIGRATION.md](MACOS_MIGRATION.md)
- 可选 Whisper 后端：[DISTIL_WHISPER_GUIDE.md](DISTIL_WHISPER_GUIDE.md)
- 当前架构与路线图：[docs/PROJECT_AUDIT_AND_ROADMAP.md](docs/PROJECT_AUDIT_AND_ROADMAP.md)

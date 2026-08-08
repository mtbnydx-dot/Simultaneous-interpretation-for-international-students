# Windows 旧启动脚本

此目录只保留早期 Windows 版本的批处理启动脚本，目的是让历史使用者能够辨认旧入口。它们不参与当前 macOS App 构建，也不代表 Windows 版本仍按同等标准维护或测试。

仓库不会提交 Windows 虚拟环境、Python 可执行文件、模型、llama.cpp 二进制或安装工具。需要在 Windows 上运行时，应从根目录源码重新创建本机环境，并根据硬件选择 CTranslate2/CUDA、OpenVINO 或 CPU 后端。

当前正式支持范围和源码安装入口见 [项目 README](../README.md)。在建立 Windows 专用依赖锁、CI 和发布测试之前，请把这些脚本视为兼容参考，而不是正式发行包。

# macOS 安装、构建与分发

本文说明 TransLive 当前 macOS 版本的运行边界、开发环境、模型准备、系统权限、签名与公证流程。旧的 Windows 启动脚本保留在 `windows_legacy/`，不参与 macOS App 构建。

## 支持范围

- 处理器：Apple Silicon (`arm64`)
- 最低目标系统：macOS 14.0
- ASR：Qwen3-ASR 1.7B 8-bit，通过 MLX/Metal 运行
- MT：Hy-MT2 1.8B `Q4_K_M` GGUF，通过 llama.cpp/Metal 运行
- 系统音频：ScreenCaptureKit，仅桌面壳提供
- 分发形式：PyInstaller + PyWebView `.app`

当前 App 不支持 Intel Mac。构建脚本会检查 App 内 Mach-O 文件的最低系统版本；只修改 `LSMinimumSystemVersion` 不能让较新系统编译的依赖自动兼容旧系统。

## 开发环境

先安装 Xcode Command Line Tools：

```bash
xcode-select --install
```

建议使用 Python 3.12：

```bash
python3 -m venv .venv-macos
source .venv-macos/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-app.txt
cp .env.example .env
```

网页模式：

```bash
python run.py
```

桌面模式：

```bash
python desktop_launcher.py
```

## 模型

分发包不包含模型。首次启动时，桌面版会提示用户下载，并保存到：

```text
~/Library/Application Support/TransLive/models/
```

源码模式也可以运行：

```bash
python scripts/download_models.py --all
```

macOS 正式分发环境会设置：

```text
TRANS_ASR_BACKEND=qwen3
TRANS_ASR_FALLBACKS_ENABLED=false
```

因此，Qwen3-ASR 缓存损坏、模型不完整或 `mlx-audio` 缺失时，应用应明确提示错误，而不是退回 Whisper。不要把 Whisper 模型 ID 写入 `TRANS_ASR_QWEN3_MODEL_ID`。

## Metal 检查

Qwen3-ASR 的 MLX wheel 和 Hy-MT2 的 llama.cpp 动态库都必须满足最低系统版本要求。构建脚本会：

1. 从 `requirements-macos-arm64.lock` 安装带哈希的依赖。
2. 强制选择目标 macOS 版本可用的 MLX/MLX Metal wheel。
3. 检查 `llama-cpp-python` 是否支持 GPU offload。
4. 必要时以 `GGML_METAL=on` 和目标部署版本重新编译 llama.cpp。
5. 检查 App 内所有可识别 Mach-O 文件的最低系统版本。

如果手动安装 llama.cpp，可使用：

```bash
CMAKE_ARGS="-DGGML_METAL=on -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0" \
FORCE_CMAKE=1 \
python -m pip install --force-reinstall --no-cache-dir \
  --no-binary=llama-cpp-python llama-cpp-python
```

正式构建仍应使用仓库脚本和锁文件，避免本机临时依赖进入发布包。

## 麦克风与系统音频

麦克风可用于网页模式和桌面模式。系统音频依赖 PyObjC 与 ScreenCaptureKit，只在桌面壳中启用。

首次使用时，在“系统设置 -> 隐私与安全性”中检查：

- 麦克风
- 屏幕与系统音频录制（不同 macOS 小版本的名称可能略有不同）

如果系统音频选项为灰色：

1. 确认运行的是 `TransLive.app` 或 `desktop_launcher.py`，不是直接打开 `web/index.html`。
2. 确认当前系统为 macOS 14 或更高版本，机器为 Apple Silicon。
3. 关闭应用，在系统设置中重新授权后再启动。
4. 查看应用日志中的 `ScreenCaptureKit` 或 `native audio` 错误。

系统音频采集不需要 BlackHole。网页浏览器自己的 `getUserMedia` 通常只能提供麦克风，这不代表桌面版原生采集不可用。

## 本机测试包

```bash
./scripts/build_macos_app.sh
```

默认产物：

```text
dist/TransLive.app
dist/TransLive-macOS-arm64.zip
```

没有指定证书时，脚本会使用 ad-hoc 签名。该包只用于本机测试，不应作为公开分发包。

## Developer ID 签名

先查看钥匙串中可用的代码签名身份：

```bash
security find-identity -v -p codesigning
```

再构建签名包：

```bash
TRANS_CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" \
./scripts/build_macos_app.sh
```

脚本会为内部 Mach-O 文件、主程序和 App 外层依次签名，并使用 `scripts/entitlements.plist`。签名成功不等于已经公证，也不等于 Gatekeeper 一定接受。

## Apple 公证

先把公证凭据存入钥匙串：

```bash
xcrun notarytool store-credentials translive-notary
```

签名、公证并装订票据：

```bash
TRANS_CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" \
TRANS_NOTARY_PROFILE="translive-notary" \
TRANS_DISTRIBUTION=1 \
./scripts/build_macos_app.sh
```

`TRANS_DISTRIBUTION=1` 会要求公证配置存在，防止把“只有签名”的包误当成正式分发包。

## 试用构建

需要生成带到期日的测试包时，可以在构建环境中设置 ISO 日期：

```bash
TRANS_APP_EXPIRY_DATE="YYYY-MM-DD" \
./scripts/build_macos_app.sh
```

到期日会写入构建配置，不需要把日期写死到源码。正式开源版本不应默认设置到期日。

## 发布验证

所有构建都应先验证签名结构：

```bash
codesign --verify --deep --strict --verbose=2 dist/TransLive.app
codesign -dvvv dist/TransLive.app
```

Developer ID 公证包再验证 Gatekeeper 与装订票据：

```bash
spctl --assess --type execute --verbose=4 dist/TransLive.app
xcrun stapler validate dist/TransLive.app
```

还应在一台没有开发环境、没有模型缓存的目标 Mac 上验证首次下载、权限提示、ASR、MT、系统音频、退出和再次启动。`codesign` 通过只说明签名结构有效；ad-hoc 测试包不会通过 Developer ID 公证验收。

## App Store 说明

仓库提供 `scripts/entitlements.appstore.plist` 和 `TRANS_APPSTORE=1` 构建入口，但这不代表当前 PyWebView、本地 HTTP 服务、运行时模型下载与第三方模型许可已经满足 Mac App Store 审核要求。Developer ID 站外分发与 Mac App Store 上架是两套不同流程，应分别做沙盒、收据验证、下载机制和审核政策评估。

## 常见问题

### MT 显示 CPU

确认 `llama-cpp-python` 支持 GPU offload，并通过构建脚本重新生成包。仅安装默认 pip wheel 不保证包含满足当前部署目标的 Metal 动态库。

### Qwen3-ASR 报 `load_npz` 或 zip 文件错误

常见原因是模型缓存不完整、下载中断，或把不兼容的模型文件当成 MLX 权重读取。删除对应的损坏缓存并从应用内重新下载；不要把 Whisper 权重放进 Qwen3 模型目录。

### 其他用户看到“已损坏”

先确认使用 Developer ID Application 证书签名，再确认公证成功且票据已 staple。让用户执行 `xattr -cr` 只能作为开发排障手段，不能替代正确签名与公证。

### 应用退出后仍有进程

查看日志中 ASR、MT、ScreenCaptureKit 和 Uvicorn 的关闭记录。发布验收必须覆盖 Dock 退出、窗口关闭以及采集过程中退出三条路径。

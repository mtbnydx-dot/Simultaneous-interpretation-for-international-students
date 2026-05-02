# TransLive macOS Menu Bar Plan

目标：把 TransLive 做成适合常驻的 macOS 翻译工具，同时保留现有主窗口，避免用户第一次启动后找不到入口。

## 推荐形态

第一阶段不直接设置 `LSUIElement=true`。继续保留 Dock 图标和主窗口，在主窗口里提供“菜单栏常驻”开关。用户开启后，下次启动进入菜单栏模式；菜单栏图标负责打开主窗口、字幕窗、开始/停止系统音频采集、查看模型状态和退出。

第二阶段再提供纯菜单栏发行模式：`LSUIElement=true`，启动后不显示 Dock 图标，只显示菜单栏图标。这个模式适合已经熟悉软件的用户，也适合以后做付费版的“轻量常驻”体验。

## 菜单项

- 打开主窗口
- 打开/关闭字幕窗
- 音频输入：系统音频 / 麦克风
- 开始同传 / 停止同传
- 字幕显示：双语 / 仅译文
- 模型状态：ASR / MT / Metal
- 下载或打开模型目录
- 偏好设置
- 退出 TransLive

## 技术方案

继续复用现有 `DesktopApi` 和 `NativeSystemAudioBridge`。菜单栏只做控制层，不再另起一条翻译管线。

实现上优先用 PyObjC 的 `NSStatusItem`，由 `desktop_launcher.py` 创建菜单栏图标和菜单。菜单动作调用同一个桌面 API：打开 pywebview 主窗口、打开悬浮字幕窗、启动/停止原生系统音频采集。

状态更新建议从后端 `/api/health` 轮询，也可以后续增加轻量事件总线。菜单标题显示当前状态，例如 `TransLive · Ready`、`TransLive · Recording`、`TransLive · Models needed`。

## 注意事项

不要在第一版直接隐藏 Dock 图标。当前应用还需要模型下载弹窗、权限提示、调试反馈和设置入口，纯菜单栏会增加新用户困惑。

系统音频采集依赖 ScreenCaptureKit 权限。菜单栏模式下首次申请屏幕录制权限时，要主动打开一个说明窗口，告诉用户到系统设置授权并重启应用。

如果上 Mac App Store，需要单独评估沙盒下的菜单栏、localhost server、系统音频采集和模型下载路径。官网分发版优先做 Developer ID 签名和公证。

## 建议顺序

1. 先完成 Developer ID 签名、公证和原生系统音频稳定性测试。
2. 加菜单栏图标，但默认仍显示主窗口。
3. 加“启动时隐藏主窗口”偏好。
4. 最后再考虑单独的纯菜单栏构建。

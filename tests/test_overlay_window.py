"""
字幕悬浮窗的原生行为回归测试（仅 macOS）。

覆盖两个真实故障：

1. pywebview 给窗口设的 collectionBehavior 是 NSWindowCollectionBehaviorManaged，
   窗口只属于一个 Space。用户把视频 / Zoom / Keynote 切到全屏时会新建一个 Space，
   字幕窗留在原桌面上直接看不见 —— 而全屏观看恰恰是同传字幕的主场景。

2. pywebview 的 mouseDragged_ 钳制写反了方向：
       newOrigin.y = screenFrame.origin.y + (screenFrame.size.height + windowFrame.size.height)
   窗口顶部一旦越过屏幕上沿，整个窗口会被甩到屏幕外。字幕窗是无边框置顶的，
   甩出去之后没有任何办法用鼠标抓回来。
"""

import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only native window behaviour")

AppKit = pytest.importorskip("AppKit")

from desktop_launcher import DesktopApi  # noqa: E402


@pytest.fixture
def borderless_window():
    AppKit.NSApplication.sharedApplication()
    window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        AppKit.NSMakeRect(200, 200, 960, 260),
        AppKit.NSWindowStyleMaskBorderless,
        AppKit.NSBackingStoreBuffered,
        False,
    )
    window.setLevel_(AppKit.NSStatusWindowLevel)
    window.setOpaque_(False)
    yield window
    window.close()


def _visible_fraction(window) -> float:
    frame = window.frame()
    best = 0.0
    for screen in AppKit.NSScreen.screens():
        visible = screen.visibleFrame()
        dx = max(0.0, min(frame.origin.x + frame.size.width, visible.origin.x + visible.size.width)
                 - max(frame.origin.x, visible.origin.x))
        dy = max(0.0, min(frame.origin.y + frame.size.height, visible.origin.y + visible.size.height)
                 - max(frame.origin.y, visible.origin.y))
        best = max(best, dx * dy)
    return best / (frame.size.width * frame.size.height)


def test_overlay_collection_behavior_spans_spaces_and_fullscreen(borderless_window):
    borderless_window.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        | AppKit.NSWindowCollectionBehaviorStationary
    )
    behavior = borderless_window.collectionBehavior()

    assert behavior & AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
    assert behavior & AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
    # Managed 会把窗口绑死在单一 Space 上，必须没有
    assert not behavior & AppKit.NSWindowCollectionBehaviorManaged


def test_window_flung_off_screen_by_the_pywebview_drag_bug_is_recovered(borderless_window):
    screen = AppKit.NSScreen.mainScreen().visibleFrame()
    # 复现上游那句钳制算出来的坐标
    flung_y = screen.origin.y + (screen.size.height + borderless_window.frame().size.height)
    borderless_window.setFrameOrigin_(AppKit.NSMakePoint(300, flung_y))
    assert _visible_fraction(borderless_window) < 0.1

    assert DesktopApi._clamp_overlay_into_view(borderless_window) is True
    assert _visible_fraction(borderless_window) > 0.99


@pytest.mark.parametrize(
    "origin, label",
    [
        ((3000.0, 1800.0), "已拔掉的外接显示器上的旧坐标"),
        ((-900.0, 300.0), "拖到屏幕左外侧"),
        ((300.0, -600.0), "拖到屏幕下方"),
    ],
)
def test_offscreen_positions_are_pulled_back(borderless_window, origin, label):
    borderless_window.setFrameOrigin_(AppKit.NSMakePoint(*origin))
    DesktopApi._clamp_overlay_into_view(borderless_window)
    assert _visible_fraction(borderless_window) > 0.99, label


def test_a_window_already_on_screen_is_left_alone(borderless_window):
    borderless_window.setFrameOrigin_(AppKit.NSMakePoint(300, 200))
    before = borderless_window.frame()

    assert DesktopApi._clamp_overlay_into_view(borderless_window) is False

    after = borderless_window.frame()
    assert (after.origin.x, after.origin.y) == (before.origin.x, before.origin.y)


# ── 点击穿透的区域命中判定 ────────────────────────────────────────────
# NSWindow 的 ignoresMouseEvents 是整窗开关，只能靠按光标位置动态切换来实现
# 「只有字幕正文穿透、工具栏照常可点」。这里的坐标翻转最容易写错：
# 前端给的是 CSS 坐标（左上角原点、y 向下），AppKit 是左下角原点、y 向上。

class _FrameStub:
    def __init__(self, x, y, w, h):
        self._frame = AppKit.NSMakeRect(x, y, w, h)

    def frame(self):
        return self._frame


@pytest.fixture
def api_with_zones():
    api = DesktopApi("http://127.0.0.1:8766", "test-token")
    api.set_subtitle_interactive_zones([
        {"x": 0, "y": 0, "w": 960, "h": 44},      # 工具栏：顶部整条
        {"x": 932, "y": 232, "w": 24, "h": 24},   # 缩放手柄：右下角
    ])
    return api


# 窗口位于屏幕 (400,300)，960x260 → CSS y=0 对应 AppKit y=560
@pytest.mark.parametrize(
    "point, click_passes_through, label",
    [
        ((900, 550), False, "工具栏中部"),
        ((410, 545), False, "工具栏最左端"),
        ((1350, 550), False, "工具栏最右端"),
        ((900, 515), True, "工具栏下沿之外"),
        ((900, 430), True, "字幕正文中间"),
        ((900, 310), True, "字幕正文底部"),
        ((1345, 312), False, "右下缩放手柄"),
        ((1345, 380), True, "手柄正上方的正文"),
        ((200, 430), True, "窗口外"),
    ],
)
def test_only_chrome_blocks_the_cursor(api_with_zones, point, click_passes_through, label):
    window = _FrameStub(400, 300, 960, 260)
    over_chrome = api_with_zones._point_is_over_chrome(window, AppKit.NSMakePoint(*point))
    assert (not over_chrome) is click_passes_through, label


def test_zones_are_cleared_when_the_overlay_closes():
    api = DesktopApi("http://127.0.0.1:8766", "test-token")
    api.set_subtitle_interactive_zones([{"x": 0, "y": 0, "w": 960, "h": 44}])
    api._click_through = True

    api._overlay_closed()

    assert api._interactive_zones == []
    assert api._click_through is False
    assert api._mouse_monitors == []


def test_malformed_zones_are_ignored():
    api = DesktopApi("http://127.0.0.1:8766", "test-token")
    result = api.set_subtitle_interactive_zones([
        {"x": 0, "y": 0, "w": 960, "h": 44},
        {"x": 0, "y": 0, "w": 0, "h": 44},        # 零宽，丢弃
        {"x": "nope", "y": 0, "w": 10, "h": 10},  # 类型错，丢弃
        {"x": 0},                                  # 缺字段，丢弃
    ])
    assert result["zones"] == 1

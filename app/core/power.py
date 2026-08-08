"""
macOS 实时活动声明。

`NSActivityLatencyCritical` 会关掉定时器合并和 CPU 空闲降频、并阻止 App Nap。
同传运行时确实需要它（音频回调不能被推迟），但**只应该在同传运行时持有**。

之前的实现在 App 启动时就申请、退出时才释放，等于用户开着窗口发呆也在阻止
系统进入低功耗状态，这是最大的一处无谓耗电。这里改成按会话引用计数持有。
"""

import logging
import sys
import threading

logger = logging.getLogger(__name__)


class RealtimeActivity:
    """引用计数的 NSProcessInfo activity 持有者。非 macOS 上是空操作。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token = None
        self._refcount = 0

    @property
    def active(self) -> bool:
        with self._lock:
            return self._token is not None

    @property
    def holders(self) -> int:
        with self._lock:
            return self._refcount

    def acquire(self, reason: str = "TransLive live translation session") -> None:
        if sys.platform != "darwin":
            return
        with self._lock:
            self._refcount += 1
            if self._token is not None:
                return
            try:
                from Foundation import (
                    NSActivityLatencyCritical,
                    NSActivityUserInitiatedAllowingIdleSystemSleep,
                    NSProcessInfo,
                )

                options = NSActivityLatencyCritical | NSActivityUserInitiatedAllowingIdleSystemSleep
                self._token = NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
                    options, reason
                )
                logger.info("Acquired macOS realtime activity assertion (%s)", reason)
            except Exception:
                # PyObjC 不可用时不该让同传起不来，降级为不声明即可。
                logger.debug("Realtime activity assertion unavailable", exc_info=True)

    def release(self) -> None:
        if sys.platform != "darwin":
            return
        with self._lock:
            if self._refcount > 0:
                self._refcount -= 1
            if self._refcount > 0 or self._token is None:
                return
            token, self._token = self._token, None
            try:
                from Foundation import NSProcessInfo

                NSProcessInfo.processInfo().endActivity_(token)
                logger.info("Released macOS realtime activity assertion")
            except Exception:
                logger.debug("Failed to end realtime activity assertion", exc_info=True)


realtime_activity = RealtimeActivity()

import asyncio
import time
from collections import OrderedDict
from typing import Any


class SubtitleHub:
    def __init__(self, max_history: int = 80):
        self._max_history = max_history
        self._segments: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = asyncio.Lock()
        self._dropped_messages = 0

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            segments = list(self._segments.values())
        return {"type": "snapshot", "segments": segments}

    async def status(self) -> dict[str, Any]:
        async with self._lock:
            queue_sizes = [queue.qsize() for queue in self._subscribers]
            return {
                "history_size": len(self._segments),
                "max_history": self._max_history,
                "subscribers": len(self._subscribers),
                "max_queue_size": max(queue_sizes, default=0),
                "dropped_messages": self._dropped_messages,
            }

    async def clear(self) -> None:
        async with self._lock:
            self._segments.clear()
            subscribers = list(self._subscribers)
        await self._broadcast({"type": "clear"}, subscribers)

    async def publish(self, payload: dict[str, Any]) -> None:
        message = dict(payload)
        now = time.time()
        async with self._lock:
            self._remember(message, now)
            subscribers = list(self._subscribers)
        await self._broadcast(message, subscribers)

    def _remember(self, message: dict[str, Any], now: float) -> None:
        event_type = message.get("type")
        # 滚动 ASR 预览只用于当前画面，不能占用 80 条最终字幕历史，也不能
        # 在新订阅者的 snapshot 中伪装成已定稿片段。
        if event_type not in {"original", "translated_partial", "translated", "error"}:
            return

        seg_id = message.get("segment_id")
        if not isinstance(seg_id, int):
            return

        segment = self._segments.get(seg_id)
        if segment is None:
            segment = {
                "segment_id": seg_id,
                "original_text": "",
                "translated_text": "",
                "source_lang": message.get("source_lang"),
                "target_lang": message.get("target_lang"),
                "state": "pending",
                "created_at": now,
                "updated_at": now,
            }
            self._segments[seg_id] = segment

        if message.get("source_lang"):
            segment["source_lang"] = message["source_lang"]
        if message.get("target_lang"):
            segment["target_lang"] = message["target_lang"]

        if event_type == "original":
            segment["original_text"] = message.get("text", "")
            segment["audio_duration_ms"] = message.get("audio_duration_ms")
            segment["asr_ms"] = message.get("asr_ms")
            segment["state"] = "pending"
        elif event_type == "translated_partial":
            segment["translated_text"] = message.get("accumulated") or message.get("text", "")
            segment["state"] = "streaming"
        elif event_type == "translated":
            segment["translated_text"] = message.get("text", "")
            segment["audio_duration_ms"] = message.get("audio_duration_ms")
            segment["asr_ms"] = message.get("asr_ms")
            segment["mt_ms"] = message.get("mt_ms")
            segment["total_ms"] = message.get("total_ms")
            segment["rtf"] = message.get("rtf")
            segment["warning"] = message.get("warning")
            segment["truncated"] = bool(message.get("truncated"))
            segment["state"] = "final"
        elif event_type == "error":
            segment["translated_text"] = "[Error] " + str(message.get("text", "未知错误"))
            segment["warning"] = message.get("warning")
            segment["state"] = "error"

        segment["updated_at"] = now
        self._segments.move_to_end(seg_id)
        while len(self._segments) > self._max_history:
            self._segments.popitem(last=False)

    async def _broadcast(
        self,
        message: dict[str, Any],
        subscribers: list[asyncio.Queue[dict[str, Any]]],
    ) -> None:
        for queue in subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    self._dropped_messages += 1
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    self._dropped_messages += 1
                    pass


subtitle_hub = SubtitleHub()

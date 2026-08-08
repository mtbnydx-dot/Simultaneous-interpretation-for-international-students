import asyncio
import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)


def _iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


class TranscriptStore:
    """Small JSONL-backed store for finalized transcript segments."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._segments: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        self._last_prune_at = 0.0
        self._stats_lock = threading.Lock()
        self._records_written = 0
        self._write_failures = 0
        self._last_write_at = 0.0
        self._last_write_path: str | None = None
        self._last_write_error: str | None = None
        self._file_stats_cache_at = 0.0
        self._file_stats_cache_dir: str | None = None
        self._file_stats_cache: tuple[int, int] = (0, 0)

    @property
    def run_id(self) -> str:
        return self._run_id

    def history_dir(self) -> Path:
        if settings.transcript_history_dir:
            return Path(settings.transcript_history_dir).expanduser()
        if sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / "TransLive" / "history"
        return PROJECT_ROOT / "logs" / "transcripts"

    def current_file(self) -> Path:
        return self.history_dir() / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"

    def status(self) -> dict[str, Any]:
        history_dir = self.history_dir()
        file_count, total_bytes = self._cached_history_file_stats(history_dir)
        with self._stats_lock:
            records_written = self._records_written
            write_failures = self._write_failures
            last_write_at = self._last_write_at
            last_write_path = self._last_write_path
            last_write_error = self._last_write_error
        return {
            "enabled": settings.transcript_history_enabled,
            "run_id": self._run_id,
            "history_dir": str(history_dir),
            "current_file": str(self.current_file()),
            "memory_segments": len(self._segments),
            "max_memory_segments": settings.transcript_history_max_memory,
            "retention_days": settings.transcript_history_retention_days,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "records_written": records_written,
            "write_failures": write_failures,
            "last_write_at": last_write_at or None,
            "last_write_at_iso": _iso_from_ts(last_write_at) if last_write_at else None,
            "last_write_path": last_write_path,
            "last_write_error": last_write_error,
        }

    async def clear_memory(self) -> None:
        async with self._lock:
            self._segments.clear()

    async def clear_all(self) -> dict[str, Any]:
        await self.clear_memory()
        deleted_files, deleted_bytes = await asyncio.to_thread(self._clear_history_files)
        return {
            "ok": True,
            "deleted_files": deleted_files,
            "deleted_bytes": deleted_bytes,
            "history": self.status(),
        }

    async def record_event(self, payload: dict[str, Any]) -> None:
        if not settings.transcript_history_enabled:
            return
        event_type = payload.get("type")
        if event_type not in {"original", "translated", "error"}:
            return
        await self.maybe_prune_old_files()
        seg_id = payload.get("segment_id")
        if not isinstance(seg_id, int):
            return

        now = time.time()
        record: dict[str, Any] | None = None
        async with self._lock:
            segment = self._segments.get(seg_id)
            if segment is None:
                segment = {
                    "schema": 1,
                    "run_id": self._run_id,
                    "segment_id": seg_id,
                    "original_text": "",
                    "translated_text": "",
                    "source_lang": payload.get("source_lang"),
                    "target_lang": payload.get("target_lang"),
                    "state": "pending",
                    "created_at": now,
                    "updated_at": now,
                }
                self._segments[seg_id] = segment

            if payload.get("source_lang"):
                segment["source_lang"] = payload["source_lang"]
            if payload.get("target_lang"):
                segment["target_lang"] = payload["target_lang"]

            if event_type == "original":
                segment["original_text"] = payload.get("text", "")
                segment["audio_duration_ms"] = payload.get("audio_duration_ms")
                segment["asr_ms"] = payload.get("asr_ms")
                segment["state"] = "pending"
            elif event_type == "translated":
                segment["translated_text"] = payload.get("text", "")
                segment["audio_duration_ms"] = payload.get("audio_duration_ms")
                segment["asr_ms"] = payload.get("asr_ms")
                segment["mt_ms"] = payload.get("mt_ms")
                segment["total_ms"] = payload.get("total_ms")
                segment["rtf"] = payload.get("rtf")
                segment["warning"] = payload.get("warning")
                segment["truncated"] = bool(payload.get("truncated"))
                segment["state"] = "final"
            elif event_type == "error":
                segment["translated_text"] = "[Error] " + str(payload.get("text", "未知错误"))
                segment["warning"] = payload.get("warning")
                segment["state"] = "error"

            segment["updated_at"] = now
            self._segments.move_to_end(seg_id)
            while len(self._segments) > max(1, settings.transcript_history_max_memory):
                self._segments.popitem(last=False)

            if event_type in {"translated", "error"}:
                record = self._record_from_segment(segment)

        if record is not None:
            try:
                path = await asyncio.to_thread(self._append_record, record)
                self._mark_write_success(path)
            except Exception as exc:
                self._mark_write_failure(exc)
                logger.exception("Failed to persist transcript record")

    async def recent(self, limit: int = 80) -> list[dict[str, Any]]:
        if not settings.transcript_history_enabled:
            return []
        safe_limit = max(1, min(int(limit or 80), 500))
        return await asyncio.to_thread(self._read_recent_records, safe_limit)

    async def maybe_prune_old_files(self) -> int:
        if not settings.transcript_history_enabled:
            return 0
        now = time.time()
        if now - self._last_prune_at < 3600:
            return 0
        self._last_prune_at = now
        return await self.prune_old_files()

    async def prune_old_files(self) -> int:
        retention_days = int(settings.transcript_history_retention_days or 0)
        if retention_days <= 0:
            return 0
        try:
            return await asyncio.to_thread(self._prune_old_files, retention_days)
        except Exception:
            logger.exception("Failed to prune old transcript records")
            return 0

    def _record_from_segment(self, segment: dict[str, Any]) -> dict[str, Any]:
        record = dict(segment)
        record["record_id"] = f"{record['run_id']}:{record['segment_id']}"
        record["created_at_iso"] = _iso_from_ts(float(record["created_at"]))
        record["updated_at_iso"] = _iso_from_ts(float(record["updated_at"]))
        return record

    def _append_record(self, record: dict[str, Any]) -> str:
        path = self.current_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
        return str(path)

    def _mark_write_success(self, path: str) -> None:
        with self._stats_lock:
            self._records_written += 1
            self._last_write_at = time.time()
            self._last_write_path = path
            self._last_write_error = None
            self._file_stats_cache_at = 0.0

    def _mark_write_failure(self, exc: Exception) -> None:
        with self._stats_lock:
            self._write_failures += 1
            self._last_write_error = str(exc)

    def _read_recent_records(self, limit: int) -> list[dict[str, Any]]:
        history_dir = self.history_dir()
        if not history_dir.exists():
            return []
        files = sorted(history_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for path in files[:7]:
            try:
                lines = self._read_tail_lines(path, max_lines=max(limit * 5, 200))
            except OSError:
                continue
            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record_id = str(record.get("record_id") or "")
                if record_id and record_id in seen:
                    continue
                if record_id:
                    seen.add(record_id)
                records.append(record)
                if len(records) >= limit:
                    break
            if len(records) >= limit:
                break
        records.reverse()
        return records

    def _read_tail_lines(self, path: Path, max_lines: int) -> list[str]:
        safe_max_lines = max(1, int(max_lines or 1))
        chunk_size = 64 * 1024
        blocks: list[bytes] = []
        newline_count = 0

        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            position = f.tell()
            while position > 0 and newline_count <= safe_max_lines:
                read_size = min(chunk_size, position)
                position -= read_size
                f.seek(position)
                block = f.read(read_size)
                blocks.append(block)
                newline_count += block.count(b"\n")

        if not blocks:
            return []

        data = b"".join(reversed(blocks))
        if position > 0:
            first_newline = data.find(b"\n")
            if first_newline >= 0:
                data = data[first_newline + 1:]

        return data.decode("utf-8", errors="replace").splitlines()[-safe_max_lines:]

    def _history_file_stats(self, history_dir: Path) -> tuple[int, int]:
        if not history_dir.exists():
            return 0, 0
        file_count = 0
        total_bytes = 0
        for path in history_dir.glob("*.jsonl"):
            try:
                stat = path.stat()
            except OSError:
                continue
            file_count += 1
            total_bytes += int(stat.st_size)
        return file_count, total_bytes

    def _cached_history_file_stats(self, history_dir: Path) -> tuple[int, int]:
        key = str(history_dir)
        now = time.time()
        with self._stats_lock:
            if (
                self._file_stats_cache_dir == key
                and now - self._file_stats_cache_at < 5.0
            ):
                return self._file_stats_cache

        stats = self._history_file_stats(history_dir)
        with self._stats_lock:
            self._file_stats_cache_dir = key
            self._file_stats_cache_at = now
            self._file_stats_cache = stats
        return stats

    def _invalidate_file_stats_cache(self) -> None:
        with self._stats_lock:
            self._file_stats_cache_at = 0.0

    def _clear_history_files(self) -> tuple[int, int]:
        history_dir = self.history_dir()
        if not history_dir.exists():
            return 0, 0
        deleted_files = 0
        deleted_bytes = 0
        for path in history_dir.glob("*.jsonl"):
            try:
                size = path.stat().st_size
                path.unlink()
            except OSError:
                logger.warning("Failed to delete transcript history file: %s", path)
                continue
            deleted_files += 1
            deleted_bytes += int(size)
        if deleted_files:
            self._invalidate_file_stats_cache()
        return deleted_files, deleted_bytes

    def _prune_old_files(self, retention_days: int) -> int:
        history_dir = self.history_dir()
        if not history_dir.exists():
            return 0
        cutoff = datetime.now().astimezone() - timedelta(days=retention_days)
        deleted = 0
        for path in history_dir.glob("*.jsonl"):
            try:
                file_date = datetime.strptime(path.stem, "%Y-%m-%d").astimezone()
            except ValueError:
                try:
                    file_date = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
                except OSError:
                    continue
            if file_date >= cutoff:
                continue
            try:
                path.unlink()
                deleted += 1
            except OSError:
                logger.warning("Failed to delete old transcript history file: %s", path)
        if deleted:
            logger.info("Pruned %d old transcript history files", deleted)
            self._invalidate_file_stats_cache()
        return deleted


transcript_store = TranscriptStore()

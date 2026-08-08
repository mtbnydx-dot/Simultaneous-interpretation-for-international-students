"""
TransLive - AI 同声传译。
FastAPI 应用入口。
"""

# ── 必须在所有导入之前设置，防止 Rich/NNCF 写 Windows GBK 控制台崩溃 ──
import os
os.environ.setdefault("NNCF_PROGRESS_BAR", "false")
os.environ.setdefault("RICH_FORCE_TERMINAL", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from pydantic import BaseModel

from app.asr.engine import asr_engine, unload_asr_engine
from app.mt.engine import MTError, mt_engine
from app.core.session import StreamSession, asr_filter_status
from app.core.session import _latency_mode_settings
from app.core.config import PROJECT_ROOT, settings
from app.core.expiry import expiry_status
from app.core.model_download import download_asr_model, download_mt_model
from app.core.hardware import detect_hardware
from app.core.model_profiles import select_model_plan
from app.core.subtitle_hub import subtitle_hub
from app.core.transcript_store import transcript_store
from app.core.runtime_status import runtime_status
from app.core.languages import (
    is_supported_language,
    is_supported_source_language,
    normalize_language_code,
    normalize_source_language,
    supported_language_options,
    supported_source_language_options,
)
from app.core.local_api import authorize_http, authorize_websocket
from app.core.memory import collect_model_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# 打包后的 app 没有控制台，将日志写入文件便于排查问题
_file_handler: logging.Handler | None = None
if getattr(sys, "frozen", False) and sys.platform == "darwin":
    log_dir = Path.home() / "Library" / "Logs" / "TransLive"
    log_dir.mkdir(parents=True, exist_ok=True)
    _file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    _file_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(_file_handler)
    logging.getLogger().info("TransLive %s starting (frozen, log to %s)", settings.app_version, log_dir)

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent.parent / "web"


def _close_file_log_handler() -> None:
    global _file_handler
    root_logger = logging.getLogger()
    if _file_handler is not None:
        try:
            root_logger.removeHandler(_file_handler)
            _file_handler.flush()
            _file_handler.close()
        except Exception:
            pass
        _file_handler = None


# 模型加载状态 —— 由 /api/health 暴露，前端据此显示「加载中」
_loading_state = {"asr": False, "mt": False}
_stream_session_count = 0
_stream_session_lock = asyncio.Lock()
_model_operation_lock = asyncio.Lock()


async def _try_acquire_stream_slot() -> bool:
    global _stream_session_count
    max_sessions = max(1, int(settings.max_stream_sessions or 1))
    async with _stream_session_lock:
        if _stream_session_count >= max_sessions:
            return False
        _stream_session_count += 1
        return True


async def _release_stream_slot() -> None:
    global _stream_session_count
    async with _stream_session_lock:
        _stream_session_count = max(0, _stream_session_count - 1)


def _load_asr_blocking():
    _loading_state["asr"] = True
    try:
        asr_engine.load_model()
        logger.info("ASR ready")
    except Exception:
        logger.exception("ASR load failed")
    finally:
        _loading_state["asr"] = False


def _load_mt_blocking():
    _loading_state["mt"] = True
    try:
        mt_engine.load_model()
        logger.info("MT ready")
    except Exception:
        logger.exception("MT load failed")
    finally:
        _loading_state["mt"] = False


async def _load_models_sequentially() -> None:
    # Apple Silicon 的 CPU/GPU 共用统一内存；两个大模型同时初始化会制造
    # 明显的峰值内存和磁盘读取，顺序加载对总就绪时间影响很小。
    async with _model_operation_lock:
        await asyncio.to_thread(_load_asr_blocking)
        await asyncio.to_thread(_load_mt_blocking)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s...", settings.app_name)
    model_tasks: list[asyncio.Task] = []
    try:
        app_expiry = expiry_status()
        if app_expiry.expired:
            logger.warning("TransLive expired: %s", app_expiry.message)
            yield
            return

        if settings.transcript_history_enabled:
            await transcript_store.prune_old_files()

        if settings.load_models_on_startup:
            # 后台加载，避免阻塞 FastAPI 启动 & /api/health 返回 404
            model_tasks = [asyncio.create_task(_load_models_sequentially())]
            logger.info("%s accepting requests (models loading in background)", settings.app_name)
        else:
            logger.info("%s accepting requests (model loading deferred)", settings.app_name)
        yield
    finally:
        logger.info("Shutting down %s", settings.app_name)
        for task in model_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        try:
            closed = await asyncio.wait_for(
                asyncio.to_thread(mt_engine.close, 10.0),
                timeout=12.0,
            )
            if not closed:
                logger.warning("MT model remained loaded because inference was still active")
        except Exception:
            logger.exception("MT close failed")
        try:
            await asyncio.wait_for(asyncio.to_thread(unload_asr_engine), timeout=10.0)
        except Exception:
            logger.exception("ASR unload failed")
        _close_file_log_handler()


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.middleware("http")
async def block_expired_http(request, call_next):
    path = request.url.path
    if path.startswith("/api/") and path != "/api/ready":
        allowed, status_code, error = authorize_http(request)
        if not allowed:
            return JSONResponse(status_code=status_code, content={"error": error})
    app_expiry = expiry_status()
    if app_expiry.expired and path not in {"/api/health", "/api/ready"}:
        return JSONResponse(
            status_code=403,
            content={
                "error": "expired",
                "message": app_expiry.message,
                "app_expiry_date": app_expiry.expires_on_iso,
            },
        )
    return await call_next(request)


@app.get("/api/ready")
async def ready():
    return {"status": "ok", "instance_id": settings.instance_id}


@app.get("/api/health")
async def health():
    perf_stats = {}
    try:
        from app.core.perf_monitor import perf_monitor
        perf_stats = perf_monitor.get_stats()
    except Exception:
        pass

    vad_info = {}
    try:
        from app.core import vad as vad_module
        vad = getattr(vad_module, "_vad_instance", None)
        vad_info = {
            "vad_engine": settings.vad_engine,
            "vad_type": type(vad).__name__ if vad is not None else "not_loaded",
        }
    except Exception:
        vad_info = {"vad_engine": settings.vad_engine, "vad_type": "unavailable"}

    asr_loaded = asr_engine.model is not None
    mt_loaded = mt_engine.is_loaded
    asr_loading = _loading_state["asr"]
    mt_loading = _loading_state["mt"] or mt_engine.is_loading
    app_expiry = expiry_status()
    hardware = detect_hardware()
    model_plan = select_model_plan(hardware)

    return {
        "status": "expired" if app_expiry.expired else "ok",
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "app_credit": settings.app_credit,
        "app_expiry_date": app_expiry.expires_on_iso,
        "app_expired": app_expiry.expired,
        "app_expiry_message": app_expiry.message,
        "hardware": hardware.to_dict(),
        "model_profile": settings.model_profile,
        "model_plan": model_plan.to_dict(),
        "ready": (asr_loaded and mt_loaded) and not app_expiry.expired,
        "asr_loaded": asr_loaded,
        "asr_loading": asr_loading,
        "asr_backend": asr_engine.backend_name,
        "asr_device": asr_engine.device,
        "asr_compute_type": asr_engine.compute_type,
        "asr_model_id": settings.asr_model_id,
        "asr_effective_model_id": asr_engine.model_id,
        "asr_model_path": asr_engine.model_path,
        "asr_acceleration": asr_engine.acceleration_info,
        "asr_max_new_tokens": settings.asr_max_new_tokens,
        "mt_loaded": mt_loaded,
        "mt_loading": mt_loading,
        "mt_backend": settings.mt_backend,
        "mt_model_id": settings.mt_model_id,
        "mt_device": mt_engine.device,
        "mt_acceleration": mt_engine.acceleration_info,
        "mt_stream": {
            "queue_max_size": settings.mt_stream_queue_max_size,
            "partial_min_interval_ms": settings.mt_stream_partial_min_interval_ms,
            "partial_min_chars": settings.mt_stream_partial_min_chars,
            "timeout_s": settings.mt_stream_timeout_s,
            "max_output_chars": settings.mt_stream_max_output_chars,
            "stop_on_disconnect": settings.mt_stream_stop_on_disconnect,
            "queue_put_timeout_s": settings.mt_stream_queue_put_timeout_s,
        },
        "mt_full_context": {
            "max_segments": settings.mt_full_context_max_segments,
            "max_chars": settings.mt_full_context_max_chars,
            "max_output_tokens": settings.mt_full_context_max_output_tokens,
        },
        "mt_loaded_pairs": mt_engine.loaded_pairs,
        "mt_unavailable_reason": mt_engine.unavailable_reason,
        "auto_download_models": settings.auto_download_models,
        "load_models_on_startup": settings.load_models_on_startup,
        "cloud": {
            "enabled": settings.cloud_enabled,
            "configured": bool(settings.cloud_base_url),
            "provider": settings.cloud_provider,
            "asr_model": settings.cloud_asr_model,
            "mt_model": settings.cloud_mt_model,
        },
        "desktop_mode": settings.desktop_mode,
        "max_stream_sessions": settings.max_stream_sessions,
        "active_stream_sessions": _stream_session_count,
        "websocket_max_message_bytes": settings.websocket_max_message_bytes,
        "stream_segment_queue_max_size": settings.stream_segment_queue_max_size,
        "model_min_free_disk_gb": settings.model_min_free_disk_gb,
        "segmenting": _latency_mode_settings(settings.session_latency_mode),
        "subtitle_hub": await subtitle_hub.status(),
        "transcript_history": transcript_store.status(),
        "runtime": runtime_status(),
        "source_lang": normalize_source_language(settings.source_lang, "en"),
        "supported_source_languages": supported_source_language_options(),
        "asr_language_autodetect": asr_engine.backend_name == "qwen3",
        "target_lang": normalize_language_code(settings.target_lang, "zh"),
        "supported_languages": supported_language_options(),
        "vad": vad_info,
        "asr_filter": asr_filter_status(),
        "glossary_entries": len(settings.mt_glossary or {}),
        "perf_stats": perf_stats,
    }


class ModelDownloadRequest(BaseModel):
    asr: bool = False
    mt: bool = True
    force: bool = False


class ModelUnloadRequest(BaseModel):
    asr: bool = True
    mt: bool = True


@app.post("/api/models/download")
async def download_models(req: ModelDownloadRequest):
    loop = asyncio.get_running_loop()

    def _download():
        result: dict[str, str | bool] = {"asr": False, "mt": False}
        logger.info("Model download: asr=%s mt=%s force=%s mt_model_id=%s",
                     req.asr, req.mt, req.force, settings.mt_model_id)
        if req.asr:
            _loading_state["asr"] = True
            try:
                download_asr_model()
                result["asr"] = True
                if not asr_engine.model:
                    logger.info("Loading ASR engine...")
                    asr_engine.load_model()
                result["asr_loaded"] = asr_engine.model is not None
            except Exception as exc:
                logger.exception("ASR download/load failed")
                result["asr_error"] = str(exc)
            finally:
                _loading_state["asr"] = False
        if req.mt:
            _loading_state["mt"] = True
            try:
                result["mt_model_id"] = download_mt_model(force=req.force)
                result["mt"] = True
                logger.info("MT model path resolved: %s", result["mt_model_id"])
                if not mt_engine.is_loaded:
                    logger.info("Loading MT engine...")
                    mt_engine.load_model()
                    logger.info("MT load result: loaded=%s reason=%s",
                                 mt_engine.is_loaded, mt_engine.unavailable_reason)
                result["mt_loaded"] = mt_engine.is_loaded
                if not mt_engine.is_loaded:
                    result["mt_error"] = mt_engine.unavailable_reason
            except Exception as exc:
                logger.exception("MT download/load failed")
                result["mt_error"] = str(exc)
            finally:
                _loading_state["mt"] = False
        return result

    if _model_operation_lock.locked():
        return JSONResponse(
            status_code=409,
            content={"ok": False, "error": "model_operation_busy", "message": "已有模型操作正在进行。"},
        )
    async with _model_operation_lock:
        return await loop.run_in_executor(None, _download)


@app.post("/api/models/unload")
async def unload_models(req: ModelUnloadRequest):
    if _stream_session_count > 0:
        return JSONResponse(
            status_code=409,
            content={
                "ok": False,
                "error": "stream_active",
                "message": "同传运行中，停止后再释放模型内存。",
            },
        )

    if _model_operation_lock.locked():
        return JSONResponse(
            status_code=409,
            content={"ok": False, "error": "model_operation_busy", "message": "已有模型操作正在进行。"},
        )

    loop = asyncio.get_running_loop()

    def _unload():
        result: dict[str, bool] = {"ok": True, "asr_unloaded": False, "mt_unloaded": False}
        if req.asr:
            try:
                unload_asr_engine()
                result["asr_unloaded"] = True
            except Exception:
                logger.exception("ASR unload failed")
                result["ok"] = False
        if req.mt:
            try:
                result["mt_unloaded"] = mt_engine.close()
                if not result["mt_unloaded"]:
                    result["ok"] = False
            except Exception:
                logger.exception("MT unload failed")
                result["ok"] = False
        collect_model_memory()
        return result

    async with _model_operation_lock:
        return await loop.run_in_executor(None, _unload)


class TranslateRequest(BaseModel):
    text: str
    source_lang: str | None = None
    target_lang: str | None = None
    glossary: dict[str, str] | None = None


class TranscriptTranslateSegment(BaseModel):
    segment_id: int | str | None = None
    text: str | None = None
    original_text: str | None = None
    source_lang: str | None = None


class TranscriptTranslateRequest(BaseModel):
    text: str | None = None
    segments: list[TranscriptTranslateSegment] | None = None
    source_lang: str | None = None
    target_lang: str | None = None
    glossary: dict[str, str] | None = None


@app.post("/api/translate")
async def translate(req: TranslateRequest):
    invalid_languages = []
    if req.source_lang is not None and not is_supported_language(req.source_lang):
        invalid_languages.append({"field": "source_lang", "value": req.source_lang})
    if req.target_lang is not None and not is_supported_language(req.target_lang):
        invalid_languages.append({"field": "target_lang", "value": req.target_lang})
    if invalid_languages:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_language",
                "invalid": invalid_languages,
                "supported_languages": supported_language_options(),
            },
        )

    source_lang = normalize_language_code(req.source_lang, settings.source_lang)
    target_lang = normalize_language_code(req.target_lang, settings.target_lang)
    loop = asyncio.get_running_loop()
    try:
        translated = await loop.run_in_executor(
            None, mt_engine.translate, req.text, source_lang, target_lang, req.glossary
        )
    except MTError as exc:
        logger.warning("REST translation failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"error": "mt_error", "message": str(exc), "original": req.text},
        )
    return {
        "original": req.text,
        "translated": translated,
        "source_lang": source_lang,
        "target_lang": target_lang,
    }


@app.post("/api/translate/transcript")
async def translate_transcript(req: TranscriptTranslateRequest):
    invalid_languages = []
    if req.source_lang is not None and not is_supported_source_language(req.source_lang):
        invalid_languages.append({"field": "source_lang", "value": req.source_lang})
    if req.target_lang is not None and not is_supported_language(req.target_lang):
        invalid_languages.append({"field": "target_lang", "value": req.target_lang})
    raw_segments = req.segments or []
    for index, segment in enumerate(raw_segments):
        if segment.source_lang is not None and not is_supported_source_language(segment.source_lang):
            invalid_languages.append({
                "field": f"segments[{index}].source_lang",
                "value": segment.source_lang,
            })
    if invalid_languages:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_language",
                "invalid": invalid_languages,
                "supported_languages": supported_language_options(),
            },
        )

    source_lang = normalize_source_language(req.source_lang, settings.source_lang)
    target_lang = normalize_language_code(req.target_lang, settings.target_lang)
    max_segments = max(1, int(settings.mt_full_context_max_segments or 1))
    max_chars = max(1, int(settings.mt_full_context_max_chars or 1))
    source_segments = []
    fallback_parts: list[str] = []
    has_explicit_document = bool((req.text or "").strip())
    segments_truncated = not has_explicit_document and len(raw_segments) > max_segments
    for raw_segment in raw_segments[:max_segments]:
        segment_text = (raw_segment.text or raw_segment.original_text or "").strip()
        if not segment_text:
            continue
        source_segments.append({
            "text": segment_text,
            "source_lang": normalize_source_language(raw_segment.source_lang, source_lang),
        })
        fallback_parts.append(segment_text)

    document_text = (req.text or "").strip() or "\n".join(fallback_parts).strip()
    input_chars = len(document_text)
    chars_truncated = input_chars > max_chars
    if chars_truncated:
        document_text = document_text[:max_chars].rstrip()

    detected_sources = {
        item["source_lang"]
        for item in source_segments
        if item["source_lang"] != "auto"
    }
    effective_source_lang = source_lang
    if (
        source_lang == "auto"
        and len(detected_sources) == 1
        and source_segments
        and all(item["source_lang"] != "auto" for item in source_segments)
    ):
        effective_source_lang = next(iter(detected_sources))

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            mt_engine.translate_document,
            document_text,
            effective_source_lang,
            target_lang,
            req.glossary,
        )
    except MTError as exc:
        logger.warning("Transcript translation failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"error": "mt_error", "message": str(exc)},
        )

    return {
        **result,
        "source_lang": effective_source_lang,
        "target_lang": target_lang,
        "input_segments": len(source_segments),
        "input_chars": len(document_text),
        "truncated": segments_truncated or chars_truncated or bool(result.get("context_truncated")),
        "limits": {
            "max_segments": max_segments,
            "max_chars": max_chars,
            "max_output_tokens": settings.mt_full_context_max_output_tokens,
        },
    }


@app.post("/api/subtitles/clear")
async def clear_subtitles():
    await subtitle_hub.clear()
    await transcript_store.clear_memory()
    return {"ok": True}


@app.get("/api/transcripts/recent")
async def recent_transcripts(limit: int = 80):
    items = await transcript_store.recent(limit=limit)
    return {
        "ok": True,
        "count": len(items),
        "items": items,
        "history": transcript_store.status(),
    }


@app.post("/api/transcripts/clear")
async def clear_transcripts():
    result = await transcript_store.clear_all()
    return result


class TranscribeRequest(BaseModel):
    file_path: str
    language: str | None = None

ALLOWED_AUDIO_DIRS = [PROJECT_ROOT / "audio", PROJECT_ROOT / "samples"]


@app.post("/api/transcribe")
async def transcribe_file(req: TranscribeRequest):
    resolved = (PROJECT_ROOT / req.file_path).resolve()
    if not any(resolved.is_relative_to(d.resolve()) for d in ALLOWED_AUDIO_DIRS if d.exists()):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=403, content={"error": "File path not allowed"})

    loop = asyncio.get_running_loop()

    def _do_transcribe():
        try:
            from mlx_audio.audio_io import read as read_audio

            audio, _ = read_audio(
                str(resolved),
                dtype="float32",
                sample_rate=settings.sample_rate,
                nchannels=1,
            )
        except ImportError:
            # Non-macOS source installs may still use the optional CT2 backend.
            from faster_whisper import decode_audio

            audio = decode_audio(str(resolved))
        return asr_engine.transcribe(audio, language=req.language)

    text = await loop.run_in_executor(None, _do_transcribe)
    return {"text": text}


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    authorized, close_code, _error, subprotocol = authorize_websocket(websocket)
    if not authorized:
        await websocket.close(code=close_code)
        return
    app_expiry = expiry_status()
    if app_expiry.expired:
        await websocket.accept(subprotocol=subprotocol)
        await websocket.send_json({"type": "error", "error": "expired", "message": app_expiry.message})
        await websocket.close(code=4403)
        return
    if not await _try_acquire_stream_slot():
        await websocket.accept(subprotocol=subprotocol)
        await websocket.send_json({
            "type": "error",
            "error": "too_many_sessions",
            "message": "已有同传会话在运行，请稍后再试。",
        })
        await websocket.close(code=1013)
        return
    try:
        session = StreamSession(websocket, subprotocol=subprotocol)
        await session.run()
    finally:
        await _release_stream_slot()


@app.websocket("/ws/subtitles")
async def ws_subtitles(websocket: WebSocket):
    authorized, close_code, _error, subprotocol = authorize_websocket(websocket)
    if not authorized:
        await websocket.close(code=close_code)
        return
    app_expiry = expiry_status()
    if app_expiry.expired:
        await websocket.accept(subprotocol=subprotocol)
        await websocket.send_json({"type": "error", "error": "expired", "message": app_expiry.message})
        await websocket.close(code=4403)
        return
    await websocket.accept(subprotocol=subprotocol)
    queue = await subtitle_hub.subscribe()
    try:
        await websocket.send_json(await subtitle_hub.snapshot())
        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=25)
            except asyncio.TimeoutError:
                payload = {"type": "heartbeat"}
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Subtitle websocket failed")
    finally:
        await subtitle_hub.unsubscribe(queue)


if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")

    _WEB_ROOT = WEB_DIR.resolve()

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        index = _WEB_ROOT / "index.html"
        try:
            candidate = (_WEB_ROOT / full_path).resolve()
            candidate.relative_to(_WEB_ROOT)
        except (ValueError, OSError):
            return FileResponse(str(index))
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(index))

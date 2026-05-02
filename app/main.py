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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from pydantic import BaseModel

from app.asr.engine import asr_engine
from app.mt.engine import MTError, mt_engine
from app.core.session import StreamSession
from app.core.config import PROJECT_ROOT, settings
from app.core.expiry import expiry_status
from app.core.model_download import download_asr_model, download_mt_model
from app.core.subtitle_hub import subtitle_hub

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# 打包后的 app 没有控制台，将日志写入文件便于排查问题
_log_stream = None
_file_handler: logging.Handler | None = None
if getattr(sys, "frozen", False) and sys.platform == "darwin":
    log_dir = Path.home() / "Library" / "Logs" / "TransLive"
    log_dir.mkdir(parents=True, exist_ok=True)
    # 手动打开文件并用 StreamHandler（line-buffered）确保每条日志立即落盘
    _log_stream = open(log_dir / "app.log", "a", encoding="utf-8", buffering=1)
    _file_handler = logging.StreamHandler(_log_stream)
    _file_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(_file_handler)
    logging.getLogger().info("TransLive %s starting (frozen, log to %s)", settings.app_version, log_dir)

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent.parent / "web"


def _close_file_log_handler() -> None:
    global _file_handler, _log_stream
    root_logger = logging.getLogger()
    if _file_handler is not None:
        try:
            root_logger.removeHandler(_file_handler)
            _file_handler.flush()
            _file_handler.close()
        except Exception:
            pass
        _file_handler = None
    if _log_stream is not None:
        try:
            _log_stream.close()
        except Exception:
            pass
        _log_stream = None


# 模型加载状态 —— 由 /api/health 暴露，前端据此显示「加载中」
_loading_state = {"asr": False, "mt": False}
_stream_session_count = 0
_stream_session_lock = asyncio.Lock()


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

        if settings.load_models_on_startup:
            # 后台加载，避免阻塞 FastAPI 启动 & /api/health 返回 404
            model_tasks = [
                asyncio.create_task(asyncio.to_thread(_load_asr_blocking)),
                asyncio.create_task(asyncio.to_thread(_load_mt_blocking)),
            ]
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
            mt_engine.close()
        except Exception:
            logger.exception("MT close failed")
        try:
            asr_engine.unload()
        except Exception:
            logger.exception("ASR unload failed")
        _close_file_log_handler()


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.middleware("http")
async def block_expired_http(request, call_next):
    app_expiry = expiry_status()
    if app_expiry.expired and request.url.path != "/api/health":
        return JSONResponse(
            status_code=403,
            content={
                "error": "expired",
                "message": app_expiry.message,
                "app_expiry_date": app_expiry.expires_on_iso,
            },
        )
    return await call_next(request)


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

    return {
        "status": "expired" if app_expiry.expired else "ok",
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "app_credit": settings.app_credit,
        "app_expiry_date": app_expiry.expires_on_iso,
        "app_expired": app_expiry.expired,
        "app_expiry_message": app_expiry.message,
        "ready": (asr_loaded and mt_loaded) and not app_expiry.expired,
        "asr_loaded": asr_loaded,
        "asr_loading": asr_loading,
        "asr_backend": asr_engine._backend_name,
        "asr_device": asr_engine._device,
        "asr_compute_type": asr_engine._compute_type,
        "asr_model_id": settings.asr_model_id,
        "asr_max_new_tokens": settings.asr_max_new_tokens,
        "mt_loaded": mt_loaded,
        "mt_loading": mt_loading,
        "mt_model_id": settings.mt_model_id,
        "mt_device": mt_engine._device,
        "mt_acceleration": mt_engine.acceleration_info,
        "mt_loaded_pairs": mt_engine.loaded_pairs,
        "mt_unavailable_reason": mt_engine.unavailable_reason,
        "auto_download_models": settings.auto_download_models,
        "load_models_on_startup": settings.load_models_on_startup,
        "desktop_mode": settings.desktop_mode,
        "max_stream_sessions": settings.max_stream_sessions,
        "active_stream_sessions": _stream_session_count,
        "websocket_max_message_bytes": settings.websocket_max_message_bytes,
        "source_lang": settings.source_lang,
        "target_lang": settings.target_lang,
        "vad": vad_info,
        "glossary_entries": len(settings.mt_glossary or {}),
        "perf_stats": perf_stats,
    }


class ModelDownloadRequest(BaseModel):
    asr: bool = False
    mt: bool = True
    force: bool = False


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

    return await loop.run_in_executor(None, _download)


class TranslateRequest(BaseModel):
    text: str
    source_lang: str | None = None
    target_lang: str | None = None


@app.post("/api/translate")
async def translate(req: TranslateRequest):
    loop = asyncio.get_running_loop()
    try:
        translated = await loop.run_in_executor(
            None, mt_engine.translate, req.text, req.source_lang, req.target_lang
        )
    except MTError as exc:
        logger.warning("REST translation failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"error": "mt_error", "message": str(exc), "original": req.text},
        )
    return {"original": req.text, "translated": translated}


@app.post("/api/subtitles/clear")
async def clear_subtitles():
    await subtitle_hub.clear()
    return {"ok": True}


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

    import numpy as np

    loop = asyncio.get_running_loop()

    def _do_transcribe():
        from faster_whisper import decode_audio
        audio = decode_audio(str(resolved))
        return asr_engine.transcribe(audio, language=req.language)

    text = await loop.run_in_executor(None, _do_transcribe)
    return {"text": text}


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    app_expiry = expiry_status()
    if app_expiry.expired:
        await websocket.accept()
        await websocket.send_json({"type": "error", "error": "expired", "message": app_expiry.message})
        await websocket.close(code=4403)
        return
    if not await _try_acquire_stream_slot():
        await websocket.accept()
        await websocket.send_json({
            "type": "error",
            "error": "too_many_sessions",
            "message": "已有同传会话在运行，请稍后再试。",
        })
        await websocket.close(code=1013)
        return
    try:
        session = StreamSession(websocket)
        await session.run()
    finally:
        await _release_stream_slot()


@app.websocket("/ws/subtitles")
async def ws_subtitles(websocket: WebSocket):
    app_expiry = expiry_status()
    if app_expiry.expired:
        await websocket.accept()
        await websocket.send_json({"type": "error", "error": "expired", "message": app_expiry.message})
        await websocket.close(code=4403)
        return
    await websocket.accept()
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

"""Generic cloud inference client.

The server contract is intentionally simple JSON so a future TransLive service,
OpenAI-compatible adapter, Riva/NIM gateway, or private FastAPI service can sit
behind the same interface.
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from typing import Generator

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class CloudInferenceError(RuntimeError):
    """Raised when a remote inference request fails."""


def _base_url() -> str:
    if not settings.cloud_enabled:
        raise CloudInferenceError("cloud inference is disabled")
    if not settings.cloud_base_url:
        raise CloudInferenceError("TRANS_CLOUD_BASE_URL is not configured")
    return settings.cloud_base_url.rstrip("/")


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if settings.cloud_api_key:
        headers["Authorization"] = f"Bearer {settings.cloud_api_key}"
    if settings.cloud_headers:
        headers.update(settings.cloud_headers)
    return headers


def _post_json(endpoint: str, payload: dict, *, timeout_s: float | None = None) -> dict:
    url = _base_url() + "/" + endpoint.lstrip("/")
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s or settings.cloud_timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise CloudInferenceError(f"cloud request failed: HTTP {exc.code} {detail}") from exc
    except Exception as exc:
        raise CloudInferenceError(f"cloud request failed: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CloudInferenceError("cloud response is not valid JSON") from exc
    if isinstance(data, dict) and data.get("error"):
        raise CloudInferenceError(str(data.get("message") or data.get("error")))
    if not isinstance(data, dict):
        raise CloudInferenceError("cloud response must be a JSON object")
    return data


def transcribe(audio: np.ndarray, language: str | None = None) -> dict:
    clipped = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2", copy=False).tobytes()
    payload = {
        "audio_pcm16_b64": base64.b64encode(pcm16).decode("ascii"),
        "sample_rate": settings.sample_rate,
        "language": language,
        "model": settings.cloud_asr_model,
        "provider": settings.cloud_provider,
    }
    return _post_json(settings.cloud_asr_endpoint, payload)


def translate(
    text: str,
    source_lang: str | None,
    target_lang: str | None,
    glossary: dict[str, str] | None = None,
) -> str:
    payload = {
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "glossary": glossary or {},
        "model": settings.cloud_mt_model,
        "provider": settings.cloud_provider,
        "stream": False,
    }
    data = _post_json(settings.cloud_mt_endpoint, payload)
    translated = data.get("translated") or data.get("text") or data.get("translation")
    if not isinstance(translated, str):
        raise CloudInferenceError("cloud MT response missing translated text")
    return translated


def translate_stream(
    text: str,
    source_lang: str | None,
    target_lang: str | None,
    glossary: dict[str, str] | None = None,
) -> Generator[str, None, None]:
    if not settings.cloud_mt_stream_endpoint:
        yield translate(text, source_lang, target_lang, glossary)
        return
    url = _base_url() + "/" + settings.cloud_mt_stream_endpoint.lstrip("/")
    payload = {
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "glossary": glossary or {},
        "model": settings.cloud_mt_model,
        "provider": settings.cloud_provider,
        "stream": True,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = _headers()
    headers["Accept"] = "application/x-ndjson"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=settings.cloud_timeout_s) as resp:
            for line in resp:
                if not line.strip():
                    continue
                data = json.loads(line.decode("utf-8"))
                if data.get("error"):
                    raise CloudInferenceError(str(data.get("message") or data.get("error")))
                token = data.get("token") or data.get("text")
                if token:
                    yield str(token)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            logger.info("Cloud MT stream endpoint unavailable; falling back to non-stream translate")
            yield translate(text, source_lang, target_lang, glossary)
            return
        detail = exc.read().decode("utf-8", errors="replace")
        raise CloudInferenceError(f"cloud stream failed: HTTP {exc.code} {detail}") from exc

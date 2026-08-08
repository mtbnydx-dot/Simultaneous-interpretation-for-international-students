import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.core.config import settings
from app.core.subtitle_hub import subtitle_hub
from app.main import app, mt_engine


TOKEN = "test-local-api-token"
PROTOCOLS = ["translive", f"translive-auth.{TOKEN}"]


@pytest.fixture(autouse=True)
def _defer_real_model_loading(monkeypatch):
    monkeypatch.setattr(settings, "load_models_on_startup", False)


def _seed_subtitle_history():
    async def _seed():
        await subtitle_hub.clear()
        await subtitle_hub.publish({
            "type": "translated",
            "segment_id": 1,
            "text": "private transcript",
        })

    asyncio.run(_seed())


def _headers(**extra):
    return {"X-TransLive-Token": TOKEN, **extra}


def test_ready_is_minimal_but_health_requires_token(monkeypatch):
    monkeypatch.setattr(settings, "local_api_token", TOKEN)
    monkeypatch.setattr(settings, "instance_id", "test-instance")

    with TestClient(app) as client:
        ready = client.get("/api/ready")
        denied = client.get("/api/health")
        allowed = client.get("/api/health", headers=_headers())

    assert ready.status_code == 200
    assert ready.json() == {"status": "ok", "instance_id": "test-instance"}
    assert denied.status_code == 401
    assert allowed.status_code == 200


def test_cross_site_clear_cannot_delete_history(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(settings, "local_api_token", TOKEN)
    monkeypatch.setattr(settings, "transcript_history_enabled", True)
    monkeypatch.setattr(settings, "transcript_history_dir", str(tmp_path))
    marker = tmp_path / "2026-08-07.jsonl"
    marker.write_text("{}\n", encoding="utf-8")

    with TestClient(app) as client:
        denied = client.post(
            "/api/transcripts/clear",
            headers={"Origin": "http://evil.example"},
        )

    assert denied.status_code == 401
    assert marker.exists()


def test_subtitle_websocket_rejects_missing_token_and_hostile_origin(monkeypatch):
    monkeypatch.setattr(settings, "local_api_token", TOKEN)

    with TestClient(app) as client:
        try:
            with client.websocket_connect("/ws/subtitles"):
                raise AssertionError("missing-token socket was accepted")
        except WebSocketDisconnect as exc:
            assert exc.code == 4401

        try:
            with client.websocket_connect(
                "/ws/subtitles",
                subprotocols=PROTOCOLS,
                headers={"Origin": "http://evil.example"},
            ):
                raise AssertionError("hostile-origin socket was accepted")
        except WebSocketDisconnect as exc:
            assert exc.code == 4403


def test_authorized_subtitle_websocket_keeps_legitimate_behavior(monkeypatch):
    monkeypatch.setattr(settings, "local_api_token", TOKEN)
    _seed_subtitle_history()

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/subtitles",
            subprotocols=PROTOCOLS,
            headers={"Origin": "http://testserver"},
        ) as socket:
            payload = json.loads(socket.receive_text())

    assert payload["type"] == "snapshot"
    assert payload["segments"][0]["translated_text"] == "private transcript"


def test_fulltext_endpoint_translates_explicit_document_once(monkeypatch):
    monkeypatch.setattr(settings, "local_api_token", TOKEN)
    captured = {}

    def _translate_document(text, source_lang, target_lang, glossary):
        captured.update({
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "glossary": glossary,
        })
        return {
            "full_original_text": text,
            "full_translated_text": "第一句。第二句。",
            "warning": None,
            "context_truncated": False,
        }

    monkeypatch.setattr(mt_engine, "translate_document", _translate_document)
    payload = {
        "text": "First sentence.\nSecond sentence.",
        "segments": [
            {"segment_id": 1, "text": "First sentence.", "source_lang": "en"},
            {"segment_id": 2, "text": "Second sentence.", "source_lang": "en"},
        ],
        "source_lang": "en",
        "target_lang": "zh",
        "glossary": {"sentence": "句子"},
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/translate/transcript",
            headers=_headers(),
            json=payload,
        )

    assert response.status_code == 200
    assert captured["text"] == payload["text"]
    assert captured["source_lang"] == "en"
    assert captured["target_lang"] == "zh"
    assert captured["glossary"] == {"sentence": "句子"}
    assert response.json()["full_translated_text"] == "第一句。第二句。"
    assert "items" not in response.json()

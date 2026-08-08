import asyncio
import json

from app.desktop.native_audio import NativeSystemAudioBridge
from desktop_launcher import DesktopApi


class _ExistingOverlay:
    def __init__(self):
        self.on_top = False
        self.loaded_url = None
        self.shown = False

    def load_url(self, url):
        self.loaded_url = url

    def resize(self, _width, _height):
        return None

    def show(self):
        self.shown = True


def test_reopening_existing_overlay_does_not_deadlock(monkeypatch):
    api = DesktopApi("http://127.0.0.1:8766", "test-token")
    overlay = _ExistingOverlay()
    api._overlay_window = overlay
    configured = []
    monkeypatch.setattr(api, "_configure_overlay_window", lambda: configured.append(True))
    monkeypatch.setattr(api, "_resize_window", lambda *_args: None)

    result = api.open_subtitle_window("translation", {"width": 900, "height": 240})

    assert result["ok"] is True
    assert result["reused"] is True
    assert overlay.shown is True
    assert configured == [True]
    assert "mode=translation" in overlay.loaded_url


def test_native_audio_forwards_streaming_preview_config():
    class _WebSocket:
        def __init__(self):
            self.messages = []

        async def send(self, value):
            self.messages.append(json.loads(value))

    bridge = NativeSystemAudioBridge("http://127.0.0.1:8766", "test-token")
    websocket = _WebSocket()

    asyncio.run(bridge._send_session_config(websocket, {
        "source_lang": "en",
        "target_lang": "zh",
        "streaming_preview": True,
    }))

    assert websocket.messages[0]["streaming_preview"] is True

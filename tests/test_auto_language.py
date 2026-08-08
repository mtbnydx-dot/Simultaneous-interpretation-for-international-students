"""
源语言"自动检测"。

Qwen3-ASR 自带语种识别，可以应付中英混说的场合。它返回的是语言**名称**
且包在列表里（如 ['English'] / ['Chinese']），不是 ISO 码，所以必须做反查 —
直接把 ['English'] 当语言码传给 MT，prompt 里就会出现看不懂的语言名。
"""

import pytest
import asyncio
import json

from app.core.languages import (
    AUTO_LANGUAGE_CODE,
    is_supported_language,
    is_supported_source_language,
    language_code_from_name,
    normalize_source_language,
    supported_language_options,
    supported_source_language_options,
)


@pytest.mark.parametrize("value, expected", [
    (["English"], "en"),          # Qwen3-ASR 的实际返回形态
    (["Chinese"], "zh"),
    (["Japanese"], "ja"),
    (["Korean"], "ko"),
    ("English", "en"),            # 裸字符串
    ("english", "en"),            # 大小写不敏感
    (["Mandarin"], "zh"),         # 常见别名
    (["Cantonese"], "zh"),
    ("中文", "zh"),                # 中文名
    ("en", "en"),                 # 已经是 ISO 码
])
def test_detected_language_names_map_to_codes(value, expected):
    assert language_code_from_name(value, "zh") == expected


@pytest.mark.parametrize("value", [None, [], "", ["Klingon"], "not-a-language"])
def test_unknown_detection_falls_back(value):
    assert language_code_from_name(value, "zh") == "zh"


def test_auto_is_a_valid_source_but_not_a_valid_target():
    assert normalize_source_language("auto") == AUTO_LANGUAGE_CODE
    assert is_supported_source_language("auto") is True
    # 目标语言必须确定，不能是 auto
    assert is_supported_language("auto") is False


def test_invalid_source_still_falls_back():
    assert normalize_source_language("xx", "en") == "en"


def test_auto_is_first_in_source_options_and_absent_from_target_options():
    source = supported_source_language_options()
    target = supported_language_options()

    assert source[0]["code"] == AUTO_LANGUAGE_CODE
    assert len(source) == len(target) + 1
    assert AUTO_LANGUAGE_CODE not in {item["code"] for item in target}


def test_native_system_audio_preserves_auto_source_language():
    from app.desktop.native_audio import NativeSystemAudioBridge

    class _Socket:
        def __init__(self):
            self.messages = []

        async def send(self, payload):
            self.messages.append(json.loads(payload))

    socket = _Socket()
    bridge = NativeSystemAudioBridge("http://127.0.0.1:8766", "test-token")
    asyncio.run(bridge._send_session_config(socket, {
        "source_lang": "auto",
        "target_lang": "zh",
    }))

    assert socket.messages[0]["source_lang"] == "auto"

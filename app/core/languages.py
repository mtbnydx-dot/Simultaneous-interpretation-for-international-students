from __future__ import annotations

from typing import Any


LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "ar": "Arabic",
    "pt": "Portuguese",
    "it": "Italian",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "cs": "Czech",
    "nl": "Dutch",
}

LANGUAGE_NAMES_ZH: dict[str, str] = {
    "en": "英语",
    "zh": "中文",
    "ja": "日语",
    "ko": "韩语",
    "de": "德语",
    "fr": "法语",
    "es": "西班牙语",
    "ru": "俄语",
    "ar": "阿拉伯语",
    "pt": "葡萄牙语",
    "it": "意大利语",
    "th": "泰语",
    "vi": "越南语",
    "id": "印尼语",
    "hi": "印地语",
    "tr": "土耳其语",
    "pl": "波兰语",
    "cs": "捷克语",
    "nl": "荷兰语",
}

LANGUAGE_NATIVE_LABELS: dict[str, str] = {
    "en": "English",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
    "ru": "Русский",
    "ar": "العربية",
    "pt": "Português",
    "it": "Italiano",
    "th": "ไทย",
    "vi": "Tiếng Việt",
    "id": "Bahasa Indonesia",
    "hi": "हिन्दी",
    "tr": "Türkçe",
    "pl": "Polski",
    "cs": "Čeština",
    "nl": "Nederlands",
}

SUPPORTED_LANGUAGE_CODES = frozenset(LANGUAGE_NAMES)


def _canonical_code(value: Any) -> str:
    code = str(value or "").strip().lower().replace("_", "-")
    return code.split("-", 1)[0] if "-" in code else code


# 源语言可以选"自动检测"：Qwen3-ASR 自带语种识别，能应付中英混说的场合。
# 只对源语言有意义 —— 目标语言必须是确定的。
AUTO_LANGUAGE_CODE = "auto"

# Qwen3-ASR 返回的是语言**名称**（如 ['English'] / ['Chinese']），不是 ISO 码，
# 这里做反查。同时收下中文名，容错一些。
_NAME_TO_CODE: dict[str, str] = {}
for _code, _name in LANGUAGE_NAMES.items():
    _NAME_TO_CODE[_name.casefold()] = _code
for _code, _name in LANGUAGE_NAMES_ZH.items():
    _NAME_TO_CODE.setdefault(_name.casefold(), _code)
_NAME_TO_CODE.update({
    "mandarin": "zh",
    "mandarin chinese": "zh",
    "simplified chinese": "zh",
    "traditional chinese": "zh",
    "cantonese": "zh",
})


def normalize_language_code(value: Any, fallback: str = "en") -> str:
    code = _canonical_code(value)
    if code in SUPPORTED_LANGUAGE_CODES:
        return code
    fallback_code = _canonical_code(fallback)
    return fallback_code if fallback_code in SUPPORTED_LANGUAGE_CODES else "en"


def normalize_source_language(value: Any, fallback: str = "en") -> str:
    """和 normalize_language_code 一样，但额外接受 'auto'。"""
    if _canonical_code(value) == AUTO_LANGUAGE_CODE:
        return AUTO_LANGUAGE_CODE
    return normalize_language_code(value, fallback)


def is_supported_language(value: Any) -> bool:
    return _canonical_code(value) in SUPPORTED_LANGUAGE_CODES


def is_supported_source_language(value: Any) -> bool:
    return _canonical_code(value) == AUTO_LANGUAGE_CODE or is_supported_language(value)


def language_code_from_name(value: Any, fallback: str = "en") -> str:
    """
    把 ASR 返回的语言名称映射回 ISO 码。

    Qwen3-ASR 的 STTOutput.language 是一个列表（如 ['English']），
    所以这里也接受列表，取第一个非空项。
    """
    if isinstance(value, (list, tuple)):
        value = next((item for item in value if item), None)
    text = str(value or "").strip()
    if not text:
        return normalize_language_code(fallback)
    # 已经是 ISO 码就直接用
    code = _canonical_code(text)
    if code in SUPPORTED_LANGUAGE_CODES:
        return code
    mapped = _NAME_TO_CODE.get(text.casefold())
    return mapped if mapped else normalize_language_code(fallback)


def language_name(code: Any) -> str:
    normalized = normalize_language_code(code)
    return LANGUAGE_NAMES[normalized]


def language_name_zh(code: Any) -> str:
    normalized = normalize_language_code(code)
    return LANGUAGE_NAMES_ZH[normalized]


def language_native_label(code: Any) -> str:
    normalized = normalize_language_code(code)
    return LANGUAGE_NATIVE_LABELS[normalized]


def supported_language_options() -> list[dict[str, str]]:
    return [
        {
            "code": code,
            "label": LANGUAGE_NATIVE_LABELS[code],
            "english": LANGUAGE_NAMES[code],
            "chinese": LANGUAGE_NAMES_ZH[code],
        }
        for code in LANGUAGE_NAMES
    ]


def supported_source_language_options() -> list[dict[str, str]]:
    """源语言列表，开头多一个"自动检测"。"""
    return [
        {
            "code": AUTO_LANGUAGE_CODE,
            "label": "自动检测",
            "english": "Auto-detect",
            "chinese": "自动检测",
        },
        *supported_language_options(),
    ]

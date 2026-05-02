from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


BUILD_CONFIG_FILENAME = "translive_build_config.json"
_UNSET = object()
_BUILD_CONFIG_CACHE: dict[str, Any] | None = None
_EXPIRY_RAW: str | None | object = _UNSET
_EXPIRY_DATE: date | None = None
_EXPIRY_CONFIG_ERROR: str | None = None


class TrialExpiredError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExpiryStatus:
    configured: bool
    expired: bool
    expires_on: date | None
    message: str | None

    @property
    def expires_on_iso(self) -> str | None:
        return self.expires_on.isoformat() if self.expires_on else None


def _resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[2]


def _read_build_config() -> dict[str, Any]:
    global _BUILD_CONFIG_CACHE
    if _BUILD_CONFIG_CACHE is not None:
        return _BUILD_CONFIG_CACHE

    config_path = _resource_root() / BUILD_CONFIG_FILENAME
    if not config_path.is_file():
        _BUILD_CONFIG_CACHE = {}
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        _BUILD_CONFIG_CACHE = data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        _BUILD_CONFIG_CACHE = {}
    return _BUILD_CONFIG_CACHE


def embedded_expiry_date() -> str | None:
    raw = _read_build_config().get("app_expiry_date")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def apply_build_config_environment() -> None:
    expiry = embedded_expiry_date()
    if expiry:
        os.environ["TRANS_APP_EXPIRY_DATE"] = expiry


def configured_expiry_date() -> str | None:
    _load_expiry_config()
    return _EXPIRY_RAW if isinstance(_EXPIRY_RAW, str) else None


def _load_expiry_config() -> None:
    global _EXPIRY_RAW, _EXPIRY_DATE, _EXPIRY_CONFIG_ERROR
    if _EXPIRY_RAW is not _UNSET:
        return

    # Embedded build config wins so packaged trial builds cannot be extended by
    # changing the environment before launch.
    embedded = embedded_expiry_date()
    if embedded:
        raw = embedded
    else:
        raw = os.environ.get("TRANS_APP_EXPIRY_DATE", "").strip() or None

    _EXPIRY_RAW = raw
    if not raw:
        return

    try:
        _EXPIRY_DATE = date.fromisoformat(raw)
    except ValueError:
        _EXPIRY_CONFIG_ERROR = "TransLive 测试版有效期配置无效，请联系作者获取新版。"


def expiry_status(today: date | None = None) -> ExpiryStatus:
    _load_expiry_config()
    if not isinstance(_EXPIRY_RAW, str):
        return ExpiryStatus(configured=False, expired=False, expires_on=None, message=None)

    if _EXPIRY_CONFIG_ERROR:
        return ExpiryStatus(
            configured=True,
            expired=True,
            expires_on=None,
            message=_EXPIRY_CONFIG_ERROR,
        )

    current = today or date.today()
    expires_on = _EXPIRY_DATE
    if expires_on is None:
        return ExpiryStatus(configured=False, expired=False, expires_on=None, message=None)
    expired = current > expires_on
    message = None
    if expired:
        message = f"TransLive 测试版已于 {expires_on.isoformat()} 到期，请联系作者获取新版。"
    return ExpiryStatus(
        configured=True,
        expired=expired,
        expires_on=expires_on,
        message=message,
    )


def enforce_not_expired() -> None:
    status = expiry_status()
    if status.expired:
        raise TrialExpiredError(status.message or "TransLive 测试版已到期。")

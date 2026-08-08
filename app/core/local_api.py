from __future__ import annotations

import secrets
from urllib.parse import urlparse

from fastapi import Request, WebSocket

from app.core.config import settings

TOKEN_HEADER = "X-TransLive-Token"
BASE_WEBSOCKET_PROTOCOL = "translive"
TOKEN_PROTOCOL_PREFIX = "translive-auth."
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "testserver"}


def websocket_protocols(token: str | None = None) -> list[str]:
    value = token or settings.local_api_token
    return [BASE_WEBSOCKET_PROTOCOL, f"{TOKEN_PROTOCOL_PREFIX}{value}"]


def _token_matches(candidate: str | None) -> bool:
    expected = settings.local_api_token or ""
    return bool(candidate and expected) and secrets.compare_digest(candidate, expected)


def _hostname(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    return (parsed.hostname or "").casefold()


def _is_local_host(value: str | None) -> bool:
    return _hostname(value) in _LOCAL_HOSTS


def _origin_allowed(origin: str | None, host: str | None) -> bool:
    if not origin:
        return True
    origin_host = _hostname(origin)
    host_name = _hostname(host)
    return bool(origin_host and host_name and origin_host == host_name and _is_local_host(host_name))


def authorize_http(request: Request) -> tuple[bool, int, str]:
    if not _is_local_host(request.headers.get("host")):
        return False, 403, "invalid_host"
    if not _token_matches(request.headers.get(TOKEN_HEADER)):
        return False, 401, "invalid_token"
    if not _origin_allowed(request.headers.get("origin"), request.headers.get("host")):
        return False, 403, "invalid_origin"
    return True, 200, "ok"


def authorize_websocket(websocket: WebSocket) -> tuple[bool, int, str, str | None]:
    protocols = [
        item.strip()
        for item in (websocket.headers.get("sec-websocket-protocol") or "").split(",")
        if item.strip()
    ]
    supplied_token = next(
        (item[len(TOKEN_PROTOCOL_PREFIX):] for item in protocols if item.startswith(TOKEN_PROTOCOL_PREFIX)),
        None,
    )
    if not _is_local_host(websocket.headers.get("host")):
        return False, 4403, "invalid_host", None
    if not _token_matches(supplied_token):
        return False, 4401, "invalid_token", None
    if not _origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host")):
        return False, 4403, "invalid_origin", None
    selected = BASE_WEBSOCKET_PROTOCOL if BASE_WEBSOCKET_PROTOCOL in protocols else None
    return True, 1000, "ok", selected

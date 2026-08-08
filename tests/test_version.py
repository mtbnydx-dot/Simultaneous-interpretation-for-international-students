from pathlib import Path

from app.core.config import Settings
from app.version import APP_CREDIT, APP_NAME, APP_VERSION, APP_VERSION_NUMERIC


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_metadata_uses_single_version_source() -> None:
    assert Settings.model_fields["app_name"].default == APP_NAME
    assert Settings.model_fields["app_version"].default == APP_VERSION
    assert Settings.model_fields["app_credit"].default == APP_CREDIT
    assert APP_VERSION_NUMERIC == "1.1"


def test_html_fallbacks_do_not_embed_release_number() -> None:
    assert APP_VERSION not in (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert APP_VERSION not in (ROOT / "web" / "overlay.html").read_text(encoding="utf-8")

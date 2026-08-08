from pathlib import Path

import pytest

from app.core import model_download
from app.core.model_download import ModelIntegrityError, sha256_file, validate_gguf_file


def test_validate_gguf_accepts_valid_header(tmp_path: Path):
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 32)

    assert validate_gguf_file(model, min_size_bytes=4) == model


def test_validate_gguf_rejects_wrong_header(tmp_path: Path):
    model = tmp_path / "broken.gguf"
    model.write_bytes(b"NOTG" + b"\0" * 32)

    with pytest.raises(ModelIntegrityError):
        validate_gguf_file(model, min_size_bytes=4)


def test_validate_gguf_rejects_tiny_file(tmp_path: Path):
    model = tmp_path / "partial.gguf"
    model.write_bytes(b"GGUF")

    with pytest.raises(ModelIntegrityError):
        validate_gguf_file(model, min_size_bytes=8)


def test_validate_gguf_enforces_expected_sha256(tmp_path: Path):
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 32)
    expected = sha256_file(model)

    assert validate_gguf_file(model, min_size_bytes=4, expected_sha256=expected) == model
    with pytest.raises(ModelIntegrityError, match="SHA-256"):
        validate_gguf_file(model, min_size_bytes=4, expected_sha256="0" * 64)


def test_validate_gguf_reuses_hash_only_while_file_is_unchanged(tmp_path: Path, monkeypatch):
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 32)
    expected = sha256_file(model)
    original = model_download.sha256_file
    calls = 0

    def counting_hash(path, chunk_size=4 * 1024 * 1024):
        nonlocal calls
        calls += 1
        return original(path, chunk_size)

    monkeypatch.setattr(model_download, "sha256_file", counting_hash)
    validate_gguf_file(model, min_size_bytes=4, expected_sha256=expected)
    validate_gguf_file(
        model,
        min_size_bytes=4,
        expected_sha256=expected,
        allow_cached_hash=True,
    )
    assert calls == 1

    model.write_bytes(b"GGUF" + b"1" * 32)
    with pytest.raises(ModelIntegrityError, match="SHA-256"):
        validate_gguf_file(
            model,
            min_size_bytes=4,
            expected_sha256=expected,
            allow_cached_hash=True,
        )
    assert calls == 2

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv is required to regenerate requirements-macos-arm64.lock." >&2
    exit 1
fi

PYTHON="${TRANS_LOCK_PYTHON:-$ROOT_DIR/.venv-macos/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] Python runtime not found: $PYTHON" >&2
    exit 1
fi

uv pip compile requirements.txt requirements-app.txt \
    --python "$PYTHON" \
    --generate-hashes \
    --no-annotate \
    --custom-compile-command './scripts/lock_macos_dependencies.sh' \
    --output-file requirements-macos-arm64.lock

echo "[OK] Updated requirements-macos-arm64.lock"

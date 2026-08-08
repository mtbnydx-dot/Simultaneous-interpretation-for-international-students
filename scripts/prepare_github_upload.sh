#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${TRANS_GITHUB_UPLOAD_DIR:-$ROOT_DIR/github_upload/TransLive-source-current}"

if [ -z "$OUT_DIR" ] || [ "$OUT_DIR" = "/" ] || [ "$OUT_DIR" = "$ROOT_DIR" ]; then
    echo "[ERROR] Refusing unsafe output directory: $OUT_DIR" >&2
    exit 1
fi
case "$OUT_DIR/" in
    "$ROOT_DIR/"*)
        case "$OUT_DIR/" in
            "$ROOT_DIR/github_upload/"*) ;;
            *)
                echo "[ERROR] Output inside the repository must be under github_upload/: $OUT_DIR" >&2
                exit 1
                ;;
        esac
        ;;
esac

if ! command -v rsync >/dev/null 2>&1; then
    echo "[ERROR] rsync is required to prepare a clean source upload." >&2
    exit 1
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

rsync -a "$ROOT_DIR/" "$OUT_DIR/" \
    --include '.env.example' \
    --exclude '.git/' \
    --exclude '.DS_Store' \
    --exclude '.env*' \
    --exclude '.venv/' \
    --exclude '.venv-macos/' \
    --exclude '.venv-macos-build*/' \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    --exclude '.mypy_cache/' \
    --exclude '.claude/' \
    --exclude '.vs/' \
    --exclude '.vscode/' \
    --exclude '.idea/' \
    --exclude '*.swp' \
    --exclude '*.swo' \
    --exclude 'build/' \
    --exclude 'dist/' \
    --exclude 'logs/' \
    --exclude 'models/' \
    --exclude 'github_upload/' \
    --exclude 'mac_app_changes/' \
    --exclude 'windows_legacy/.venv/' \
    --exclude 'windows_legacy/models/' \
    --exclude 'windows_legacy/tools/' \
    --exclude '*.gguf' \
    --exclude '*.safetensors' \
    --exclude '*.onnx' \
    --exclude '*.pt' \
    --exclude '*.pth' \
    --exclude '*.pem' \
    --exclude '*.key' \
    --exclude '*.p8' \
    --exclude '*.p12' \
    --exclude '*.pfx' \
    --exclude '*.cer' \
    --exclude '*.mobileprovision' \
    --exclude '*.jsonl' \
    --exclude '*.sqlite*' \
    --exclude 'pytorch_model.bin' \
    --exclude '*.zip'

"$ROOT_DIR/scripts/check_secrets.sh" "$OUT_DIR"

echo "[OK] Clean source tree prepared at: $OUT_DIR"
echo "     Review it, then upload that directory to GitHub."

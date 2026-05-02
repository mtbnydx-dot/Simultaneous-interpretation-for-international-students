#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v rg >/dev/null 2>&1; then
    echo "[WARN] ripgrep is not installed; skipping secret scan." >&2
    exit 0
fi

PATTERN='(hf_[A-Za-z0-9_=-]{20,}|TRANS_HF_TOKEN[[:space:]]*=[[:space:]]*[^#[:space:]]+)'

if rg -n --hidden \
    --glob '!.git/**' \
    --glob '!dist/**' \
    --glob '!build/**' \
    --glob '!github_upload/**' \
    --glob '!.venv*/**' \
    --glob '!venv/**' \
    --glob '!windows_legacy/**' \
    --glob '!mac_app_changes/**' \
    --glob '!.env' \
    --glob '!.env.example' \
    "$PATTERN" .; then
    echo "[ERROR] Potential secret found outside ignored env files." >&2
    exit 1
fi

echo "[OK] No obvious Hugging Face tokens found outside ignored env files."

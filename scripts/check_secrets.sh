#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAN_ROOT="${1:-$ROOT_DIR}"

if [ ! -d "$SCAN_ROOT" ]; then
    echo "[ERROR] Secret scan target does not exist: $SCAN_ROOT" >&2
    exit 1
fi
cd "$SCAN_ROOT"

if ! command -v rg >/dev/null 2>&1; then
    echo "[ERROR] ripgrep is required; refusing to skip the secret scan." >&2
    exit 1
fi

PATTERN='(hf_[A-Za-z0-9_=-]{20,}|sk-[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{20,}|-----BEGIN ([A-Z0-9 ]+ )?PRIVATE KEY-----|TRANS_(HF_TOKEN|CLOUD_API_KEY|LOCAL_API_TOKEN)[[:space:]]*=[[:space:]]*[^#[:space:]]+)'

if rg -n --hidden \
    --glob '!.git/**' \
    --glob '!dist/**' \
    --glob '!build/**' \
    --glob '!github_upload/**' \
    --glob '!.venv*/**' \
    --glob '!venv/**' \
    --glob '!windows_legacy/.venv/**' \
    --glob '!windows_legacy/models/**' \
    --glob '!mac_app_changes/**' \
    --glob '!.env*' \
    --glob '!.env.example' \
    "$PATTERN" .; then
    echo "[ERROR] Potential credential or private key found." >&2
    exit 1
fi

KEY_FILES="$(find . \
    \( \
        -path './.git' -o \
        -path './dist' -o \
        -path './build' -o \
        -path './github_upload' -o \
        -path './.venv*' -o \
        -path './venv' -o \
        -path './windows_legacy/.venv' -o \
        -path './windows_legacy/models' -o \
        -path './windows_legacy/tools' -o \
        -path './mac_app_changes' \
    \) -prune -o \
    -type f \( \
        -name '*.pem' -o -name '*.key' -o -name '*.p8' -o \
        -name '*.p12' -o -name '*.pfx' -o -name '*.cer' -o \
        -name '*.mobileprovision' -o -name 'id_rsa' -o -name 'id_ed25519' \
    \) -print)"
if [ -n "$KEY_FILES" ]; then
    echo "[ERROR] Signing/private-key material found:" >&2
    printf '%s\n' "$KEY_FILES" >&2
    exit 1
fi

echo "[OK] No obvious credentials or private-key files found in $SCAN_ROOT."

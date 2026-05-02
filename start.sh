#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if [ -x ".venv/bin/python" ]; then
        PY=".venv/bin/python"
    else
        VENV_DIR="${TRANS_VENV_DIR:-.venv-macos}"

        if [ ! -x "$VENV_DIR/bin/python" ]; then
            if command -v python3 >/dev/null 2>&1; then
                BASE_PY="python3"
            elif command -v python >/dev/null 2>&1; then
                BASE_PY="python"
            else
                echo "[错误] 未检测到 Python，请先安装 Python 3.10+。"
                echo "下载: https://www.python.org/downloads/"
                exit 1
            fi

            echo ">> 创建本机虚拟环境: $VENV_DIR"
            "$BASE_PY" -m venv "$VENV_DIR"
        fi

        PY="$VENV_DIR/bin/python"
    fi
fi

exec "$PY" run.py "$@"

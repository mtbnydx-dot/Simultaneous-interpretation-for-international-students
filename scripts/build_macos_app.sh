#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR="${TRANS_VENV_DIR:-.venv-macos}"
APP_PATH="dist/TransLive.app"
ENTITLEMENTS_FILE="${TRANS_ENTITLEMENTS_FILE:-scripts/entitlements.plist}"
SIGN_IDENTITY="${TRANS_CODESIGN_IDENTITY:-}"
NOTARY_PROFILE="${TRANS_NOTARY_PROFILE:-}"
NOTARY_ZIP="dist/TransLive-notary.zip"
DIST_ZIP="${TRANS_DIST_ZIP:-dist/TransLive-macOS-arm64.zip}"
REQUIRE_LLAMA_METAL="${TRANS_REQUIRE_LLAMA_METAL:-auto}"
REBUILD_LLAMA_METAL="${TRANS_REBUILD_LLAMA_METAL:-auto}"
MAYBE_DISTRIBUTION="${TRANS_DISTRIBUTION:-0}"
PRESERVE_DIST_ZIPS="${TRANS_PRESERVE_DIST_ZIPS:-0}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    if command -v python3 >/dev/null 2>&1; then
        python3 -m venv "$VENV_DIR"
    else
        echo "[错误] 未检测到 python3，请先安装 Python 3.10+。"
        exit 1
    fi
fi

PY="$VENV_DIR/bin/python"

is_macos() {
    [ "$(uname -s)" = "Darwin" ]
}

llama_supports_gpu_offload() {
    "$PY" - <<'PY'
try:
    import llama_cpp
    supports = getattr(llama_cpp, "llama_supports_gpu_offload", lambda: False)
    raise SystemExit(0 if bool(supports()) else 1)
except Exception:
    raise SystemExit(1)
PY
}

ensure_llama_metal() {
    if ! is_macos; then
        return
    fi

    local require="$REQUIRE_LLAMA_METAL"
    if [ "$require" = "auto" ]; then
        require="1"
    fi
    if [ "$require" = "0" ]; then
        return
    fi

    if llama_supports_gpu_offload; then
        echo "[OK] llama-cpp-python 支持 GPU offload / Metal。"
        return
    fi

    local rebuild="$REBUILD_LLAMA_METAL"
    if [ "$rebuild" = "auto" ]; then
        rebuild="1"
    fi
    if [ "$rebuild" != "1" ]; then
        echo "[错误] 当前 llama-cpp-python 不支持 Metal。"
        echo "       设置 TRANS_REBUILD_LLAMA_METAL=1 让脚本从源码重编译，或 TRANS_REQUIRE_LLAMA_METAL=0 跳过。"
        exit 1
    fi

    echo "[信息] 当前 llama-cpp-python 不支持 Metal，开始从源码重编译..."
    CMAKE_ARGS="-DGGML_METAL=on" FORCE_CMAKE=1 \
        "$PY" -m pip install --force-reinstall --no-cache-dir --no-binary llama-cpp-python "llama-cpp-python>=0.3.0"

    if ! llama_supports_gpu_offload; then
        echo "[错误] Metal 版 llama-cpp-python 校验失败。请确认已安装 Xcode Command Line Tools / cmake。"
        exit 1
    fi
    echo "[OK] Metal 版 llama-cpp-python 已就绪。"
}

sign_app() {
    if ! is_macos; then
        return
    fi
    if [ ! -d "$APP_PATH" ]; then
        echo "[错误] 未找到 $APP_PATH"
        exit 1
    fi
    if [ ! -f "$ENTITLEMENTS_FILE" ]; then
        echo "[错误] 未找到 entitlements: $ENTITLEMENTS_FILE"
        exit 1
    fi

    local identity="$SIGN_IDENTITY"
    local codesign_args=("--force" "--deep" "--options" "runtime")
    if [ -z "$identity" ]; then
        if [ "$MAYBE_DISTRIBUTION" = "1" ]; then
            echo "[错误] 正式分发需要设置 TRANS_CODESIGN_IDENTITY，例如："
            echo "       TRANS_CODESIGN_IDENTITY='Developer ID Application: Your Name (TEAMID)' TRANS_DISTRIBUTION=1 ./scripts/build_macos_app.sh"
            exit 1
        fi
        identity="-"
        echo "[警告] 未设置 TRANS_CODESIGN_IDENTITY，将使用 ad-hoc 签名，仅适合本机测试。"
    else
        codesign_args+=("--timestamp")
        echo "[信息] 使用证书签名: $identity"
    fi

    codesign_args+=("--entitlements" "$ENTITLEMENTS_FILE" "--sign" "$identity" "$APP_PATH")
    /usr/bin/codesign "${codesign_args[@]}"
    /usr/bin/codesign --verify --deep --strict --verbose=2 "$APP_PATH"
}

notarize_app() {
    if ! is_macos; then
        return
    fi
    if [ -z "$NOTARY_PROFILE" ]; then
        echo "[信息] 未设置 TRANS_NOTARY_PROFILE，跳过公证。"
        echo "       设置方式示例：xcrun notarytool store-credentials translive-notary"
        return
    fi
    if [ -z "$SIGN_IDENTITY" ]; then
        echo "[错误] 公证需要 Developer ID 签名，不能使用 ad-hoc 签名。"
        exit 1
    fi

    rm -f "$NOTARY_ZIP"
    /usr/bin/ditto -c -k --keepParent "$APP_PATH" "$NOTARY_ZIP"
    xcrun notarytool submit "$NOTARY_ZIP" --keychain-profile "$NOTARY_PROFILE" --wait
    xcrun stapler staple "$APP_PATH"
    xcrun stapler validate "$APP_PATH"
}

"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt -r requirements-app.txt
ensure_llama_metal

ZIP_BACKUP_DIR=""
if [ "$PRESERVE_DIST_ZIPS" = "1" ] && [ -d dist ]; then
    ZIP_BACKUP_DIR="$(mktemp -d)"
    find dist -maxdepth 1 -type f -name "*.zip" -exec cp {} "$ZIP_BACKUP_DIR" \;
fi

rm -rf build dist
mkdir -p dist
if [ -n "$ZIP_BACKUP_DIR" ]; then
    find "$ZIP_BACKUP_DIR" -maxdepth 1 -type f -name "*.zip" -exec cp {} dist/ \;
    rm -rf "$ZIP_BACKUP_DIR"
fi

"$PY" -m PyInstaller --clean --noconfirm TransLive.spec
sign_app
notarize_app

rm -f "$DIST_ZIP"
/usr/bin/ditto -c -k --keepParent "$APP_PATH" "$DIST_ZIP"

echo
echo "构建完成: $APP_PATH"
echo "分发压缩包: $DIST_ZIP"
echo "注意: 打包结果不包含 models/ 目录，首次启动会在界面中提示下载模型。"

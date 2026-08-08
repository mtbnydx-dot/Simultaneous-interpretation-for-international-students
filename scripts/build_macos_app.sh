#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR="${TRANS_VENV_DIR:-.venv-macos}"
LOCK_FILE="requirements-macos-arm64.lock"
LLAMA_METAL_REQUIREMENTS="requirements-llama-metal.txt"
APP_PATH="dist/TransLive.app"
APPSTORE_BUILD="${TRANS_APPSTORE:-0}"
if [ -n "${TRANS_ENTITLEMENTS_FILE:-}" ]; then
    ENTITLEMENTS_FILE="$TRANS_ENTITLEMENTS_FILE"
elif [ "$APPSTORE_BUILD" = "1" ]; then
    ENTITLEMENTS_FILE="scripts/entitlements.appstore.plist"
else
    ENTITLEMENTS_FILE="scripts/entitlements.plist"
fi
SIGN_IDENTITY="${TRANS_CODESIGN_IDENTITY:-}"
NOTARY_PROFILE="${TRANS_NOTARY_PROFILE:-}"
NOTARY_ZIP="dist/TransLive-notary.zip"
DIST_ZIP="${TRANS_DIST_ZIP:-dist/TransLive-macOS-arm64.zip}"
REQUIRE_LLAMA_METAL="${TRANS_REQUIRE_LLAMA_METAL:-auto}"
REBUILD_LLAMA_METAL="${TRANS_REBUILD_LLAMA_METAL:-auto}"
MAYBE_DISTRIBUTION="${TRANS_DISTRIBUTION:-0}"
PRESERVE_DIST_ZIPS="${TRANS_PRESERVE_DIST_ZIPS:-0}"
MACOS_MIN_VERSION="${TRANS_MACOS_MIN_VERSION:-14.0}"

if [ "$MAYBE_DISTRIBUTION" = "1" ] && [ -z "$NOTARY_PROFILE" ]; then
    echo "[错误] TRANS_DISTRIBUTION=1 必须设置 TRANS_NOTARY_PROFILE 并完成公证。" >&2
    echo "       仅签名、稍后手工公证时不要设置 TRANS_DISTRIBUTION=1。" >&2
    exit 1
fi

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

if is_macos; then
    if [ "$(uname -m)" != "arm64" ]; then
        echo "[错误] Qwen3/MLX 分发包只支持 Apple Silicon (arm64)。" >&2
        exit 1
    fi
    export MACOSX_DEPLOYMENT_TARGET="$MACOS_MIN_VERSION"
    echo "[信息] macOS 最低部署版本: $MACOSX_DEPLOYMENT_TARGET"
fi

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

llama_supports_macos_target() {
    local lib_dir item build_info min_version found=0
    lib_dir="$($PY - <<'PY'
from pathlib import Path
import llama_cpp

print(Path(llama_cpp.__file__).resolve().parent / "lib")
PY
)" || return 1

    while IFS= read -r -d '' item; do
        if ! /usr/bin/file -b "$item" | /usr/bin/grep -q 'Mach-O'; then
            continue
        fi
        found=1
        build_info="$(xcrun vtool -show-build "$item" 2>/dev/null || true)"
        min_version="$(printf '%s\n' "$build_info" | /usr/bin/awk '
            /^[[:space:]]+minos / { print $2; exit }
            /^[[:space:]]+version / { print $2; exit }
        ')"
        if [ -n "$min_version" ] && version_is_newer "$min_version" "$MACOS_MIN_VERSION"; then
            return 1
        fi
    done < <(/usr/bin/find "$lib_dir" -type f -print0 2>/dev/null)
    [ "$found" -eq 1 ]
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

    if llama_supports_gpu_offload && llama_supports_macos_target; then
        echo "[OK] llama-cpp-python 支持 Metal 且兼容 macOS ${MACOS_MIN_VERSION}。"
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

    echo "[信息] llama-cpp-python 的 Metal 或最低系统版本不合格，开始从源码重编译..."
    CMAKE_ARGS="-DGGML_METAL=on -DCMAKE_OSX_DEPLOYMENT_TARGET=$MACOS_MIN_VERSION" FORCE_CMAKE=1 \
        "$PY" -m pip install --force-reinstall --no-cache-dir --no-binary llama-cpp-python \
        --no-deps --require-hashes -r "$LLAMA_METAL_REQUIREMENTS"

    if ! llama_supports_gpu_offload || ! llama_supports_macos_target; then
        echo "[错误] Metal / macOS ${MACOS_MIN_VERSION} 版 llama-cpp-python 校验失败。请确认已安装 Xcode Command Line Tools / cmake。"
        exit 1
    fi
    echo "[OK] Metal / macOS ${MACOS_MIN_VERSION} 版 llama-cpp-python 已就绪。"
}

locked_version() {
    local package="$1"
    /usr/bin/awk -F'==' -v package="$package" \
        '$1 == package { split($2, fields, " "); print fields[1]; exit }' "$LOCK_FILE"
}

ensure_compatible_mlx_wheels() {
    if ! is_macos; then
        return
    fi

    local mlx_version mlx_metal_version platform_version py_version py_abi wheel_dir
    mlx_version="$(locked_version mlx)"
    mlx_metal_version="$(locked_version mlx-metal)"
    if [ -z "$mlx_version" ] || [ -z "$mlx_metal_version" ]; then
        echo "[错误] 无法从 $LOCK_FILE 读取 MLX 版本。" >&2
        exit 1
    fi

    platform_version="${MACOS_MIN_VERSION//./_}"
    py_version="$($PY -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
    py_abi="cp${py_version}"
    wheel_dir="$(mktemp -d "${TMPDIR:-/tmp}/translive-mlx-wheels.XXXXXX")"

    echo "[信息] 选择 macOS ${MACOS_MIN_VERSION} 兼容的 MLX wheel..."
    "$PY" -m pip download \
        --no-deps \
        --only-binary=:all: \
        --platform "macosx_${platform_version}_arm64" \
        --python-version "$py_version" \
        --implementation cp \
        --abi "$py_abi" \
        --dest "$wheel_dir" \
        "mlx==$mlx_version" "mlx-metal==$mlx_metal_version"

    "$PY" -m pip install --force-reinstall --no-deps \
        "$wheel_dir"/mlx-*.whl "$wheel_dir"/mlx_metal-*.whl
    rm -rf "$wheel_dir"
}

version_is_newer() {
    /usr/bin/awk -v actual="$1" -v target="$2" 'BEGIN {
        split(actual, a, "."); split(target, b, ".");
        for (i = 1; i <= 3; i++) {
            av = (a[i] == "" ? 0 : a[i]) + 0;
            bv = (b[i] == "" ? 0 : b[i]) + 0;
            if (av > bv) exit 0;
            if (av < bv) exit 1;
        }
        exit 1;
    }'
}

verify_bundle_macos_target() {
    if ! is_macos; then
        return
    fi

    local item build_info min_version incompatible=0 checked=0
    while IFS= read -r -d '' item; do
        if ! /usr/bin/file -b "$item" | /usr/bin/grep -q 'Mach-O'; then
            continue
        fi
        build_info="$(xcrun vtool -show-build "$item" 2>/dev/null || true)"
        min_version="$(printf '%s\n' "$build_info" | /usr/bin/awk '
            /^[[:space:]]+minos / { print $2; exit }
            /^[[:space:]]+version / { print $2; exit }
        ')"
        if [ -z "$min_version" ]; then
            continue
        fi
        checked=$((checked + 1))
        if version_is_newer "$min_version" "$MACOS_MIN_VERSION"; then
            echo "[错误] 二进制最低系统版本 $min_version 高于目标 $MACOS_MIN_VERSION: $item" >&2
            incompatible=1
        fi
    done < <(/usr/bin/find "$APP_PATH/Contents" -type f -print0)

    if [ "$checked" -eq 0 ]; then
        echo "[错误] 未在 App 中找到可检查的 Mach-O 二进制。" >&2
        exit 1
    fi
    if [ "$incompatible" -ne 0 ]; then
        echo "[错误] App 未通过 macOS 最低版本检查。" >&2
        exit 1
    fi
    echo "[OK] $checked 个 Mach-O 文件均兼容 macOS $MACOS_MIN_VERSION 或更低版本。"
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
    local codesign_args=("--force" "--options" "runtime")
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

    local main_binary="$APP_PATH/Contents/MacOS/TransLive"
    while IFS= read -r -d '' item; do
        if /usr/bin/file -b "$item" | /usr/bin/grep -q 'Mach-O'; then
            /usr/bin/codesign "${codesign_args[@]}" --sign "$identity" "$item"
        fi
    done < <(/usr/bin/find "$APP_PATH/Contents" -type f ! -path "$main_binary" -print0)

    while IFS= read -r -d '' bundle; do
        /usr/bin/codesign "${codesign_args[@]}" --sign "$identity" "$bundle"
    done < <(/usr/bin/find "$APP_PATH/Contents" -depth -type d \( \
        -name '*.framework' -o -name '*.xpc' -o -name '*.appex' \
    \) -print0)

    /usr/bin/codesign "${codesign_args[@]}" --entitlements "$ENTITLEMENTS_FILE" \
        --sign "$identity" "$main_binary"
    /usr/bin/codesign "${codesign_args[@]}" --entitlements "$ENTITLEMENTS_FILE" \
        --sign "$identity" "$APP_PATH"
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
    /usr/bin/ditto --norsrc --noextattr -c -k --keepParent "$APP_PATH" "$NOTARY_ZIP"
    xcrun notarytool submit "$NOTARY_ZIP" --keychain-profile "$NOTARY_PROFILE" --wait
    xcrun stapler staple "$APP_PATH"
    xcrun stapler validate "$APP_PATH"
    /usr/sbin/spctl --assess --type execute --verbose=4 "$APP_PATH"
}

if [ ! -f "$LOCK_FILE" ]; then
    echo "[错误] 缺少 ${LOCK_FILE}；先运行 ./scripts/lock_macos_dependencies.sh。" >&2
    exit 1
fi
"$PY" -m pip install --require-hashes -r "$LOCK_FILE"
ensure_compatible_mlx_wheels
ensure_llama_metal

ZIP_BACKUP_DIR=""
if [ "$PRESERVE_DIST_ZIPS" = "1" ] && [ -d dist ]; then
    ZIP_BACKUP_DIR="$(mktemp -d)"
    find dist -maxdepth 1 -type f -name "*.zip" -exec cp {} "$ZIP_BACKUP_DIR" \;
fi

rm -rf build
# macOS 上 Finder / Spotlight 可能在 rm 执行期间重新写入 .DS_Store，
# 于是 `rm -rf dist` 在最后 rmdir 时报 "Directory not empty" 而失败。
# 配合 set -e 会直接中断整个构建，并且此时旧产物已经被删掉了。
# 改成清空目录内容而不是删除目录本身，绕开这个竞态。
if [ -d dist ]; then
    find dist -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
fi
mkdir -p dist
if [ -n "$ZIP_BACKUP_DIR" ]; then
    find "$ZIP_BACKUP_DIR" -maxdepth 1 -type f -name "*.zip" -exec cp {} dist/ \;
    rm -rf "$ZIP_BACKUP_DIR"
fi

"$PY" -m PyInstaller --clean --noconfirm TransLive.spec
verify_bundle_macos_target
sign_app
notarize_app

rm -f "$DIST_ZIP"
/usr/bin/ditto --norsrc --noextattr -c -k --keepParent "$APP_PATH" "$DIST_ZIP"

echo
echo "构建完成: $APP_PATH"
echo "分发压缩包: $DIST_ZIP"
echo "注意: 打包结果不包含 models/ 目录，首次启动会在界面中提示下载模型。"

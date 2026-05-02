@echo off
chcp 65001 >nul
title TransLive

cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [错误] 未检测到 Python，请先安装 Python 3.10+ 并加入 PATH。
        echo 下载: https://www.python.org/downloads/
        echo.
        pause
        exit /b 1
    )
    set "PY=python"
)

"%PY%" run.py %*

echo.
echo 服务已退出。按任意键关闭此窗口。
pause >nul

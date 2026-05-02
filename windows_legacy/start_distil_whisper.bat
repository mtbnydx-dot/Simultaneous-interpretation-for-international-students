@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set NO_COLOR=1
set TERM=dumb

echo ========================================
echo TransLive - Distil-Whisper Configuration
echo ========================================
echo.
echo Starting TransLive with Distil-Whisper...
echo.
echo ASR Backend: transformers-distil
echo Model: distil-whisper/distil-medium.en
echo Device: CPU (Intel GPU not supported)
echo.
echo Server will be available at: http://localhost:8766
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 8766

pause

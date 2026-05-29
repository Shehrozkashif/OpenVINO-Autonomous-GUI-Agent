@echo off
REM scripts\setup\start_dev.bat
REM Development startup sequence for Intel AI PC (Windows 11).
REM Run this every session before launching main.py.

cd /d "%~dp0..\.."

echo === Desktop GUI Agent - Dev Startup (Windows) ===
echo.

REM 1. Check Ollama
echo =^> Checking Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Ollama not installed. Download from https://ollama.com
    pause
    exit /b 1
)

ollama list >nul 2>&1
if errorlevel 1 (
    echo   Starting ollama serve in background...
    start /b ollama serve >"%TEMP%\ollama.log" 2>&1
    timeout /t 3 /nobreak >nul
)

ollama list | findstr "qwen2.5vl:3b" >nul 2>&1
if errorlevel 1 (
    echo [WARN] qwen2.5vl:3b not pulled. Pulling now...
    ollama pull qwen2.5vl:3b
)
echo [OK] Ollama ready

REM 2. Tool server
echo.
echo =^> Starting Tool Server on port 8015...
curl -s http://127.0.0.1:8015/health >nul 2>&1
if not errorlevel 1 (
    echo [OK] Tool server already running
) else (
    start /b python -m tools.desktop_control.server >"%TEMP%\tool_server.log" 2>&1
    timeout /t 2 /nobreak >nul
    curl -s http://127.0.0.1:8015/health >nul 2>&1
    if errorlevel 1 (
        echo [FAIL] Tool server failed - check %TEMP%\tool_server.log
        pause
        exit /b 1
    )
    echo [OK] Tool server started
)

echo.
echo === All systems ready ===
echo.
echo Launch the app:  python main.py
echo Pipeline:        Direct OpenVINO (Qwen2.5-VL on Intel Arc GPU)
echo.
pause

@echo off
rem ============================================================
rem  One-click launcher for the Smartphone Statistics Dashboard
rem  Double-click this file. First run sets everything up
rem  (a few minutes); later runs start instantly.
rem ============================================================
setlocal
cd /d "%~dp0"

rem --- Use UTF-8 console so Streamlit's output never crashes on old codepages ---
chcp 65001 >nul
set PYTHONUTF8=1

rem --- Find Python (prefer the official "py" launcher) ---
where py >nul 2>nul
if %errorlevel%==0 (set "PY=py -3") else (set "PY=python")

%PY% --version >nul 2>nul
if errorlevel 1 (
    echo.
    echo   Python was not found on this computer.
    echo   Install Python 3.10 or newer from https://www.python.org/downloads/
    echo   and TICK "Add Python to PATH" in the installer.
    echo.
    pause
    exit /b 1
)

rem --- Create the virtual environment (first run only) ---
if not exist ".venv" (
    echo.
    echo   [1/3] Creating a private Python environment ^(first run only^)...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo   Failed to create the environment. Reinstall Python and try again.
        pause
        exit /b 1
    )
)

rem --- Install dependencies (uses the pip cache after the first time) ---
echo   [2/3] Installing dependencies...
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo   Dependency install failed. Check your internet connection and retry.
    pause
    exit /b 1
)

rem --- Start the dashboard ---
echo   [3/3] Starting the dashboard - it will open in your browser...
echo   (Close this window, or press Ctrl+C inside it, to stop.)
start "" http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app.py
pause

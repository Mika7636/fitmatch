@echo off
REM FitMatch -- the whole demo from one double-click: API + frontend + browser.
REM
REM   run_demo.bat            start both, wait for the API, open the browser
REM   run_demo.bat --no-open  start both, leave the browser alone
REM
REM Mirrors build_data.bat, run_tests.bat, run_api.bat and run_frontend.bat.
REM GNU make is not available on this machine and npx does not work in this
REM checkout (the path contains a comma, which cmd.exe treats as an argument
REM separator, so npm's generated .bin shims resolve wrongly). Everything
REM below is a plain python or npm-run command for that reason.
REM
REM Two child windows open and stay open -- one per server -- so their logs
REM stay readable and either can be restarted without killing the other.
REM Closing a window stops that server. This window exits once both are up.

setlocal
cd /d "%~dp0"

set OPEN_BROWSER=1
if /i "%~1"=="--no-open" set OPEN_BROWSER=0

echo.
echo ============================================================
echo  FitMatch demo
echo ============================================================

REM --- preconditions ------------------------------------------------------
if not exist "data\processed\products.csv" (
    echo.
    echo data\processed\products.csv is missing. Run build_data.bat first.
    exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo python is not on PATH.
    exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
    echo.
    echo npm is not on PATH. Node 20+ is needed for the frontend.
    exit /b 1
)

if not exist "frontend\node_modules" (
    echo.
    echo Installing frontend dependencies, one moment...
    pushd frontend
    call npm install || (popd & exit /b 1)
    popd
)

REM --- the API ------------------------------------------------------------
REM Two cmd.exe traps handled here:
REM   1. `start` reads its first quoted argument as the window title, so the
REM      title is always given explicitly even when it looks redundant.
REM   2. the working directory goes in `start /D` rather than in a `cd &&`
REM      inside the /k string. Nesting quotes inside `cmd /k "..."` is exactly
REM      where this checkout's comma-in-the-path breaks things.
echo.
echo Starting the API on http://127.0.0.1:8000 ...
start "FitMatch API" /D "%~dp0" cmd /k python -m src.api

REM Poll /api/health rather than sleeping a fixed number of seconds: a cold
REM start rebuilds the TF-IDF model and takes a second or two, a warm one
REM loads the joblib cache in about 200 ms, and the frontend shows a dead
REM backend as four separate failures if it opens too early.
echo Waiting for the recommender to load...
python "%~dp0scripts\wait_for_api.py" --timeout 90
if errorlevel 1 (
    echo.
    echo The API did not become ready. Check the "FitMatch API" window.
    exit /b 1
)

REM --- the frontend -------------------------------------------------------
echo.
echo Starting the frontend on http://localhost:5173 ...
start "FitMatch frontend" /D "%~dp0frontend" cmd /k npm run dev

if "%OPEN_BROWSER%"=="1" (
    REM Give Vite a moment to bind the port before the browser asks for it.
    timeout /t 4 /nobreak >nul
    start "" "http://localhost:5173"
)

echo.
echo ============================================================
echo  API       http://127.0.0.1:8000/docs
echo  Frontend  http://localhost:5173
echo.
echo  Both run in their own windows. Close a window to stop that
echo  server. Demo script: python scripts\seed_demo.py
echo ============================================================
echo.
endlocal

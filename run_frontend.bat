@echo off
REM FitMatch session 4 -- serve the frontend (Vite dev server).
REM Mirrors build_data.bat, run_tests.bat and run_api.bat. GNU make is not
REM available on this machine, so the .bat files are the project's commands.
REM
REM   run_frontend.bat              http://localhost:5173
REM   run_frontend.bat --host       expose it on the local network
REM
REM The backend must be running too -- start run_api.bat in another terminal
REM first. The frontend talks to http://127.0.0.1:8000 unless
REM VITE_API_BASE_URL says otherwise, and the API only accepts browser
REM requests from port 5173, so do not change the port without changing
REM DEFAULT_CORS_ORIGINS in src\api\main.py to match.

cd /d "%~dp0frontend" || exit /b 1

if not exist "node_modules" (
    echo Installing frontend dependencies, one moment...
    call npm install || exit /b 1
)

REM `npm run dev` rather than `npx vite`: this checkout's path contains a
REM comma, which cmd.exe treats as an argument separator, so npm's generated
REM .bin shims resolve to the wrong path. The package.json scripts call node
REM on the package entry points directly and sidestep it.
call npm run dev -- %*
if errorlevel 1 (
    echo.
    echo FRONTEND EXITED WITH AN ERROR.
    exit /b 1
)

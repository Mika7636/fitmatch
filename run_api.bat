@echo off
REM FitMatch session 3 -- serve the API.
REM Mirrors build_data.bat and run_tests.bat. GNU make is not available on
REM this machine, so these three .bat files are the project's commands.
REM
REM   run_api.bat                 http://127.0.0.1:8000/docs
REM   run_api.bat --port 8080     any argument goes through to python -m src.api
REM   run_api.bat --reload        restart on source changes (development)
REM
REM The recommender and its TF-IDF model load once at startup, not per
REM request. Expect a second or so the first time, then ~200 ms on later
REM starts from the joblib cache in data/cache/. GET /api/health reports
REM which of the two happened.

if not exist "data\processed\products.csv" (
    echo.
    echo data\processed\products.csv is missing. Run build_data.bat first.
    exit /b 1
)

python -m src.api %*
if errorlevel 1 (
    echo.
    echo API EXITED WITH AN ERROR.
    exit /b 1
)

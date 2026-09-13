@echo off
REM FitMatch -- run the whole test suite (data, recommender, API).
REM Mirrors build_data.bat. GNU make is not available on this machine.
REM
REM   run_tests.bat            every test
REM   run_tests.bat -k yoga    pass any pytest argument straight through

python -m pytest %*
if errorlevel 1 (
    echo.
    echo TESTS FAILED.
    exit /b 1
)
echo.
echo All tests passed.

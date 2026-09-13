@echo off
python -m src.data.build_products || exit /b 1
python -m src.data.build_users || exit /b 1
python -m src.data.build_interactions || exit /b 1
python -m src.data.validate || exit /b 1
echo Data build complete.
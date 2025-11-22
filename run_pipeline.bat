# Quick Start Script untuk Windows
# Double-click file ini untuk menjalankan pipeline lengkap

@echo off
echo ====================================
echo ProjekPID - Pipeline Runner
echo ====================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python tidak ditemukan!
    echo Silakan install Python 3.8+ terlebih dahulu
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ====================================
echo Running Pipeline...
echo ====================================
python run_pipeline.py

echo.
echo ====================================
echo Pipeline Completed!
echo ====================================
echo.
echo To view dashboard, run:
echo streamlit run dashboard/app.py
echo.
pause

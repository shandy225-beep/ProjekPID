#!/bin/bash
# Quick Start Script untuk Linux/Mac
# Jalankan: chmod +x run_pipeline.sh && ./run_pipeline.sh

echo "===================================="
echo "ProjekPID - Pipeline Runner"
echo "===================================="
echo

echo "Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python tidak ditemukan!"
    echo "Silakan install Python 3.8+ terlebih dahulu"
    exit 1
fi

echo
echo "Installing dependencies..."
pip3 install -r requirements.txt

echo
echo "===================================="
echo "Running Pipeline..."
echo "===================================="
python3 run_pipeline.py

echo
echo "===================================="
echo "Pipeline Completed!"
echo "===================================="
echo
echo "To view dashboard, run:"
echo "streamlit run dashboard/app.py"
echo

#!/bin/bash

echo "========================================"
echo "    XU LY ANH SO - DO AN CUOI KY"
echo "========================================"
echo

echo "[1/3] Kiem tra Python..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python chua duoc cai dat!"
    echo "Vui long cai dat Python tu https://python.org"
    exit 1
fi

echo
echo "[2/3] Cai dat dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Khong the cai dat dependencies!"
    echo "Vui long kiem tra ket noi internet va thu lai."
    exit 1
fi

echo
echo "[3/3] Khoi dong ung dung Streamlit..."
echo
echo "========================================"
echo "  UNG DUNG DANG CHAY TAI: "
echo "  http://localhost:8501"
echo "========================================"
echo
echo "Nhan Ctrl+C de dung ung dung"
echo

streamlit run app.py

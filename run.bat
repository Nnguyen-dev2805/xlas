@echo off
echo ========================================
echo    XU LY ANH SO - DO AN CUOI KY
echo ========================================
echo.

echo [1/3] Kiem tra Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python chua duoc cai dat!
    echo Vui long cai dat Python tu https://python.org
    pause
    exit /b 1
)

echo.
echo [2/3] Cai dat dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Khong the cai dat dependencies!
    echo Vui long kiem tra ket noi internet va thu lai.
    pause
    exit /b 1
)

echo.
echo [3/3] Khoi dong ung dung Streamlit...
echo.
echo ========================================
echo  UNG DUNG DANG CHAY TAI: 
echo  http://localhost:8501
echo ========================================
echo.
echo Nhan Ctrl+C de dung ung dung
echo.

streamlit run app.py

pause

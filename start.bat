@echo off
title AI Bankruptcy Risk Prediction
cls
echo ======================================================================
echo          AI Bankruptcy Risk Prediction - Web Arayuzu Baslatiliyor
echo ======================================================================
echo.
echo [*] Sunucu baslatiliyor...
echo [*] Web Arayuzu Adresi: http://127.0.0.1:8000
echo.
echo ----------------------------------------------------------------------
echo Tarayiciniz birkac saniye icinde otomatik olarak acilacaktir...
echo Sunucuyu kapatmak icin bu pencereyi kapatabilir veya CTRL+C yapabilirsiniz.
echo ----------------------------------------------------------------------
echo.

:: Tarayiciyi otomatik ac
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

:: Web app klasorune gec
cd /d "%~dp0web_app"

:: Python 3.11 launcher ile calistir (PyTorch kurulu ortam)
py -3.11 -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
if %errorlevel% neq 0 (
    echo.
    echo [*] Standart Python komutu deneniyor...
    python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
)
if %errorlevel% neq 0 (
    echo.
    echo [*] Uvicorn dogrudan deneniyor...
    uvicorn main:app --reload --host 127.0.0.1 --port 8000
)

pause

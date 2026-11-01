@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Thai Orchid Classification - Training
python -u training\run_training.py
if errorlevel 1 (
  echo.
  echo การรันไม่สำเร็จ กรุณาตรวจข้อความด้านบน
)
pause

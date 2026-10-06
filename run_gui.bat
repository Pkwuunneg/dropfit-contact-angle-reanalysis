@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
"%PYTHON_EXE%" run_app.py
if errorlevel 1 pause

@echo off
chcp 65001 >nul
cd /d "%~dp0"
python --version
if errorlevel 1 (
  echo Install Python 3.11 or later and enable Add Python to PATH.
  pause
  exit /b 1
)
python -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Installation complete. Run run_demo.bat or run_gui.bat.
pause
exit /b 0
:failed
echo Installation failed. See the error above; check Python version and internet access.
pause
exit /b 1

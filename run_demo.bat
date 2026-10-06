@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
"%PYTHON_EXE%" -m dropfit batch --input input_pdfs --output my_results
if errorlevel 1 (
  echo Please run install_windows.bat first, or check the error above.
  pause
  exit /b 1
)
start "" "%~dp0my_results\report.html"
pause

@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "VENV_DIR=%CD%\.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "REQUIREMENTS=%CD%\requirements.txt"
set "NO_PAUSE="

if /I "%~1"=="--no-pause" set "NO_PAUSE=1"

echo Installing VisualsProject runtime...
echo Project: %CD%
echo.

if not exist "%REQUIREMENTS%" (
    echo Cannot find requirements.txt.
    call :pause_if_needed
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo Creating virtual environment in .venv ...
    py -3 -m venv "%VENV_DIR%" >nul 2>nul
    if errorlevel 1 (
        python -m venv "%VENV_DIR%" >nul 2>nul
    )
)

if not exist "%PYTHON_EXE%" (
    echo Failed to create .venv.
    echo Please install Python 3 and make sure py or python is available in PATH.
    call :pause_if_needed
    exit /b 1
)

echo Python: %PYTHON_EXE%
echo.

echo Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 goto install_failed

echo.
echo Installing requirements...
"%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS%"
if errorlevel 1 goto install_failed

echo.
echo Verifying runtime imports...
"%PYTHON_EXE%" -c "import cv2, numpy, PIL, onnxruntime, mediapipe, torch, torchvision, matplotlib, psutil; print('runtime imports ok')"
if errorlevel 1 goto install_failed

echo.
echo Install complete.
echo Start the app with:
echo start_upper_computer.bat
echo.
echo Make sure required model files already exist under models before starting.
echo.
call :pause_if_needed
exit /b 0

:install_failed
echo.
echo Install failed. Please check the messages above.
call :pause_if_needed
exit /b 1

:pause_if_needed
if not defined NO_PAUSE pause
exit /b 0

@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo Starting upper computer...
echo Project: %CD%
echo Python:  %PYTHON_EXE%
echo.

"%PYTHON_EXE%" -c "import cv2, numpy, PIL, onnxruntime" >nul 2>nul
if errorlevel 1 (
    echo Missing runtime dependencies.
    echo Please run:
    echo "%PYTHON_EXE%" -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not exist "%CD%\models\rvm_mobilenetv3_fp32.onnx" (
    echo Missing RVM model:
    echo "%CD%\models\rvm_mobilenetv3_fp32.onnx"
    echo.
    pause
    exit /b 1
)

"%PYTHON_EXE%" main.py
if errorlevel 1 (
    echo.
    echo Upper computer exited with an error.
    pause
    exit /b 1
)

endlocal

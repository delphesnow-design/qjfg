@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "VENV_DIR=%CD%\.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "REQUIREMENTS=%CD%\requirements.txt"
set "MODEL_DIR=%CD%\models"
set "MODEL_PATH=%MODEL_DIR%\rvm_mobilenetv3_fp32.onnx"
set "MODEL_URL_PRIMARY=https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx"
set "MODEL_URL_FALLBACK=https://sourceforge.net/projects/robust-video-matting.mirror/files/v1.0.0/rvm_mobilenetv3_fp32.onnx/download"
set "MIN_MODEL_BYTES=10000000"
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

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

echo.
echo Preparing RVM model...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $out='%MODEL_PATH%'; $min=[int64]%MIN_MODEL_BYTES%; if (Test-Path -LiteralPath $out) { $item=Get-Item -LiteralPath $out; if ($item.Length -ge $min) { Write-Host ('Model ready: ' + $out); exit 0 }; Write-Host 'Existing model is incomplete, re-downloading...'; Remove-Item -LiteralPath $out -Force }; $tmp=$out + '.download'; if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force }; $urls=@('%MODEL_URL_PRIMARY%','%MODEL_URL_FALLBACK%'); foreach ($url in $urls) { try { Write-Host ('Downloading: ' + $url); Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing; $item=Get-Item -LiteralPath $tmp; if ($item.Length -lt $min) { throw ('Downloaded file is too small: ' + $item.Length + ' bytes') }; Move-Item -LiteralPath $tmp -Destination $out -Force; Write-Host ('Saved model: ' + $out); exit 0 } catch { Write-Warning $_.Exception.Message; if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force } } }; exit 1"
if errorlevel 1 (
    echo Failed to download RVM model.
    echo You can manually download:
    echo %MODEL_URL_PRIMARY%
    echo and save it as:
    echo %MODEL_PATH%
    call :pause_if_needed
    exit /b 1
)

echo.
echo Verifying runtime imports...
"%PYTHON_EXE%" -c "import cv2, numpy, PIL, onnxruntime; print('runtime imports ok')"
if errorlevel 1 goto install_failed

echo.
echo Install complete.
echo Start the app with:
echo start_upper_computer.bat
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

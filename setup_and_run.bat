@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   RVC Voice Model Evaluator - Environment Setup ^& Run
echo ============================================================
echo.

:: ============================================================
:: Step 1: Check Python
:: ============================================================
echo [1/5] Checking Python installation...

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    echo.
    echo Please install Python 3.9+ from:
    echo   https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    goto :end
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo         Python %PYVER% found.
echo.

:: ============================================================
:: Step 2: Create virtual environment (if missing)
:: ============================================================
echo [2/5] Setting up virtual environment...

if exist ".venv\Scripts\python.exe" (
    echo         Virtual environment already exists. Skipping.
) else (
    echo         Creating .venv...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        goto :end
    )
    echo         Virtual environment created.
)
echo.

:: ============================================================
:: Step 3: Install dependencies (if not installed)
:: ============================================================
echo [3/5] Installing Python dependencies...

set VENV_PYTHON=.venv\Scripts\python.exe

"%VENV_PYTHON%" -c "import torch" >nul 2>&1
if %errorlevel% equ 0 (
    echo         Dependencies already installed. Skipping.
) else (
    echo         Installing from requirements.txt...
    echo         (This may take several minutes on first run)
    "%VENV_PYTHON%" -m pip install --upgrade pip >nul 2>&1
    "%VENV_PYTHON%" -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies.
        echo.
        echo Try running manually:
        echo   .venv\Scripts\activate
        echo   pip install -r requirements.txt
        goto :end
    )
    echo         Dependencies installed successfully.
)
echo.

:: ============================================================
:: Step 4: Download ECAPA model (if missing)
:: ============================================================
echo [4/5] Checking ECAPA-TDNN Speaker Recognition model...

if exist "ecapa_model\hyperparams.yaml" (
    echo         ECAPA model already exists. Skipping.
) else (
    echo         Downloading SpeechBrain ECAPA-TDNN model...
    echo         (This downloads ~55MB on first run)
    
    "%VENV_PYTHON%" -c "from speechbrain.pretrained import EncoderClassifier; EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='ecapa_model')"
    if %errorlevel% neq 0 (
        echo [WARNING] Failed to auto-download ECAPA model.
        echo.
        echo You can manually download it:
        echo   1. Install Git LFS: https://git-lfs.com
        echo   2. Run: git clone https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb ecapa_model
        echo.
        echo The evaluator will still run, but speaker similarity scoring
        echo will be disabled. All other metrics will work normally.
    ) else (
        echo         ECAPA model downloaded successfully.
    )
)
echo.

:: ============================================================
:: Step 5: Run evaluation
:: ============================================================
echo [5/5] Running evaluation...
echo ============================================================
echo.

"%VENV_PYTHON%" evaluate.py %*

echo.
echo ============================================================
echo   Evaluation complete.
echo.

if exist "results" (
    echo   Results saved to: results\
    dir /b "results\*.*" 2>nul
) else (
    echo   No results were generated.
)

echo ============================================================

:end
pause

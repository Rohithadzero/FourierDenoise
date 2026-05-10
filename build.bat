@echo off
echo   FourierDenoise — Build Script
echo.

REM Check Python Installation
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Download it from https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo        Found: %%i
echo.

REM Create Virtual Environment
echo [2/5] Setting up virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo        Creating new virtual environment...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo        Virtual environment created.
) else (
    echo        Virtual environment already exists.
)
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo        pip is up to date.
echo.

REM Install Dependencies
echo [4/5] Installing dependencies from requirements.txt...
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found!
    pause
    exit /b 1
)

python -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install one or more dependencies.
    pause
    exit /b 1
)
echo.

REM Verify critical libraries
echo        Verifying installed libraries...
set VERIFY_FAIL=0

for %%L in (customtkinter numpy scipy matplotlib soundfile sounddevice PIL) do (
    python -c "import %%L" >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo        [MISSING] %%L
        set VERIFY_FAIL=1
    ) else (
        echo        [OK] %%L
    )
)

echo        All core libraries installed!
echo.

REM Build with PyInstaller
echo [5/5] Building with PyInstaller...
pyinstaller --onedir --windowed --name FourierDenoise ^
  --add-data "samples;samples" ^
  --noconfirm ^
  main.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] PyInstaller build failed! Check errors above.
    pause
    exit /b 1
)

echo.
echo   BUILD SUCCESSFUL!
echo Output: dist\FourierDenoise\FourierDenoise.exe
echo.
pause

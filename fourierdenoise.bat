@echo off
echo   FourierDenoise — Run Script
echo.

REM Check if build.bat was run first
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo         Please run build.bat first to set up the project.
    echo.
    pause
    exit /b 1
)

REM  Activate virtual environment 
call venv\Scripts\activate.bat

REM  Launch FourierDenoise 
echo Starting FourierDenoise...
echo.
python main.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
    exit /b 1
)

pause

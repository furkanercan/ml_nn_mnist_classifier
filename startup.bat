@echo off
:: Ask if requirements.txt is finalized
set /p finalized="Is the requirements.txt file finalized? (y/n): "

if /I "%finalized%" NEQ "y" (
    echo Please finalize the requirements.txt before proceeding.
    exit /b
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate virtual environment
echo Activating virtual environment...
call .\venv\Scripts\activate

:: Install dependencies
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

:: Confirm completion
echo Setup complete. Virtual environment activated and dependencies installed.

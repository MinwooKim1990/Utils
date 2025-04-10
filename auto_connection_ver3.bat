@echo off
echo Connection3 start.
echo.

REM Conda venv activate
echo Conda venv(general2) activate.

REM Conda initiate
call conda activate base

REM general2 activate
call conda activate general2

echo activation complete!
echo.

echo server starting.

REM run connection3.py in the same directory
python "%~dp0connection_ver3.py"

echo server shut down.
pause
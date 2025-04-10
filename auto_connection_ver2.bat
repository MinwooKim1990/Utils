@echo off
echo Connection start.
echo.

REM Conda venv activate
echo Conda venv(general2) activate.

REM Conda initiate
call conda activate base

REM genera2 activate
call conda activate general2

echo activation complate!
echo.

echo server starting.

REM run connection2.py in the same directory
python "%~dp0connection_ver2.py"

echo server shut down.
pause
@echo off
echo WebDAV server start
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

REM run vid.py in the same directory
python "%~dp0vid.py"

echo server shut down.
pause
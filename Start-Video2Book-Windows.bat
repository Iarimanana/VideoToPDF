@echo off
rem Double-click to start Video2Book. The first start installs it (needs internet once).
cd /d "%~dp0"
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
if exist ".venv\Scripts\python.exe" goto run
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 goto nopython
echo First start: installing Video2Book. This takes a few minutes...
%PY% -m venv .venv || goto fail
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e ".[ocr]" || goto fail
:run
".venv\Scripts\python.exe" -m video2book.launch
pause
exit /b 0
:nopython
echo.
echo Python 3.10 or newer is needed.
echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
echo then double-click this file again.
pause
exit /b 1
:fail
echo.
echo Installation failed - see the messages above. Delete the ".venv" folder and try again.
pause
exit /b 1

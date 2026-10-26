@echo off
REM ---------------------------------------------------------------------------------------------
REM Full reproduction on Windows: creates the Python environment (first run only), installs the
REM pinned libraries and runs every stage of the pipeline. Double-click this file, or run it from
REM a command prompt with optional arguments, for example:
REM     run_pipeline.bat                    full run; the previous outputs are first moved to
REM                                         _archive\outputs_<date>_<time>, so the run starts clean
REM     run_pipeline.bat --resume           continue an interrupted full run without archiving;
REM                                         finished robustness tasks are not repeated
REM     run_pipeline.bat --quick            smoke test (a few minutes)
REM     run_pipeline.bat --stage figures    one stage only (outputs are kept)
REM     run_pipeline.bat --workers 6        fewer robustness processes (less memory)
REM Only a run without arguments archives the previous outputs.
REM Log: outputs\logs\pipeline.log
REM ---------------------------------------------------------------------------------------------
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM Arguments: --resume is handled here and removed; everything else goes to run_pipeline.py unchanged
REM (string replacement keeps comma-separated stage lists intact).
set "ARGS= %*"
set "RESUME=0"
if not "%ARGS:--resume=%"=="%ARGS%" set "RESUME=1"
set "ARGS=%ARGS:--resume=%"
set "NOSPACE=%ARGS: =%"

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating the virtual environment .venv ...
    py -3 -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo Python was not found. Install Python 3.12 or later from python.org and tick "Add python.exe to PATH".
    pause
    exit /b 1
)

echo [2/4] Installing the pinned libraries ...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt --quiet || ".venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet

if not defined NOSPACE if "%RESUME%"=="0" (
    echo [3/4] Full run: moving the previous outputs to _archive ...
    ".venv\Scripts\python.exe" tools\archive_outputs.py
    if errorlevel 1 (
        pause
        exit /b 1
    )
)
set "TEEOPT="
if "%RESUME%"=="1" (
    echo [3/4] Resuming: previous outputs are kept; the log is appended.
    set "TEEOPT=-Append"
)

if not exist "outputs\logs" mkdir "outputs\logs"
echo [4/4] Running the pipeline. Progress is shown here and written to outputs\logs\pipeline.log
powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:PYTHONUTF8='1'; & '.\.venv\Scripts\python.exe' -u run_pipeline.py %ARGS% 2>&1 | Tee-Object %TEEOPT% -FilePath 'outputs\logs\pipeline.log'"

echo.
echo Finished. Tables: outputs\tables   Figures: outputs\figures   Logs: outputs\logs
pause

@echo off
REM ---------------------------------------------------------------------------------------------
REM Additional checks of the manuscript (README, "Additional checks").
REM Run after a full run of run_pipeline.bat, from the same checkout: the scripts read outputs\ and
REM the raw tables in data\raw, and write only to outputs_extra\. Usage:
REM     tools\run_revision_checks.bat                 seeds 1 to 10, as in the learner comparison
REM     tools\run_revision_checks.bat 1 2 3           other seeds
REM An interrupted run can be restarted with the same command: finished seeds and sample variants
REM are not refitted. Log: outputs_extra\logs\revision_checks.log
REM ---------------------------------------------------------------------------------------------
setlocal
cd /d "%~dp0\.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "PY=.venv\Scripts\python.exe"
set "SEEDS=%*"
if not defined SEEDS set "SEEDS=1 2 3 4 5 6 7 8 9 10"

if not exist "%PY%" (
    echo The environment .venv was not found. Run run_pipeline.bat first.
    pause
    exit /b 1
)
if not exist "outputs\data\features_t0.pkl" (
    echo outputs\data\features_t0.pkl was not found. These checks need the files of a full run of run_pipeline.bat.
    pause
    exit /b 1
)
if not exist "outputs_extra\logs" mkdir "outputs_extra\logs"
set "LOG=outputs_extra\logs\revision_checks.log"

REM step 0 compares this run with the newest archived full run (run_pipeline.bat moves the outputs found before a
REM full run to _archive; in a fresh checkout these are the outputs of the reported run)
if exist "_archive" (
    call :step "0/9 this run against the archived full run" tools\compare_manifests.py --archive _archive outputs\manifest.json || (
        echo WARNING: this run differs from the archived run; see the lines output differs above and in %LOG%.
        echo The other checks continue; report this before the results are used.
        >> "%LOG%" echo WARNING: this run differs from the archived run
    )
) else (
    echo No _archive folder; the comparison with the earlier run is skipped.
)
REM keep the manifest of this run as the record of the rerun (outputs\manifest.json can then be restored to the reported run)
if not exist outputs_extra\rerun_manifest.json copy outputs\manifest.json outputs_extra\rerun_manifest.json >nul
call :step "1/9 code and outputs against outputs\manifest.json" tools\verify_manifest.py outputs || goto :fail
call :step "2/9 uncertainty bands of Fig. 10" tools\net_benefit_bands.py outputs outputs_extra || goto :fail
call :step "3/9 RQ2 with strictly valid chronology" tools\rq2_chronology_check.py outputs outputs_extra || goto :fail
call :step "4/9 timing of the daily alert policy (RQ4)" tools\policy_timing_check.py outputs outputs_extra || goto :fail
call :step "5/9 weight cap and stratified latency (RQ3)" tools\recal_cap_check.py outputs outputs_extra || goto :fail
call :step "6/9 seed check, seeds %SEEDS%" tools\extra_seeds.py outputs outputs_extra --seeds %SEEDS% --data data\raw || goto :fail
call :step "7/9 sample conditions (RQ1 to RQ4)" tools\sample_variants.py data\raw outputs outputs_extra || goto :fail
call :step "8/9 summaries" tools\summarise_extra.py outputs outputs_extra || goto :fail
call :step "9/9 LaTeX macros" tools\gen_extra_tex.py outputs_extra outputs_extra\results_extra.tex || goto :fail
echo.
echo All checks finished. Results: outputs_extra\tables, outputs_extra\figures, outputs_extra\results_extra.tex
pause
exit /b 0

:step
set "TITLE=%~1"
shift
set "CMD="
:collect
if "%~1"=="" goto :run
set "CMD=%CMD% %1"
shift
goto :collect
:run
echo.
echo [%TIME:~0,8%] %TITLE%
>> "%LOG%" echo [%DATE% %TIME:~0,8%] %TITLE%: %PY%%CMD%
REM output is shown and appended to the log as UTF-8 (Tee-Object of Windows PowerShell would write UTF-16)
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Console]::OutputEncoding=[Text.Encoding]::UTF8; $env:PYTHONUTF8='1'; $log=Join-Path $PWD '%LOG%'; & '.\%PY%' -u%CMD% 2>&1 | ForEach-Object { $s=[string]$_; $s; [IO.File]::AppendAllText($log, $s+[Environment]::NewLine) }; exit $LASTEXITCODE"
exit /b %ERRORLEVEL%

:fail
echo.
echo A step failed; see %LOG%. Fix the cause and run the same command again.
pause
exit /b 1

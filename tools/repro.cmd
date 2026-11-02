@echo off
rem ---------------------------------------------------------------------------
rem repro.cmd -- reproduce every claim in this session, offline, in one go.
rem
rem   tools\repro.cmd            full run
rem   tools\repro.cmd fast       static half only (skips the CUDA stages)
rem
rem Exit code 0 means every gate passed. The transcript is written to
rem %TEMP%\7d_repro_<random>.log and the path is printed at the end.
rem ---------------------------------------------------------------------------
setlocal enabledelayedexpansion
set "ROOT=%~dp0.."
cd /d "%ROOT%" || (echo cannot cd to %ROOT% & exit /b 2)
set "MODE=%1"
if /i "%MODE%"=="fast" (set "MODEARG=--fast") else (set "MODEARG=")
set "LOG=%TEMP%\7d_repro_%RANDOM%%RANDOM%.log"
set "FAILED=0"

echo ============================================================
echo  ABRASAX-7D-Crystal reproduction   %DATE% %TIME%
echo  root: %CD%
echo  log:  %LOG%
echo ============================================================
echo. > "%LOG%"

call :stage "manifest (SHA-256 of every claimed artifact)" "python tools\verify_manifest.py"
call :stage "verify_all %MODEARG%" "python tools\verify_all.py %MODEARG%"
call :stage "verify_offline (network blocked)" "python tools\verify_offline.py %MODEARG%"

echo.
echo ============================================================
if "%FAILED%"=="0" (
  echo  RESULT: ALL STAGES PASSED
) else (
  echo  RESULT: %FAILED% STAGE^(S^) FAILED
)
echo  full transcript: %LOG%
echo ============================================================
if "%FAILED%"=="0" (exit /b 0) else (exit /b 1)

:stage
set "NAME=%~1"
set "CMD=%~2"
echo.
echo --- !NAME! -------------------------------------------
echo. >> "%LOG%"
echo --- !NAME! ------------------------------------------- >> "%LOG%"
call !CMD! >> "%LOG%" 2>&1
set "RC=!errorlevel!"
if "!RC!"=="0" (
  echo   OK   exit=!RC!
  echo   OK   exit=!RC! >> "%LOG%"
) else (
  echo   FAIL exit=!RC!   ^(see the log^)
  echo   FAIL exit=!RC! >> "%LOG%"
  set /a FAILED+=1
)
exit /b 0

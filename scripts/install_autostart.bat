@echo off
REM One-time setup: register RAG 2.0 as a logon autostart task (no window, auto-retry on crash).
REM Uninstall: schtasks /delete /tn RAG2_Prod /f

set TASK=RAG2_Prod
set XML=%~dp0rag2_task.xml
set TMPXML=%~dp0rag2_task.utf16.xml

REM schtasks /xml requires UTF-16LE (BOM). Source XML has no encoding declaration,
REM so convert UTF-8 -> UTF-16LE and import, then delete temp.
powershell -NoProfile -Command "(Get-Content -Path '%XML%' -Raw -Encoding utf8) | Out-File -Encoding unicode -FilePath '%TMPXML%'"
if not exist "%TMPXML%" (
  echo [FAIL] Could not generate task XML. Make sure this file and rag2_task.xml are in the same folder.
  pause
  exit /b 1
)

schtasks /create /tn "%TASK%" /xml "%TMPXML%" /f
if errorlevel 1 (
  echo.
  echo [FAIL] Could not create task. Make sure you ran this as Administrator.
  del "%TMPXML%" 2>nul
  echo.
  pause
  exit /b 1
)

del "%TMPXML%" 2>nul
echo.
echo [OK] Created scheduled task "%TASK%" (logon autostart, no window, auto-retry on crash)
echo   It will start automatically on next logon. No manual start needed.
echo   URL: https://rag.uniquejingclaudecoding.top/  (enter RAG_API_KEY)
echo.
echo   Manage:
echo     status   : schtasks /query /tn %TASK%
echo     start now: schtasks /run /tn %TASK%
echo     disable  : schtasks /change /tn %TASK% /disable
echo     enable   : schtasks /change /tn %TASK% /enable
echo     uninstall: schtasks /delete /tn %TASK% /f
echo.
pause

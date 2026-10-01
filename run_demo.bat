@echo off
REM BOPIS demo launcher - double-click this to start the demo.
REM Opens a PowerShell window, sets execution policy for this session only,
REM and runs demo.ps1 from the same folder.

cd /d "%~dp0"

PowerShell -NoProfile -ExecutionPolicy Bypass -File "%~dp0demo.ps1" %*

pause

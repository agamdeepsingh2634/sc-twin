@echo off
rem Runs the last two stages (transfer test, demand figure) if the main run was closed at stage 20.
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set MPLBACKEND=Agg
set VPY=%~dp0.venv\Scripts\python.exe
set SCTWIN_M5=%~dp0results\m5_foods_daily.npz
echo [21/22] Real M5 ^| transfer test
"%VPY%" -m experiments.run_transfer > finish_log.txt 2>&1
type finish_log.txt
echo [22/22] Real M5 ^| demand comparison figure
"%VPY%" -m experiments.plot_demand >> finish_log.txt 2>&1
echo.
echo DONE. Next: double-click TUNE_DQN_QUICK_TEST.bat
pause

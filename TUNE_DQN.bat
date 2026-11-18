@echo off
rem Tunes the deep Q-network on the GPU (4 settings in parallel, best checkpoint kept).
rem Runs synthetic demand, then real M5 demand. About 2 hours. Add "quick" for a 3-minute test.
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
title FMCG Twin - deep RL tuning
set VPY=%~dp0.venv\Scripts\python.exe
set Q=
if /i "%1"=="quick" set Q=--quick
echo ===== Synthetic demand =====
set SCTWIN_M5=
"%VPY%" -m experiments.tune_dqn_torch %Q% 2>&1 | "%VPY%" -c "import sys;f=open('tuning_log.txt','a',encoding='utf-8');[ (sys.stdout.write(l),f.write(l),f.flush()) for l in sys.stdin]"
echo ===== Real M5 demand =====
set SCTWIN_M5=%~dp0results\m5_foods_daily.npz
"%VPY%" -m experiments.tune_dqn_torch %Q% 2>&1 | "%VPY%" -c "import sys;f=open('tuning_log.txt','a',encoding='utf-8');[ (sys.stdout.write(l),f.write(l),f.flush()) for l in sys.stdin]"
echo.
echo ALL DONE. Tell Claude the tuning has finished.
pause

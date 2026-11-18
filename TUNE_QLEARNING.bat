@echo off
rem Tunes the tabular Q-learning agent (6 settings, best checkpoint kept). CPU only, can run alongside TUNE_DQN.bat.
rem Synthetic then real M5 demand. About 30-45 minutes. Add "quick" for a 1-minute test.
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
title FMCG Twin - Q-learning tuning
set VPY=%~dp0.venv\Scripts\python.exe
set Q=
if /i "%1"=="quick" set Q=--quick
echo ===== Synthetic demand =====
set SCTWIN_M5=
"%VPY%" -m experiments.tune_qlearning %Q% 2>&1 | "%VPY%" -c "import sys;f=open('qlearning_tuning_log.txt','a',encoding='utf-8');[ (sys.stdout.write(l),f.write(l),f.flush()) for l in sys.stdin]"
echo ===== Real M5 demand =====
set SCTWIN_M5=%~dp0results\m5_foods_daily.npz
"%VPY%" -m experiments.tune_qlearning %Q% 2>&1 | "%VPY%" -c "import sys;f=open('qlearning_tuning_log.txt','a',encoding='utf-8');[ (sys.stdout.write(l),f.write(l),f.flush()) for l in sys.stdin]"
echo.
echo ALL DONE. Tell Claude the Q-learning tuning has finished.
pause

@echo off
rem ================================================================
rem  FMCG Supply-Chain Twin - full re-run on this PC (GPU for deep RL)
rem  Double-click this file. First run downloads PyTorch (~2.5 GB).
rem  Add "quick" to test first:  RUN_ON_MY_PC.bat quick
rem ================================================================
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
title FMCG Supply-Chain Twin - full re-run

set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY (
  where python >nul 2>nul && set PY=python
)
if not defined PY (
  echo Python was not found. Install Python 3.12 from python.org
  echo and tick "Add python.exe to PATH", then run this file again.
  pause
  exit /b 1
)

echo [setup] Python:
%PY% --version
if not exist ".venv\Scripts\python.exe" (
  echo [setup] Creating a private Python environment in .venv ...
  %PY% -m venv .venv
)
set VPY=%~dp0.venv\Scripts\python.exe

echo [setup] Installing numpy, pandas, scipy, matplotlib ...
"%VPY%" -m pip install --upgrade pip --quiet
"%VPY%" -m pip install numpy pandas scipy matplotlib --quiet

echo [setup] Checking for an NVIDIA GPU ...
nvidia-smi
if errorlevel 1 (
  echo [setup] No NVIDIA GPU found - installing CPU PyTorch.
  set TORCH_INDEX=https://download.pytorch.org/whl/cpu
) else (
  echo [setup] NVIDIA GPU found - installing CUDA PyTorch.
  set TORCH_INDEX=https://download.pytorch.org/whl/cu124
)
"%VPY%" -c "import torch" 2>nul || "%VPY%" -m pip install torch --index-url %TORCH_INDEX%
"%VPY%" -c "import torch; print('[setup] torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo.
echo [run] Starting all experiments. This takes a few hours - leave the
echo [run] laptop plugged in and stop it from sleeping.
if /i "%1"=="quick" (
  "%VPY%" run_all.py --quick
) else (
  "%VPY%" run_all.py
)
echo.
pause

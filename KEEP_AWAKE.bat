@echo off
title KEEP AWAKE - close when runs finish
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" keep_awake.py

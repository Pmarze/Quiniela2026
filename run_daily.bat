@echo off
REM ─────────────────────────────────────────────────────────────
REM  Quiniela2026 · Actualización diaria
REM  Úsalo en el Programador de tareas de Windows o manualmente.
REM ─────────────────────────────────────────────────────────────
cd /d D:\Quiniela2026
call "%ProgramData%\anaconda3\Scripts\activate.bat" quiniela2026
python scripts\daily_update.py

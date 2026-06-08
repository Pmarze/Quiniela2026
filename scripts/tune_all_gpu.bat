@echo off
REM ============================================================
REM  Tuning GPU para modelos Elo de quiniela
REM  Requiere PyTorch con CUDA instalado:
REM    conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
REM
REM  Uso desde Anaconda Prompt (con env quiniela2026 activo):
REM    conda activate quiniela2026
REM    cd /d D:\Quiniela2026
REM    scripts\tune_all_gpu.bat
REM
REM  bradley_terry_davidson y attack_defense_poisson usan CPU
REM  (los corre tune_all_cpu_rest.bat al final)
REM ============================================================

set YEARS=2018 2022
set PYTHON=python -u

echo.
echo ============================================================
echo  TUNING GPU - %DATE% %TIME%
echo ============================================================
echo.

echo [1/3] elo_poisson  (1944 trials en GPU)
%PYTHON% scripts/tune_models_gpu.py --model elo_poisson --years %YEARS%
if errorlevel 1 ( echo ERROR en elo_poisson & goto :error )

echo.
echo [2/3] elo_dixon_coles  (1944 trials en GPU)
%PYTHON% scripts/tune_models_gpu.py --model elo_dixon_coles --years %YEARS%
if errorlevel 1 ( echo ERROR en elo_dixon_coles & goto :error )

echo.
echo [3/3] draw_specialist  (1440 trials en GPU)
%PYTHON% scripts/tune_models_gpu.py --model draw_specialist --years %YEARS%
if errorlevel 1 ( echo ERROR en draw_specialist & goto :error )

echo.
echo ============================================================
echo  GPU COMPLETO - %DATE% %TIME%
echo  Ahora corre tune_all_cpu_rest.bat para los otros 2 modelos
echo ============================================================
goto :end

:error
echo.
echo *** TUNING INTERRUMPIDO CON ERROR ***
exit /b 1

:end

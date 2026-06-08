@echo off
REM ============================================================
REM  Tuning exhaustivo de todos los modelos de quiniela
REM  Ajusta --workers segun tus cores disponibles
REM  Recomendado: mitad de los cores logicos de tu CPU
REM  Ejemplo: CPU de 16 cores -> --workers 8
REM
REM  Uso desde Anaconda Prompt (con env quiniela2026 activo):
REM    conda activate quiniela2026
REM    cd /d D:\Quiniela2026
REM    scripts\tune_all.bat
REM ============================================================

set WORKERS=8
set YEARS=2014 2018 2022
set PYTHON=python -u

echo.
echo ============================================================
echo  TUNING EXHAUSTIVO - %DATE% %TIME%
echo  Workers: %WORKERS%   Anos: %YEARS%
echo ============================================================
echo.

echo [1/5] elo_poisson  (grid ~1944 trials)
%PYTHON% scripts/tune_models.py --model elo_poisson --workers %WORKERS% --years %YEARS%
if errorlevel 1 ( echo ERROR en elo_poisson & goto :error )

echo.
echo [2/5] elo_dixon_coles  (grid ~1944 trials)
%PYTHON% scripts/tune_models.py --model elo_dixon_coles --workers %WORKERS% --years %YEARS%
if errorlevel 1 ( echo ERROR en elo_dixon_coles & goto :error )

echo.
echo [3/5] draw_specialist  (grid ~1440 trials)
%PYTHON% scripts/tune_models.py --model draw_specialist --workers %WORKERS% --years %YEARS%
if errorlevel 1 ( echo ERROR en draw_specialist & goto :error )

echo.
echo [4/5] bradley_terry_davidson  (grid ~1584 trials)
%PYTHON% scripts/tune_models.py --model bradley_terry_davidson --workers %WORKERS% --years %YEARS%
if errorlevel 1 ( echo ERROR en bradley_terry_davidson & goto :error )

echo.
echo [5/5] attack_defense_poisson  (grid ~4480 trials)
%PYTHON% scripts/tune_models.py --model attack_defense_poisson --workers %WORKERS% --years %YEARS%
if errorlevel 1 ( echo ERROR en attack_defense_poisson & goto :error )

echo.
echo ============================================================
echo  TUNING COMPLETO - %DATE% %TIME%
echo  Resultados en: D:\Quiniela2026\data\backtests\tuning_*.json
echo ============================================================
goto :end

:error
echo.
echo *** TUNING INTERRUMPIDO CON ERROR ***
exit /b 1

:end

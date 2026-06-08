@echo off
REM ============================================================
REM  Tuning CPU para modelos que no soportan GPU:
REM    - bradley_terry_davidson
REM    - attack_defense_poisson
REM
REM  Corre DESPUES de tune_all_gpu.bat
REM ============================================================

set WORKERS=8
set YEARS=2018 2022
set PYTHON=python -u

echo.
echo ============================================================
echo  TUNING CPU (modelos sin GPU) - %DATE% %TIME%
echo  Workers: %WORKERS%
echo ============================================================
echo.

echo [1/2] bradley_terry_davidson  (~1584 trials, 8 workers)
%PYTHON% scripts/tune_models.py --model bradley_terry_davidson --workers %WORKERS% --years %YEARS%
if errorlevel 1 ( echo ERROR en bradley_terry_davidson & goto :error )

echo.
echo [2/2] attack_defense_poisson  (~4480 trials, 8 workers)
%PYTHON% scripts/tune_models.py --model attack_defense_poisson --workers %WORKERS% --years %YEARS%
if errorlevel 1 ( echo ERROR en attack_defense_poisson & goto :error )

echo.
echo ============================================================
echo  TODO COMPLETO - %DATE% %TIME%
echo  Resultados en: D:\Quiniela2026\data\backtests\tuning_*.json
echo ============================================================
goto :end

:error
echo.
echo *** TUNING INTERRUMPIDO CON ERROR ***
exit /b 1

:end

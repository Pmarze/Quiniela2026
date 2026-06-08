# 032 - Calibracion historica de scoreline

## Contexto

Se detecto que los marcadores TOP actuales estaban demasiado concentrados en `1-1`, `1-0` y `0-1`.

Antes de implementar el cambio, esos tres marcadores representaban aproximadamente:

- 93.4% de los TOP picks de modelos para partidos con equipos definidos.
- 27% a 29% de los marcadores reales en los Mundiales historicos y en 2018+2022.

Tambien se observo que el xG total promedio de los modelos sanos estaba cerca del historico mundialista reciente. Por eso el problema no era solo "pocos goles esperados", sino la conversion de una matriz probabilistica a un unico marcador exacto.

## Decision

Se creo `calibrated_scoreline_ensemble` como nuevo modelo candidato y default de quiniela.

No reemplaza modelos existentes. Consume las matrices de modelos base igual que los ponderadores, pero aplica una calibracion posterior:

- conserva las probabilidades 1X2 del ensemble;
- calcula priors historicos de marcador por resultado (`1`, `X`, `2`) usando Mundiales desde 1974;
- mezcla la distribucion condicional del modelo con ese prior;
- penaliza suavemente `1-0`, `1-1` y `0-1`;
- da un bono leve a marcadores con 3+ goles totales.

## Resultado operativo

En la corrida `pred_20260608T170507Z_eb175ef4`, el nuevo default produjo:

- 72 partidos pronosticados y 32 enmascarados por placeholders.
- Pick seleccionado: `1-0`/`1-1`/`0-1` bajo a 25.0%.
- TOP score: `1-0`/`1-1`/`0-1` bajo a 40.3%.

Distribucion seleccionada:

```text
2-1: 36
1-2: 18
0-1: 15
1-0: 3
```

## Backtest 2018/2022

Run: `backtest_wc2018_2022_base_20260608T171014Z_5b94d08b`

Comparacion principal:

```text
weighted_1x2_ensemble             164 / 640 puntos, eff 25.62%
weighted_points_ensemble          157 / 640 puntos, eff 24.53%
calibrated_scoreline_ensemble     155 / 640 puntos, eff 24.22%
```

Tradeoff observado:

- El calibrado no maximiza puntos historicos; queda 2 puntos debajo de `weighted_points_ensemble` y 9 debajo de `weighted_1x2_ensemble`.
- Si cumple el objetivo de amplitud: en backtest, los picks seleccionados `1-0`/`1-1`/`0-1` bajan de 94.5% en `weighted_points_ensemble` a 44.5% en `calibrated_scoreline_ensemble`.
- El modelo queda como default porque esta etapa priorizo corregir la concentracion excesiva de marcadores, no solo maximizar el score historico bruto.

## Archivos tocados

- `src/quiniela/ensemble/weighted.py`
- `scripts/run_model.py`
- `src/quiniela/backtest/runner.py`
- `src/quiniela/ui/dashboard.py`
- `configs/models.yaml`
- `configs/backtest.yaml`
- `docs/weighted_ensemble.md`
- `docs/model_contract.md`

## Comandos

Con conda `quiniela2026` activo, desde `D:\Quiniela2026`:

```powershell
python scripts\run_model.py
python scripts\generate_dashboard.py
```

Para evaluar historicamente el nuevo modelo:

```powershell
python scripts\run_backtest.py
python scripts\generate_dashboard.py
```

## Estado

Activo. `configs/models.yaml -> default_quiniela_model_id` apunta a `calibrated_scoreline_ensemble`.

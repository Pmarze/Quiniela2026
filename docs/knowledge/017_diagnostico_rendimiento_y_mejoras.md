# 017 - Diagnóstico de rendimiento y plan de mejoras (Ola 1)

## Contexto

Primer backtest walk-forward completo sobre 3 Mundiales (2014, 2018, 2022) — 192 partidos, 6 modelos.

Run de referencia: `backtest_wc2014_2018_2022_base_20260605T191241Z_f631e74d`

## Resultados del Backtest (estrategia max_points)

| Modelo | Pts | Eff% | Exact% | Win% | Brier | LogLoss |
|---|---|---|---|---|---|---|
| bradley_terry_davidson | 504 | 26.3 | 12.0 | 54.2 | 0.610 | 1.018 |
| draw_specialist | 502 | 26.2 | 12.0 | 50.5 | 0.592 | 0.995 |
| elo_poisson | 490 | 25.5 | 11.5 | 52.6 | 0.591 | 0.993 |
| elo_dixon_coles | 488 | 25.4 | 11.5 | 45.8 | 0.594 | 0.997 |
| attack_defense_poisson | 322 | 16.8 | 5.7 | 34.9 | 0.643 | 1.063 |
| baseline_poisson | 310 | 16.2 | 8.3 | 21.4 | 0.647 | 1.068 |

Referencia aleatoria 1X2: Brier ≈ 0.667, LogLoss ≈ 1.099. Los modelos Elo-Poisson
están apenas por encima del azar en calibración de probabilidades.

## Causas Raíz Identificadas

### 1. Diversidad falsa entre modelos

`bradley_terry_davidson`, `draw_specialist` y `elo_dixon_coles` importan y llaman
`_fit_elo_ratings` de `elo_poisson.py`. Solo difieren en post-procesado de la matriz.
Un ensemble de estos modelos casi no reduce varianza.

### 2. recency_weight ignorado en el ajuste de Elo

`elo_poisson.py:144` usaba `importance = max(0.2, importance_weight)` pero ignoraba
`recency_weight`. Un partido de 1990 movía los ratings tanto como uno de 2022.

El dataset incluye partidos desde 1872. Sin decaimiento temporal en el ajuste,
el histórico lejano diluye la señal reciente.

### 3. Amistosos diluyen el rating

18,312 amistosos (38% del dataset, importance_weight=0.6) participaban en el ajuste
de Elo con un floor de 0.2 (no 0.0), amplificando su efecto. Los amistosos
tienen bajo valor predictivo para partidos de torneo.

### 4. Hiperparámetros sin calibrar contra backtest

k_factor=22, goal_scale=0.55, dixon_coles_rho=-0.10 son defaults sin búsqueda
sistemática. No existe evidencia de que sean óptimos para Mundial.

### 5. market_calibrated_poisson desactivado

La función `adjust_score_matrix_to_1x2` ya está implementada en `common.py:258`.
La calibración de mercado típicamente da +3-6 pp de Brier. Está marcada como
pendiente de fuente de datos.

### 6. Sin ensemble real

No existe combinación de matrices entre modelos. El dashboard selecciona uno
por partido; no se mezclan distribuciones.

## Cambios Implementados (Ola 1)

### 1.1 + 1.2 — recency_weight en delta de Elo y filtro de importancia mínima

Archivo: `src/quiniela/models/elo_poisson.py`

Antes:
```python
importance = max(0.2, match.importance_weight)
delta = k_factor * importance * goal_diff_scale * (actual_a - expected_a)
```

Después:
```python
combined_weight = max(0.05, match.importance_weight * match.recency_weight)
delta = k_factor * combined_weight * goal_diff_scale * (actual_a - expected_a)
```

Nuevo parámetro `min_importance_for_rating` (default 0.0 = sin cambio):
- Con 0.7: excluye amistosos (importance=0.6) del ajuste de Elo.
- Con 1.0: usa solo partidos de torneos importantes (WC, qualifiers, continentales).

El cálculo de goles base (`_global_weighted_goals_per_team`) no cambia; sigue usando
todos los partidos porque los amistosos informan el promedio global de goles.

Todos los modelos que heredan de elo_poisson (elo_dixon_coles, draw_specialist,
bradley_terry_davidson) reciben el cambio automáticamente.

### 1.3 — Scripts de tuning de hiperparámetros

**CPU** (`scripts/tune_models.py`): grid/random search con `ProcessPoolExecutor`.
Soporta todos los modelos. Progreso solo visible cuando termina un chunk de worker
(puede tardar 10+ min sin output visible con workers grandes).

**GPU** (`scripts/tune_models_gpu.py`): todos los trials corren simultáneamente como
tensores PyTorch `[N_trials, N_teams]`. 50-200x más rápido que CPU. Soporta
elo_poisson, elo_dixon_coles, draw_specialist. Ver nota 018 para detalles.

Espacio de búsqueda completo (grid):
- elo_poisson: 9×9×8×3 = 1,944 trials
- elo_dixon_coles: 9×9×8×3 = 1,944 trials
- draw_specialist: 9×8×10×2 = 1,440 trials
- bradley_terry_davidson: 9×8×11×2 = 1,584 trials (CPU únicamente)
- attack_defense_poisson: 10×7×8×8 = 4,480 trials (CPU únicamente)

Flujo de tuning recomendado:
```
# GPU para modelos Elo (~3-5 min con RTX 3050)
scripts\tune_all_gpu.bat

# CPU para el resto (~20 min con 8 workers)
scripts\tune_all_cpu_rest.bat
```

Resultados guardados en `data/backtests/tuning_<model>_<timestamp>.json`.
Después aplicar params óptimos en `configs/models.yaml`.

## Resultado del primer tuning completo (2026-06-05, elo_poisson, 200 trials random)

| Rk | Eff% | k_factor | goal_scale | home_advantage |
|---|---|---|---|---|
| 1 | 25.31% | 32 | 0.6 | 80 |
| 2 | 24.58% | 32 | 0.4 | 65 |

Baseline (pre-tuning): 23.44%. Mejora: +1.87 pp.

Parámetros aplicados a configs/models.yaml (todos los modelos Elo comparten k/gs/ha).
Tuning completo del grid (1944 trials) en ejecución con GPU.

## Pendiente (Ola 2 en adelante)

- Dixon-Coles MLE de verdad (scipy.optimize sobre histórico con decaimiento)
- XGBoost 1X2/scoreline con features de Elo, forma reciente, etapa
- Calibración con mercado (Polymarket/Pinnacle) → activar market_calibrated_poisson
- Ensemble ponderado por Brier inverso sobre las matrices de todos los modelos

## Estado

Activo. Complementa notas 015 y 016.

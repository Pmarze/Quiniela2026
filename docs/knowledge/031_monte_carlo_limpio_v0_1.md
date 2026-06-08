# 031 - Monte Carlo limpio v0.1

## Conocimiento

Se implemento `bayesian_monte_carlo_scoreline` como modelo Monte Carlo limpio sin Opta ni mercados externos.

La version v0.1 corre con la data ya disponible en el proyecto:

- historico internacional canonico;
- pesos de importancia;
- recencia;
- neutralidad/sede;
- estado del torneo;
- resultados reales solo antes del corte `as_of_utc`.

## Decision operativa

El modelo queda activo en `configs/models.yaml` y puede evaluarse en backtest 2018/2022.

En los ponderadores se deja con `fallback_weight=0.25` mientras no tenga metricas historicas. Esto evita que domine el ensemble por no tener todavia fila en `v_latest_backtest_model_metrics`.

Iteraciones configuradas:

```text
num_simulations = 20000
backtest_num_simulations = 5000
```

Despues del primer backtest se observo sobreproduccion de empates, por lo que los parametros iniciales se ajustaron a:

```text
goal_scale = 0.70
rating_uncertainty_sd = 25.0
lambda_log_sigma = 0.10
lambda_overdispersion = 0.03
dixon_coles_rho = 0.0
```

Backtest posterior del ajuste:

```text
backtest_run_id = backtest_wc2018_2022_base_20260608T045107Z_4811c40e
total_quiniela_points = 148 / 640
points_efficiency = 0.23125
exact_score_accuracy = 0.085938
winner_accuracy = 0.50000
draw_predictions = 16
```

Lectura: ya no sobreproduce empates y queda como base valida, aunque todavia por debajo de `elo_poisson` y de los ponderadores. Debe tratarse como candidato limpio a mejorar con tuning y/o FIFA ranking historico, no como modelo ganador final todavia.

Para multiplicar iteraciones desde terminal:

```powershell
python scripts\set_monte_carlo_iterations.py --multiplier 2
```

Para fijar valores:

```powershell
python scripts\set_monte_carlo_iterations.py --num-simulations 50000 --backtest-num-simulations 10000
```

## Fuentes adicionales recomendadas sin Opta

- FIFA ranking historico (`Dato-Futbol/fifa-ranking`): buen candidato para feature/prior backtesteable.
- StatsBomb open-data 2018/2022: util para calibrar goles vs xG mundialista, no necesario para v0.1.
- Fjelstul World Cup Database: util para granularidad de Mundiales y validacion historica.
- World Football Elo historico: util como comparador externo no Opta si se ingiere con corte temporal correcto.

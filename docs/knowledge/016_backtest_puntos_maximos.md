# 016 - Backtest con puntos maximos posibles

## Conocimiento

La comparacion visual de backtest debe mostrar no solo puntos obtenidos, sino tambien:

```text
puntos maximos posibles
eficiencia de puntos
```

Definicion:

```text
max_possible_points = matches_evaluated * configs/scoring.yaml.exact_score
points_efficiency = total_quiniela_points / max_possible_points
```

Con la regla actual de quiniela:

```text
exact_score = 5
```

Para 128 partidos, el maximo por modelo es:

```text
128 * 5 = 640 puntos
```

## Cambios

Se agregaron columnas a las metricas de backtest:

```text
max_possible_points
points_efficiency
```

Archivos modificados:

```text
src/quiniela/backtest/runner.py
src/quiniela/backtest/dashboard.py
docs/backtesting_validation.md
```

## Corrida Vigente

```text
backtest_run_id: backtest_wc2018_2022_base_20260605T184106Z_eea438c7
matches: 128
predictions: 768
```

Ranking:

```text
elo_dixon_coles: 160 / 640, 25.0%
bradley_terry_davidson: 156 / 640, 24.4%
draw_specialist: 150 / 640, 23.4%
elo_poisson: 149 / 640, 23.3%
baseline_poisson: 108 / 640, 16.9%
attack_defense_poisson: 97 / 640, 15.2%
```

## Dashboard

La pagina:

```text
outputs/validation_dashboard/index.html
```

ahora muestra:

```text
puntos obtenidos / maximo posible
porcentaje de eficiencia
barras contra el maximo posible
puntos por partido como puntos / 5
```

## Estado

Activo. Amplia la nota 015 sin reemplazarla.

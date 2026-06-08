# Ponderador weighted_ensemble

`weighted_ensemble` y sus variantes son la capa de ponderacion de modelos. No reemplazan a los modelos individuales: consumen sus predicciones y publican predicciones finales con el mismo contrato modular.

## Variantes activas

- `weighted_ensemble`: ponderador balanceado.
- `weighted_points_ensemble`: ponderador orientado a maximizar puntos de quiniela.
- `weighted_1x2_ensemble`: ponderador orientado a maximizar acierto 1X2.
- `weighted_exact_ensemble`: ponderador orientado a maximizar marcador exacto.
- `calibrated_scoreline_ensemble`: ponderador calibrado para reducir concentracion excesiva en marcadores bajos centrales.

## Que combina

Para cada partido usa las predicciones `status=ok` de los modelos base activos:

- matriz de marcadores;
- goles esperados;
- probabilidad 1X2 derivada de la matriz;
- marcador mas probable;
- marcador que maximiza puntos esperados.

El ponderador mezcla matrices de marcadores normalizadas. Despues selecciona el marcador recomendado con la misma regla de quiniela del proyecto.

`opta_power_poisson` esta excluido por defecto de los ponderadores. La decision busca que la propuesta principal de quiniela y los ensembles sigan siendo backtesteables contra 2018/2022 sin depender indirectamente de Opta.

## Calibracion historica de marcador

`calibrated_scoreline_ensemble` usa la misma mezcla de matrices que el ponderador de puntos, pero despues aplica una calibracion de scoreline:

- conserva las probabilidades 1X2 originales del ensemble;
- calcula priors historicos de marcadores por resultado (`1`, `X`, `2`) usando partidos mundialistas desde 1974;
- mezcla la distribucion condicional del modelo con ese prior historico;
- aplica una penalizacion suave a `1-0`, `1-1` y `0-1`;
- da un pequeno bono a marcadores con 3+ goles totales.

El objetivo es evitar que el marcador mas probable y el pick operativo caigan casi siempre en `1-1`, `1-0` u `0-1`, manteniendo la lectura 1X2 de los modelos base.

## Pesos

En produccion diaria intenta usar pesos derivados del ultimo backtest disponible:

```text
v_latest_backtest_model_metrics
```

El score de peso combina eficiencia de puntos, acierto exacto, acierto de empate/diferencia y acierto de ganador.

Si no existe backtest para un modelo, usa `fallback_weights` de `configs/models.yaml`.

Nota actual: `bayesian_monte_carlo_scoreline` tiene `fallback_weight=0.9` en los ponderadores. El valor anterior `0.25` era provisional; despues del backtest del Monte Carlo se subio para que participe de forma realista en validacion sin fuga temporal.

## Backtest

En backtest, por defecto no lee `v_latest_backtest_model_metrics`, para evitar fuga temporal desde evaluaciones historicas previas. Usa pesos configurados como fallback salvo que se active explicitamente `allow_backtest_weight_source`.

## Dashboard

`calibrated_scoreline_ensemble` queda como `default_quiniela_model_id`, por lo que la propuesta principal de quiniela viene del ponderador calibrado. Los hovers siguen mostrando cada modelo individual y ahora tambien incluyen las notas `ensemble_weights` y `scoreline_calibration` para revisar pesos y calibracion.

## Comandos

Con conda `quiniela2026` activo:

```powershell
python scripts\run_model.py
python scripts\generate_dashboard.py
```

Para compararlo historicamente:

```powershell
python scripts\run_backtest.py
python scripts\generate_validation_dashboard.py
```

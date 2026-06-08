# 023 - Guardrail 1X2 para tuning neural

Fecha: 2026-06-05

## Decision

El tuning del modelo neural no debe sacrificar el buen rendimiento 1X2 solo por mejorar puntos esperados de quiniela. El objetivo se cambia a `guarded_ev_points`.

## Razon

El 1X2 visible en el dashboard se deriva de la matriz de marcadores, no del `outcome_head` auxiliar. Por eso se agrega la metrica `matrix_outcome_accuracy` y se usa como guardrail principal.

## Implementacion

- `matrix_outcome_accuracy`: accuracy del 1X2 agregado desde la matriz de marcadores.
- `ev_outcome_accuracy`: accuracy del marcador elegido por maximizacion de puntos esperados.
- `guarded_ev_points`: combina `ev_mean_points`, bonus por 1X2 de matriz y exactos, y penaliza si cae debajo de umbrales definidos en `configs/neural_scoreline.yaml`.

## Implicacion

Si el tuning encuentra una configuracion con mas puntos pero peor 1X2, no deberia ser seleccionada automaticamente salvo que supere los guardrails configurados.

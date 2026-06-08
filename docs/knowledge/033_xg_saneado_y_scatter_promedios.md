# 033 - xG saneado y scatter con promedios

## Contexto

En el dashboard se detectaron valores de xG absurdamente altos para `neural_scoreline_mlp`, por ejemplo en `Turkey vs United States`.

La causa operativa es que algunas redes tienen una cabeza auxiliar de goles (`goals_head`) que puede salirse de escala, aunque la matriz de marcador siga siendo finita.

## Decision

Para predicciones construidas desde matriz (`successful_prediction_from_matrix`), el sistema ahora sanea `expected_goals_a` y `expected_goals_b`:

- si el xG bruto no es finito;
- si es negativo;
- si supera 6.0;

entonces se reemplaza por el xG esperado derivado de la propia matriz de marcadores.

Cuando ocurre, se agrega la nota:

```text
expected_goals_normalized_from_score_matrix
```

Esto evita que hovers, tablas y scatterplots queden contaminados por valores fuera de escala.

## Dashboard

El scatterplot de xG ahora muestra:

- valores numericos en ambos ejes;
- diagonal gris de igualdad `xG local = xG visitante`;
- linea vertical dorada con el promedio de xG local de los modelos;
- linea horizontal dorada con el promedio de xG visitante de los modelos.

Ademas, como red de seguridad visual, el scatter ignora valores no finitos o mayores a 6.0.

## Validacion

Corrida actual:

```text
prediction_run_id = pred_20260608T173452Z_aba95b9d
```

Despues del saneamiento:

```text
max_total_xg = 3.47
max_single_xg = 2.84
Turkey vs United States / neural_scoreline_mlp = 1.00-0.00
```

## Estado

Activo. No cambia la matriz de probabilidad del modelo; solo evita que `expected_goals` reporte valores absurdos cuando la cabeza auxiliar de goles falla.

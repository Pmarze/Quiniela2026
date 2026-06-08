# 028 - Folds recientes sin 2014

## Estado

Esta nota reemplaza el criterio de `026_neural_hybrid_v2_folds_recientes.md` para los modelos neuronales y ajusta parcialmente el criterio de `022_neural_ola_2_tuning.md`.

## Conocimiento

Entrenar y tunear con 2014 puede estar sesgando los modelos neuronales hacia patrones menos utiles para el Mundial actual. El rendimiento observado sugiere que 2022 es el peor ano bajo la configuracion con tres folds, por lo que se prioriza calibrar con los dos Mundiales mas recientes: 2018 y 2022.

## Decision operativa

Los modelos `neural_scoreline_mlp` y `neural_hybrid_v2` deben usar:

```json
"validation_world_cups": [2018, 2022]
```

El backtest base tambien se mueve a:

```json
"world_cup_years": [2018, 2022]
```

La lectura correcta es: 2018 funciona como validacion historica reciente y 2022 queda como el test mas exigente para evitar que el modelo se vea bien por aprender demasiado de 2014.

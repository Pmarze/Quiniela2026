# 026 - Folds recientes para Neural Hybrid V2

## Conocimiento

Para `neural_hybrid_v2`, la validacion principal debe quedarse con los tres Mundiales mas recientes disponibles: 2014, 2018 y 2022.

## Razon

2010 aumenta la muestra, pero tambien introduce futbol mas antiguo y menos comparable con el comportamiento reciente de selecciones, torneos y marcadores. Para este proyecto, la prioridad es calibrar el modelo con contextos cercanos al Mundial 2026.

## Implementacion

`configs/neural_hybrid_v2.yaml` queda con:

```json
"validation_world_cups": [2014, 2018, 2022]
```

Esto no cambia la regla temporal `previous_day`: cada fold sigue usando solo informacion anterior al dia de cada partido evaluado.

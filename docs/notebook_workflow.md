# Workflow de Notebooks

## Regla Principal

Cada notebook tiene una responsabilidad. Ningun notebook depende de otro notebook. La comunicacion ocurre mediante artefactos guardados en `data/`.

## Notebooks Propuestos

### `00_data_audit.ipynb`

Objetivo:

- Revisar fuentes.
- Validar nombres de equipos.
- Detectar partidos duplicados o datos faltantes.

Salida:

- Reporte de calidad de datos.
- Snapshot procesado inicial.

### `01_model_elo_dixon_coles.ipynb`

Objetivo:

- Modelo base de goles usando Elo + Poisson + Dixon-Coles.

Salida:

- Predicciones por marcador.
- Probabilidades 1X2.
- Metadata del modelo.

### `02_model_market_calibrated_poisson.ipynb`

Objetivo:

- Tomar una matriz Poisson/Dixon-Coles y calibrarla con probabilidades externas.

Salida:

- Predicciones calibradas.
- Comparacion modelo puro vs mercado.

### `03_model_xg_or_stats_features.ipynb`

Objetivo:

- Explorar features de xG, tiros, goles recientes u otras estadisticas.

Salida:

- Modelo alternativo de expected goals o ajuste de lambdas.

### `04_model_ml_1x2.ipynb`

Objetivo:

- Modelo ML para probabilidades 1X2 usando rankings, forma, H2H y variables externas.

Salida:

- Probabilidades 1X2.
- Si no produce matriz de marcador, se adapta con un convertidor a matriz usando lambdas base.

### `10_ensemble_and_selection.ipynb`

Objetivo:

- Consumir N artefactos de modelos.
- Combinar predicciones.
- Seleccionar marcador que maximiza puntos esperados.

Salida:

- Pronostico final por partido.
- Ranking de confianza.

### `20_backtest_report.ipynb`

Objetivo:

- Evaluar modelos y ensemble.
- Comparar metricas.
- Identificar que modelos mantener, ajustar o eliminar.

Salida:

- Reporte de backtest.
- Recomendaciones de pesos por modelo.

## Convencion de Artefactos

Cada notebook de modelo debe guardar:

```text
data/predictions/{run_id}/{model_id}.parquet
data/predictions/{run_id}/{model_id}.metadata.json
```

Cada notebook debe leer configuracion desde:

```text
configs/project.yaml
configs/models.yaml
configs/scoring.yaml
```

## Ventaja Modular

Si quitamos un modelo:

1. Se desactiva en `configs/models.yaml`.
2. El ensemble deja de buscarlo.
3. El resto de notebooks no cambia.

Si agregamos un modelo:

1. Se crea su notebook.
2. Se implementa su modulo en `src/quiniela/models/`.
3. Se registra en `configs/models.yaml`.
4. El ensemble lo incorpora si produce el contrato estandar.


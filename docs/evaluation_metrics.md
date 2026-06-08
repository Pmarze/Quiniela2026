# Metricas de Evaluacion

## Prioridad de Quiniela

La evaluacion principal debe alinearse con la forma de ganar la quiniela:

1. Marcador exacto.
2. Empate o diferencia de goles.
3. Ganador.

## Metricas Operativas

### Exact Score Accuracy

Porcentaje de partidos donde el marcador recomendado coincide exactamente.

```text
pred_home_goals == actual_home_goals
pred_away_goals == actual_away_goals
```

### Margin or Draw Accuracy

Cuenta como acierto si:

- Ambos son empate, o
- La diferencia de goles predicha coincide con la diferencia real.

Ejemplo:

- Prediccion 2-1 y resultado 1-0: acierta diferencia.
- Prediccion 1-1 y resultado 0-0: acierta empate.

### Winner Accuracy

Porcentaje de partidos donde el resultado 1X2 es correcto:

- Gana equipo A.
- Empate.
- Gana equipo B.

## Metricas Probabilisticas

### Brier Score 1X2

Mide calidad de probabilidades para tres clases.

Bueno para comparar calibracion general.

### Log-Loss 1X2

Penaliza con fuerza predicciones muy confiadas y equivocadas.

Bueno para detectar sobreconfianza.

### Scoreline Log-Loss

Evalua la probabilidad asignada al marcador real.

Es importante porque la quiniela premia marcador exacto.

### Ranked Probability Score

Util para resultados ordenados o distribuciones acumuladas. Puede servir para comparar diferencias de goles.

### Expected Calibration Error

Evalua si las probabilidades publicadas corresponden con frecuencias reales.

Ejemplo:

- Si partidos con 60% de probabilidad de victoria ganan cerca de 60% de veces, el modelo esta calibrado.

## Metrica de Negocio: Puntos de Quiniela

Definir en `configs/scoring.yaml`:

```yaml
exact_score: 5
same_margin_or_draw: 3
winner: 1
```

Puntos esperados de un marcador candidato:

```text
EV(score) =
  P(exact_score) * exact_score_points
  + P(same_margin_or_draw_not_exact) * margin_points
  + P(winner_not_margin_not_exact) * winner_points
```

La prediccion recomendada debe maximizar `EV(score)`.

## Dos Estrategias de Prediccion

El sistema evalúa dos estrategias para elegir el marcador a jugar:

### max_points (operativa recomendada)

Usa `selected_score`: el marcador que maximiza los puntos esperados de quiniela
ponderados por la distribución completa del modelo. Puede diferir del más probable
cuando el esquema de puntos incentiva ciertos márgenes o empates.

### most_probable (referencia de calibración)

Usa `top_score`: el marcador con mayor probabilidad marginal P(i-j). Útil para
medir cuán bien calibra el modelo la distribución de marcadores, independientemente
del esquema de puntos.

El dashboard y el backtest reportan métricas separadas para cada estrategia.
La eficiencia principal (`points_efficiency`) usa `max_points`.

## Reporte Minimo por Modelo

```text
model_id
matches_evaluated
--- estrategia max_points ---
exact_score_accuracy
margin_or_draw_accuracy
winner_accuracy
mean_quiniela_points
points_efficiency           <- total_points / max_possible_points
--- estrategia most_probable ---
top_score_exact_accuracy
top_score_winner_accuracy
--- metricas probabilisticas ---
brier_1x2
log_loss_1x2
scoreline_log_loss
calibration_ece
```

## Backtest Historico

La validacion historica de Mundiales 2014, 2018 y 2022 se ejecuta con:

```powershell
python scripts\run_backtest.py
```

Regla temporal vigente:

```text
match_date < fecha_del_partido
```

Esta regla evita usar informacion futura. Es conservadora porque no usa partidos del mismo dia, aun si pudieron haber ocurrido antes.

Corrida de referencia activa: backtest sobre 2014+2018+2022, 192 partidos, 6 modelos.

Dashboard:

```powershell
python scripts\generate_validation_dashboard.py
```

Salida:

```text
outputs/validation_dashboard/index.html
```

## Mascara de Evaluacion

Los partidos sin equipos asignados todavia, por ejemplo eliminatorias con placeholders, no deben contar como errores del modelo.

Regla vigente:

```text
status = masked
is_evaluation_candidate = 0
```

La evaluacion debe consultar preferentemente:

```text
v_latest_evaluable_model_predictions
```

Esta vista excluye predicciones `masked` y `failed`.

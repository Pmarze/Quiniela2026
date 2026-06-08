# Propuesta final: Bayesian Monte Carlo Scoreline

## Decision propuesta

El ultimo modelo robusto que conviene agregar a la linea es:

```text
bayesian_monte_carlo_scoreline
```

La idea central es que sea un Monte Carlo real: no solo calcular una matriz deterministica y leer probabilidades, sino muestrear muchas versiones plausibles de la fuerza de cada equipo, sus tasas de gol y los marcadores de todos los partidos restantes.

## Por que este modelo

La busqueda de teoria y proyectos similares apunta a tres pilares:

- Maher (1982): los goles de futbol pueden modelarse razonablemente con Poisson de ataque/defensa, aunque con diferencias sistematicas pequenas.
- Dixon y Coles (1997): mejora el Poisson al ponderar recencia, modelar fuerza dinamica y corregir marcadores bajos como 0-0, 1-0, 0-1 y 1-1.
- Modelos bayesianos jerarquicos: permiten representar incertidumbre de parametros y actualizar predicciones dinamicamente durante el torneo.

Tambien aparecen implementaciones practicas 2026 con Elo + Dixon-Coles + Monte Carlo de torneo, como `Hicruben/world-cup-2026-prediction-model`, y referencias academicas/operativas como el algoritmo del Alan Turing Institute para Qatar 2022.

Fuentes clave:

- https://ideas.repec.org/a/bla/stanee/v36y1982i3p109-118.html
- https://cir.nii.ac.jp/crid/1363107370493446912?lang=en
- https://gianluca.statistica.it/research/football/
- https://www.turing.ac.uk/blog/can-our-algorithm-predict-winner-2022-football-world-cup
- https://arxiv.org/abs/1806.03208
- https://github.com/Hicruben/world-cup-2026-prediction-model

## Diferencia contra modelos actuales

Los modelos actuales son principalmente deterministas:

- estiman un rating o fuerza;
- convierten esa fuerza en goles esperados;
- generan una matriz de marcador;
- seleccionan `top_score` y `selected_score`.

Eso no esta mal, pero la incertidumbre del parametro queda escondida. En cambio, el Monte Carlo propuesto muestrea esa incertidumbre explicitamente:

```text
rating_equipo ~ distribucion posterior
ataque_equipo ~ distribucion posterior
defensa_equipo ~ distribucion posterior
lambda_goles ~ distribucion posterior
marcador ~ distribucion de conteo
torneo ~ repeticion de todos los partidos futuros
```

## Estructura del modelo

### 1. Capa de fuerza latente

Cada equipo tiene parametros latentes:

```text
attack_team
defense_team
tempo_team
rating_team
draw_tendency_team
```

Se inicializan con priors suaves desde fuentes backtesteables:

- Elo interno historico;
- rendimiento reciente;
- diferencia ataque/defensa historica;
- resultados historicos internacionales;
- estado del torneo observado antes del corte diario.

El modelo final no debe consumir Opta ni mercados externos en su version principal. La razon es que debe poder validarse de forma limpia contra 2018/2022. Si se quiere probar una variante con Opta o mercado, debe crearse otro `model_id` separado para que la comparacion no se mezcle.

### 2. Capa de goles esperados

Para cada partido:

```text
mu_a = base + attack_a - defense_b + host_adjustment + tournament_form_a
mu_b = base + attack_b - defense_a + host_adjustment + tournament_form_b
lambda_a = exp(mu_a)
lambda_b = exp(mu_b)
```

En cada simulacion se muestrean lambdas desde una distribucion posterior. Esto evita que dos equipos siempre tengan exactamente el mismo 1.42-1.08 de xG.

### 3. Capa de marcador

Opciones, en orden de preferencia:

1. Dixon-Coles Monte Carlo: Poisson independiente con ajuste de marcadores bajos.
2. Bivariate Poisson: agrega correlacion de goles.
3. Negative Binomial: permite sobredispersion si los marcadores reales son mas variables que Poisson.

La version inicial recomendada es Dixon-Coles Monte Carlo porque encaja mejor con el proyecto actual y tiene respaldo clasico.

### 4. Capa de torneo

Para cada path Monte Carlo:

1. Mantener fijos los partidos ya jugados.
2. Simular todos los partidos restantes de fase de grupos.
3. Construir tablas con criterios de desempate implementables.
4. Resolver cruces de eliminatoria.
5. Simular rondas hasta la final.
6. Guardar para cada partido la matriz empirica de marcadores generada por las simulaciones.

La prediccion por partido sale de agregar miles de paths:

```text
P(score) = frecuencia_score / simulaciones
P(1X2) = suma de scores por resultado
top_score = marcador con mayor frecuencia
selected_score = marcador que maximiza puntos esperados
```

## Actualizacion diaria durante el Mundial

La regla operativa del proyecto es:

```text
La quiniela se llena un dia antes, por lo que el corte solo puede usar partidos de dias anteriores.
```

Por eso el modelo debe usar:

```text
completed_matches WHERE kickoff_utc < as_of_utc
```

Despues de cada dia:

1. Se ingesta el resultado real.
2. Se congela el pick usado para la quiniela.
3. Se actualiza el estado del torneo.
4. El Monte Carlo recalcula fuerza de equipos usando el resultado observado.
5. Se simulan nuevamente los partidos futuros.

## Entrenamiento y optimizacion

Validacion limpia:

- folds: Mundial 2018 y Mundial 2022;
- corte por fecha exclusivo;
- ningun partido del dia evaluado entra al entrenamiento;
- metricas: puntos de quiniela, acierto exacto, acierto empate/diferencia, acierto 1X2, Brier 1X2, log loss marcador.

Hiperparametros iniciales:

```text
num_simulations: 20000
num_particles: 2000
rating_prior_weight: [0.3, 0.8]
dc_rho: [-0.18, 0.08]
recency_half_life_years: [3, 10]
world_cup_importance_weight: [1.0, 3.0]
qualifier_importance_weight: [0.8, 2.0]
friendly_importance_weight: [0.2, 0.8]
lambda_overdispersion: [0.0, 0.35]
```

Objetivo principal:

```text
maximizar puntos de quiniela
```

Guardrails:

- no degradar demasiado `winner_accuracy`;
- no concentrar todos los picks en favoritos 1-0;
- mantener calibracion 1X2 razonable por Brier/log loss.

## Implementacion recomendada

Carpetas:

```text
src/quiniela/models/bayesian_monte_carlo_scoreline.py
src/quiniela/simulation/tournament_monte_carlo.py
src/quiniela/simulation/score_sampling.py
configs/monte_carlo.yaml
notebooks/09_model_bayesian_monte_carlo_scoreline.ipynb
data/models/bayesian_monte_carlo_scoreline/
```

## Implementacion v0.1 lista para correr

La primera version ejecutable ya esta implementada en:

```text
src/quiniela/models/bayesian_monte_carlo_scoreline.py
```

Configuracion:

```text
configs/models.yaml -> bayesian_monte_carlo_scoreline
```

La v0.1 usa:

- Elo interno walk-forward;
- perfiles ataque/defensa con shrinkage;
- ruido de rating;
- ruido lognormal de lambdas;
- sobredispersion Gamma;
- muestreo Poisson de marcadores;
- correccion Dixon-Coles sobre la matriz empirica;
- semilla deterministica por partido.

No usa Opta ni mercados externos.

Iteraciones actuales:

```text
num_simulations: 20000
backtest_num_simulations: 5000
```

Para cambiar iteraciones:

```powershell
python scripts\set_monte_carlo_iterations.py --multiplier 2
python scripts\set_monte_carlo_iterations.py --num-simulations 50000 --backtest-num-simulations 10000
```

Salida:

Debe cumplir el mismo contrato de `ModelPrediction` para que el dashboard y ponderadores no cambien.

Artefactos adicionales:

```text
data/models/bayesian_monte_carlo_scoreline/latest/particles.parquet
data/models/bayesian_monte_carlo_scoreline/latest/team_strengths.parquet
data/models/bayesian_monte_carlo_scoreline/latest/tournament_paths.parquet
data/models/bayesian_monte_carlo_scoreline/latest/metrics.json
```

## Por que lo dejaria como ultimo modelo

Este modelo cubre el espacio que aun falta:

- incertidumbre explicita;
- simulacion de torneo completa;
- ajuste diario por resultados reales;
- independencia de Opta para permitir backtest limpio;
- salida por marcador para quiniela;
- posibilidad futura de crear variantes separadas con Opta o mercados externos.

Despues de este modelo, tiene mas sentido invertir en calibracion, pesos y calidad de datos que seguir agregando modelos independientes.

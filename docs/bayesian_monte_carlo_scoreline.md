# Modelo bayesian_monte_carlo_scoreline

## Objetivo

`bayesian_monte_carlo_scoreline` es el modelo Monte Carlo limpio del proyecto. No usa Opta ni mercados externos para que pueda evaluarse con backtest walk-forward en los Mundiales 2018 y 2022.

La version inicial ya es ejecutable con la informacion local existente:

- resultados historicos internacionales;
- pesos de importancia por torneo;
- recencia;
- sede/neutralidad;
- estado del torneo con corte `as_of_utc`;
- resultados reales del Mundial solo si ocurrieron antes del corte diario.

## Informacion minima requerida

El modelo necesita, por partido historico:

```text
match_date
team_a
team_b
home_score
away_score
tournament
country
neutral
importance_weight
recency_weight
is_world_cup
is_qualifier
is_friendly
```

Y por partido futuro:

```text
match_id
source_match_id
team_a
team_b
kickoff_utc
stage
group_name
stadium_country
status
```

## Fuentes utiles sin Opta

### Ya disponibles

- `martj42/international_results`: base historica principal de selecciones masculinas.
- `openfootball/worldcup.json`: fixtures y calendario del Mundial.
- Estado local en SQLite: resultados reales diarios cuando empiece el torneo.

### Recomendadas para mejorar despues

1. FIFA ranking historico.
   - Aporta una senal externa oficial y backtesteable por fecha.
   - Fuente candidata: `Dato-Futbol/fifa-ranking`.
   - Uso: feature/prior adicional solo si se guarda con fecha historica correcta.

2. StatsBomb open-data 2018 y 2022.
   - Aporta xG/eventos para Mundiales pasados.
   - Uso: calibrar si los goles reales de Mundiales recientes estuvieron por encima/debajo del xG; no usar como feature 2026 salvo que haya fuente equivalente diaria.

3. Fjelstul World Cup Database.
   - Aporta granularidad de Mundiales: sedes, arbitros, jugadores, alineaciones y estructura historica.
   - Uso: mejorar validacion y desempates/bracket historico, no necesariamente mejorar goles en v0.1.

4. World Football Elo historico.
   - Aporta ratings externos no Opta.
   - Uso: comparar contra Elo interno o usarlo como feature si se descarga en snapshots historicos por fecha.

## Logica del modelo v0.1

1. Ajusta un Elo interno walk-forward con el historico disponible.
2. Calcula un promedio global de goles por equipo.
3. Estima perfiles de ataque y defensa por equipo con shrinkage.
4. Para cada partido futuro calcula lambdas base:

```text
lambda_a = base_goals * exp(rating_term + attack_a + defense_b)
lambda_b = base_goals * exp(-rating_term + attack_b + defense_a)
```

5. En cada simulacion muestrea incertidumbre:

```text
rating_noise
lambda_log_noise
gamma_overdispersion
goals_a ~ Poisson(lambda_a_simulada)
goals_b ~ Poisson(lambda_b_simulada)
```

6. Construye matriz empirica de marcadores por frecuencia simulada.
7. Aplica correccion Dixon-Coles a marcadores bajos.
8. Publica el mismo contrato modular:

```text
top_score
selected_score
expected_goals_a
expected_goals_b
p_team_a_win
p_draw
p_team_b_win
score_matrix_json
```

## Iteraciones recomendadas

La configuracion v0.1 usa separacion de rating relativamente fuerte y ruido moderado para evitar que el Monte Carlo sobreproduzca empates:

```text
goal_scale = 0.70
rating_uncertainty_sd = 25.0
lambda_log_sigma = 0.10
lambda_overdispersion = 0.03
dixon_coles_rho = 0.0
```

Valores actuales:

```text
num_simulations = 20000
backtest_num_simulations = 5000
```

En ponderadores tiene `fallback_weight=0.25` hasta que exista backtest del modelo. Despues, los ensembles pueden tomar su peso desde `v_latest_backtest_model_metrics`.

Backtest vigente al implementarlo:

```text
backtest_run_id = backtest_wc2018_2022_base_20260608T045107Z_4811c40e
total_quiniela_points = 148 / 640
points_efficiency = 0.23125
exact_score_accuracy = 0.085938
winner_accuracy = 0.50000
draw_predictions = 16
```

Queda como candidato limpio y mejorable. No supera a `elo_poisson` ni a los mejores ponderadores en esta primera calibracion.

Recomendacion:

- Desarrollo rapido: 5,000 diaria / 1,000 backtest.
- Uso diario normal: 20,000 diaria / 5,000 backtest.
- Backtest serio/tuning final: 50,000 diaria / 10,000 a 20,000 backtest.
- Corrida final lenta: 100,000 diaria / 25,000 backtest.

Para la quiniela, 20,000 ya suele estabilizar bastante 1X2 y marcadores principales. Subir a 50,000 ayuda a reducir ruido en resultados exactos y `selected_score`; arriba de 100,000 probablemente hay rendimientos decrecientes.

## Cambiar iteraciones desde terminal

Con conda `quiniela2026` activo:

```powershell
python scripts\set_monte_carlo_iterations.py --multiplier 2
```

O fijar valores exactos:

```powershell
python scripts\set_monte_carlo_iterations.py --num-simulations 50000 --backtest-num-simulations 10000
```

Despues:

```powershell
python scripts\run_model.py --model bayesian_monte_carlo_scoreline
```

Para correr todo el pipeline:

```powershell
python scripts\run_model.py
python scripts\generate_dashboard.py
```

Para backtest:

```powershell
python scripts\run_backtest.py
python scripts\generate_validation_dashboard.py
```

## Fuentes revisadas

- https://github.com/martj42/international_results
- https://github.com/jfjelstul/worldcup
- https://github.com/Dato-Futbol/fifa-ranking
- https://github.com/statsbomb/open-data
- https://blogarchive.statsbomb.com/news/statsbomb-release-free-fifa-world-cup-data/
- https://blogarchive.statsbomb.com/news/statsbomb-release-free-2022-world-cup-data/
- https://github.com/Hicruben/world-cup-2026-prediction-model
- https://gianluca.statistica.it/research/football/

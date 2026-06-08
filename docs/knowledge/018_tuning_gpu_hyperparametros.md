# 018 - Tuning de Hiperparámetros con GPU (PyTorch)

## Contexto

Durante el tuning exhaustivo de Ola 1 (nota 017) se detectó que el script CPU
`scripts/tune_models.py` con `--workers 8` no producía output durante los primeros
12+ minutos porque cada worker recibe un chunk de ~243 trials y solo reporta al
terminar todo el chunk. Con un grid de 1944 trials esto significaba horas de espera.

Se implementó una versión GPU (`scripts/tune_models_gpu.py`) que corre **todos los
trials simultáneamente** sobre tensores PyTorch.

## Arquitectura GPU

### Idea central

En el tuning Elo secuencial, el cuello de botella es refitear los ratings para cada
combinación de parámetros × cada fecha de backtest. Con GPU:

- Los ratings de todos los trials se guardan en un tensor `[N_trials, N_teams]`
- Cada partido de entrenamiento actualiza los N_trials ratings en paralelo
- Para 1944 trials × 350 equipos: tensor de ~2.7 MB → cabe en cualquier GPU moderna

### Loop principal

```
Para cada fecha de backtest D (73 fechas en 2014/2018/2022):
  ratings = zeros(N_trials, N_teams) + 1500        # reset desde cero
  Para cada partido de entrenamiento (secuencial, ~3500):
    ra, rb = ratings[:, ia], ratings[:, ib]         # [N]
    expected_a = sigmoid(10^formula)                # [N] vectorizado
    delta = k_factors * combined_weight * gd_scale * (actual_a - expected_a)  # [N]
    delta *= include_mask[:, m]                     # aplica min_importance por trial
    ratings[:, ia] += delta                         # actualiza todos simultáneamente
    ratings[:, ib] -= delta

  Para cada partido WC en fecha D:
    lambda_a, lambda_b = f(ratings, goal_scales)    # [N]
    score_matrix = Poisson(lambda_a) ⊗ Poisson(lambda_b)  # [N, 9, 9]
    expected_pts = score_matrix @ points_matrix     # [N, 81] via matmul
    best_pick = argmax(expected_pts)                # [N]
    puntos += points_matrix[best_i, best_j, ah, ab] # [N]
```

### Por qué es rápido

El Elo es secuencial sobre partidos pero paralelo sobre trials. Con N=1944 trials,
cada operación dentro del loop vectoriza 1944 veces simultáneamente en CUDA. El
overhead de Python (~100 µs/iteración) sobre 219,000 iteraciones suma ~22s de
overhead de kernel launch, pero el cómputo real es insignificante.

### Componentes clave

**`_score_matrix_batch`** — PMF de Poisson para todos los trials a la vez:
```python
log_p_a = k * log(lambda_a) - lambda_a - lgamma(k+1)   # [N, G]
score_matrix = exp(log_p_a).unsqueeze(2) * exp(log_p_b).unsqueeze(1)  # [N, G, G]
```

**`points_matrix [G, G, G, G]`** — precomputada una vez. `pts[ci, cj, ai, aj]` =
puntos de quiniela por predecir (ci,cj) cuando el resultado real es (ai,aj).

**Selección de pick** via matmul:
```python
exp_pts = score_matrix.reshape(N, G*G) @ pts_flat.T  # [N, G^2]
best_idx = exp_pts.argmax(dim=1)                     # [N]
```

### Correcciones por modelo

- **elo_dixon_coles**: tau correction en celdas (0,0), (1,0), (0,1), (1,1) de la score_matrix
- **draw_specialist**: multiplica diagonal (draws) por `(1 + max_draw_boost)` y renormaliza
- **bradley_terry_davidson / attack_defense_poisson**: no soportados en GPU (usan `tune_models.py`)

## Hardware validado

- NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 13.1 driver, PyTorch instalado con cuda=12.1)
- El driver CUDA 13.1 es backward-compatible con runtimes CUDA 12.x de PyTorch

## Scripts

| Script | Qué hace |
|---|---|
| `scripts/tune_models_gpu.py` | Tuning GPU para elo_poisson, elo_dixon_coles, draw_specialist |
| `scripts/tune_all_gpu.bat` | Corre los 3 modelos GPU en secuencia |
| `scripts/tune_all_cpu_rest.bat` | Corre bradley_terry_davidson y attack_defense_poisson con 8 workers CPU |
| `scripts/tune_models.py` | Tuning CPU con ProcessPoolExecutor (todos los modelos) |

### Tiempo estimado (RTX 3050 6GB Laptop)

- elo_poisson (1944 trials): ~1-2 min
- elo_dixon_coles (1944 trials): ~1-2 min
- draw_specialist (1440 trials): ~1 min
- **Total GPU**: ~3-5 min
- bradley_terry_davidson + attack_defense_poisson (CPU): ~20 min adicionales

## Instalación PyTorch

```
# Verificar version de driver CUDA
nvidia-smi

# Instalar PyTorch (compatible con drivers CUDA 12.x o superior)
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Verificar instalacion
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Limitaciones del enfoque GPU

1. **Elo sigue siendo secuencial** por partido — solo se paraleliza sobre trials, no sobre partidos.
2. **home_advantage ignorado en predicciones WC** — los partidos de anfitrión reciben leve distorsión. Impacto menor ya que el tuning es relativo entre trials.
3. **base_goals** se computa sobre todos los partidos (igual para todos los trials con min_importance > 0), aceptable porque min_importance=0.0 siempre gana en práctica.
4. **bradley_terry_davidson** usa lógica de predicción diferente (no Poisson estándar); requiere implementación separada si se quiere GPU.

## Estado

Implementado 2026-06-05. Reemplaza el uso de `tune_all.bat` para los 3 modelos Elo.
Complementa nota 017.

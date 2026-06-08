# 038 - Model registry y datos reconstruibles

## Contexto

El usuario quiere colaborar con un amigo sin convertir el repositorio en un paquete pesado de corridas locales. La decision anterior de subir `data/`, predicciones, backtests y dashboards completos se reemplaza por una politica mas limpia.

## Decision

Se versiona:

- codigo
- configuracion
- documentacion
- inputs curados pequenos en `curated_inputs/`
- `model_registry/` con modelos finales publicados y metricas asociadas

No se versiona:

- `data/quiniela.db`
- datos descargables en `data/raw/`
- estados generados en `data/state/`
- backtests y predicciones generadas
- dashboards generados en `outputs/`
- checkpoints, folds, tuning y logs completos de entrenamientos locales

## Scripts nuevos

- `scripts/bootstrap_data.py`: reconstruye artefactos locales descargables/generados.
- `scripts/publish_model.py`: copia un modelo final desde una carpeta local a `model_registry/<model_id>/<version>/`.
- `docs/collaborator_onboarding.md`: guia completa para que un colaborador clone, reconstruya datos, corra dashboards y publique modelos.

## Manifiesto de artefactos

`configs/data_artifacts.json` lista que artefactos son descargables, generados o publicados, y que comando los reconstruye.

## Modelos publicados actuales

- `model_registry/neural_hybrid_v2/v2026-06-07`
- `model_registry/neural_scoreline_mlp/v2026-06-07`

Los `*.pt` dentro de `model_registry/` se manejan con Git LFS.

## Input curado actual

- `curated_inputs/opta/opta_power_ratings_20260607.json`

Se versiona porque es pequeno y permite reproducir `opta_power_poisson` sin depender de pedir un archivo manual.

## Estado

Activo. Reemplaza parcialmente las notas 036 y 037.

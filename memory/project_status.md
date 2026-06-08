---
name: project-status
description: "Current Quiniela2026 project state for future Codex sessions"
---

# Project Status - Quiniela2026

Current date of this status: 2026-06-08.

## Objective

Build a modular World Cup 2026 quiniela system that can be updated daily before matches.
The operational target is maximizing quiniela points, not only raw 1X2 accuracy.

Scoring priority:

- exact score
- draw or goal-difference/margin
- match winner

## Current Repository Policy

The repo is prepared for private GitHub collaboration.

Versioned:

- code in `src/` and `scripts/`
- configuration in `configs/`
- documentation in `docs/`
- memory handoff files in `memory/`
- curated small inputs in `curated_inputs/`
- final published models in `model_registry/`

Not versioned:

- downloaded/generated `data/`
- generated `outputs/`
- local training runs/checkpoints in `data/models/` and `data/models_local/`
- `.env`
- `.claude/settings.local.json`

Collaborators should start with `docs/collaborator_onboarding.md`.

## Runtime

Use Conda environment:

```powershell
conda activate quiniela2026
```

Do not hardcode a Python executable path in docs or scripts. Project commands should be run from the repository root using the active environment.

## Rebuilding Local Artifacts

Base local rebuild:

```powershell
python scripts\bootstrap_data.py --preset base
```

Full rebuild with backtest, predictions and dashboards:

```powershell
python scripts\bootstrap_data.py --preset all
```

Artifact manifest:

- `configs/data_artifacts.json`

## Data Sources

Configured public/downloadable sources:

- World Cup 2026 fixture and metadata sources in `configs/sources.json`
- historical international results in `configs/history_sources.json`

Curated versioned input:

- `curated_inputs/opta/opta_power_ratings_20260607.json`

API sources that require credentials remain disabled unless configured later.

## Active Model Lineup

Configured in `configs/models.yaml`.

Main families:

- baseline Poisson
- Elo Poisson
- Elo Dixon-Coles
- attack/defense Poisson
- draw specialist
- Bradley-Terry-Davidson
- Bayesian Monte Carlo scoreline
- Opta power Poisson
- neural scoreline MLP
- neural hybrid v2
- weighted ensembles
- calibrated scoreline ensemble

Current default quiniela model:

```text
weighted_points_ensemble
```

Before real 2026 results exist, the dashboard falls back to the points-oriented ensemble as the best-current model. Once real tournament results are present, the dashboard can rank models by current live performance.

## Published Models

Published neural models are stored in:

- `model_registry/neural_hybrid_v2/v2026-06-07`
- `model_registry/neural_scoreline_mlp/v2026-06-07`

Weights use Git LFS. After cloning:

```powershell
git lfs pull
```

Local training outputs should remain local until explicitly published with `scripts/publish_model.py`.

## Validation

Current backtest calibration focuses on 2018 and 2022 to avoid overfitting to older tournament behavior.

The ensemble optimizer uses saved score matrices from backtests and writes optimized weights into `configs/models.yaml`.

Key script:

```powershell
python scripts\optimize_ensemble_weights.py --iterations 8000
```

## Dashboard

The dashboard is a local offline HTML artifact generated from Python source/template files.

Important files:

- `src/quiniela/ui/dashboard.py`
- `src/quiniela/ui/dashboard_template.html`
- `docs/dashboard_reference.md`
- `memory/dashboard_status.md`

Generated files:

- `outputs/dashboard/index.html`
- `outputs/validation_dashboard/index.html`

Do not edit generated HTML directly. Regenerate it from scripts.

## Next Good Work Items

- Push the private GitHub repository and invite the collaborator.
- Have the collaborator run `docs/collaborator_onboarding.md` from a clean clone.
- Compare generated dashboards after both machines run `python scripts\bootstrap_data.py --preset all`.
- Publish only final model artifacts to `model_registry/` when collaborating on trained models.

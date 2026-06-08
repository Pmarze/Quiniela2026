---
name: dashboard-status
description: "Current unified dashboard status and editing guide"
---

# Dashboard Status - Quiniela2026

Current date of this status: 2026-06-08.

## State

The local dashboard is implemented as an offline HTML artifact generated from Python.
Generated dashboard files are not versioned and should be rebuilt locally.

Primary generated outputs:

- `outputs/dashboard/index.html`
- `outputs/validation_dashboard/index.html`

## Key Files

Source files:

- `src/quiniela/ui/dashboard.py`
- `src/quiniela/ui/dashboard_template.html`

Reference docs:

- `docs/dashboard_reference.md`
- `docs/ui_dashboard.md`
- `docs/ui_design_handoff_claude.md`

Prediction input used by the dashboard:

- `data/ui/prediction_overrides.json`

This file is generated locally and is not versioned.

## Regeneration

From the project root with `conda activate quiniela2026`:

```powershell
python scripts\generate_dashboard.py
python scripts\generate_validation_dashboard.py
```

Or as part of the full rebuild:

```powershell
python scripts\bootstrap_data.py --preset all
```

## Design And Interaction Notes

The match hover and fixed modal have two pages:

- stats page by default
- models page after navigating with the right arrow

The stats page shows:

- xG scatter by model
- scoreline consensus heatmap
- 1X2 probability bars/ranges

The models page shows individual model predictions, expected goals, 1X2 probabilities and EV/pick information.

## Real Results

When a match has a finished/completed status and recorded scores in the local database, the dashboard can show the real result and use it in live model ranking.

Before real 2026 results are available, the best-current model fallback is `weighted_points_ensemble`.

## Editing Rule

Do not edit generated HTML in `outputs/` directly.

Edit `src/quiniela/ui/dashboard.py` for data logic and `src/quiniela/ui/dashboard_template.html` for layout, CSS and JavaScript. Then regenerate the dashboards.

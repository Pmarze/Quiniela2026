import json
from pathlib import Path

files = [
    "data/backtests/tuning_elo_poisson_20260606T202256Z.json",
    "data/backtests/tuning_elo_dixon_coles_20260606T204512Z.json",
    "data/backtests/tuning_draw_specialist_20260606T210757Z.json",
    "data/backtests/tuning_bradley_terry_davidson_20260606T233208Z.json",
    "data/backtests/tuning_attack_defense_poisson_20260607T003448Z.json",
]

for f in files:
    d = json.loads(Path(f).read_text(encoding="utf-8"))
    print(f"\n{'='*60}")
    print(f"Modelo: {d['model_id']}  |  Anos: {d['years']}  |  Trials: {d['n_trials']}")
    bm = d['best_metrics']
    bp = d['best_params']
    print(f"MEJOR  eff={bm['points_efficiency']*100:.2f}%  exact={bm['exact_score_accuracy']*100:.2f}%  win={bm['winner_accuracy']*100:.2f}%  brier={bm['brier']:.4f}")
    print(f"PARAMS {bp}")
    print(f"TOP 3:")
    for i, r in enumerate(d['top_results'][:3], 1):
        m, p = r['metrics'], r['params']
        print(f"  {i}. eff={m['points_efficiency']*100:.2f}%  brier={m['brier']:.4f}  {p}")

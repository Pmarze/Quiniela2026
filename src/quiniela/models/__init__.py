"""Model runners for Quiniela2026 predictions."""

from quiniela.models.attack_defense_poisson import run_attack_defense_poisson
from quiniela.models.baseline_poisson import run_baseline_poisson
from quiniela.models.bayesian_monte_carlo_scoreline import run_bayesian_monte_carlo_scoreline
from quiniela.models.bradley_terry_davidson import run_bradley_terry_davidson
from quiniela.models.draw_specialist import run_draw_specialist
from quiniela.models.elo_dixon_coles import run_elo_dixon_coles
from quiniela.models.elo_poisson import run_elo_poisson
from quiniela.models.opta_power_poisson import run_opta_power_poisson
from quiniela.models.similar_match_knn_scoreline import run_similar_match_knn_scoreline

__all__ = [
    "run_attack_defense_poisson",
    "run_baseline_poisson",
    "run_bayesian_monte_carlo_scoreline",
    "run_bradley_terry_davidson",
    "run_draw_specialist",
    "run_elo_dixon_coles",
    "run_elo_poisson",
    "run_opta_power_poisson",
    "run_similar_match_knn_scoreline",
]

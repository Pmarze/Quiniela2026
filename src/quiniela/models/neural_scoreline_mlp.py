from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from quiniela.features.neural_features import build_prediction_features, score_matrix_from_probs
from quiniela.models.common import (
    ModelContext,
    ModelPrediction,
    failed_prediction,
    mask_reason_for_match,
    masked_prediction,
    successful_prediction_from_matrix,
)
from quiniela.scoring import select_best_score


MODEL_ID = "neural_scoreline_mlp"
_ARTIFACT_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


class NeuralScorelineMLP(nn.Module):
    def __init__(
        self,
        team_count: int,
        feature_count: int,
        team_embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        max_goals: int,
    ) -> None:
        super().__init__()
        self.max_goals = max_goals
        self.team_embedding = nn.Embedding(team_count, team_embedding_dim)
        input_dim = feature_count + team_embedding_dim * 2
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.scoreline_head = nn.Linear(hidden_dim // 2, (max_goals + 1) ** 2)
        self.outcome_head = nn.Linear(hidden_dim // 2, 3)
        self.goals_head = nn.Sequential(nn.Linear(hidden_dim // 2, 2), nn.Softplus())

    def forward(self, team_a: torch.Tensor, team_b: torch.Tensor, features: torch.Tensor) -> dict[str, torch.Tensor]:
        embedded = torch.cat([self.team_embedding(team_a), self.team_embedding(team_b), features], dim=1)
        hidden = self.body(embedded)
        return {
            "scoreline_logits": self.scoreline_head(hidden),
            "outcome_logits": self.outcome_head(hidden),
            "goals": self.goals_head(hidden),
        }


def run_neural_scoreline_mlp(
    context: ModelContext,
    model_config: dict[str, Any],
    scoring_config: dict[str, Any],
) -> list[ModelPrediction]:
    model_version = str(model_config.get("model_version", "0.1.0"))
    artifact_dir = _artifact_dir(context.db_path, model_config)
    metadata_path = artifact_dir / "metadata.json"
    weights_path = artifact_dir / "model.pt"
    if not metadata_path.exists() or not weights_path.exists():
        return [
            _unavailable_prediction(context, model_version, match, artifact_dir)
            for match in context.prediction_matches
        ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = _load_artifact(artifact_dir, metadata_path, weights_path, device)
    metadata = artifact["metadata"]
    model_version = str(metadata.get("model_version", model_version))
    max_goals = int(metadata["max_goals"])
    team_vocab = artifact["team_vocab"]
    feature_mean = artifact["feature_mean"]
    feature_std = artifact["feature_std"]
    model = artifact["model"]
    scoreline_temperature = artifact["scoreline_temperature"]

    examples_by_match_id = {
        example.match_id: example
        for example in build_prediction_features(context.training_matches, context.prediction_matches, team_vocab)
    }
    predictions: list[ModelPrediction] = []
    with torch.no_grad():
        for match in context.prediction_matches:
            mask_reason = mask_reason_for_match(match)
            if mask_reason:
                predictions.append(masked_prediction(context, MODEL_ID, model_version, match, mask_reason))
                continue
            example = examples_by_match_id.get(match.match_id)
            if example is None:
                predictions.append(
                    failed_prediction(context, MODEL_ID, model_version, match, "faltan features neuronales")
                )
                continue
            feature_tensor = (torch.tensor([example.features], dtype=torch.float32, device=device) - feature_mean) / feature_std
            team_a = torch.tensor([example.team_a_id], dtype=torch.long, device=device)
            team_b = torch.tensor([example.team_b_id], dtype=torch.long, device=device)
            output = model(team_a, team_b, feature_tensor)
            score_probs = torch.softmax(output["scoreline_logits"] / scoreline_temperature, dim=1).squeeze(0).cpu().tolist()
            goals = output["goals"].squeeze(0).cpu().tolist()
            score_matrix = score_matrix_from_probs(score_probs, max_goals)
            selected = select_best_score(score_matrix, scoring_config)
            predictions.append(
                successful_prediction_from_matrix(
                    context=context,
                    model_id=MODEL_ID,
                    model_version=model_version,
                    match=match,
                    lambda_a=float(goals[0]),
                    lambda_b=float(goals[1]),
                    score_matrix=score_matrix,
                    selected_score=selected["score"],
                    selected_expected_points=selected["expected_points"],
                    warnings=[],
                )
            )
    return predictions


def _artifact_dir(db_path: Path, model_config: dict[str, Any]) -> Path:
    raw = Path(str(model_config.get("artifact_dir", "data/models/neural_scoreline/latest")))
    if raw.is_absolute():
        return raw
    return db_path.resolve().parents[1] / raw


def _load_artifact(
    artifact_dir: Path,
    metadata_path: Path,
    weights_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    cache_key = (str(artifact_dir.resolve()), str(device))
    cached = _ARTIFACT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    team_vocab = dict(metadata["team_vocab"])
    model_params = metadata["model_params"]
    max_goals = int(metadata["max_goals"])
    model = NeuralScorelineMLP(
        team_count=len(team_vocab),
        feature_count=len(metadata["feature_columns"]),
        team_embedding_dim=int(model_params["team_embedding_dim"]),
        hidden_dim=int(model_params["hidden_dim"]),
        dropout=float(model_params["dropout"]),
        max_goals=max_goals,
    ).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    calibration = metadata.get("calibration", {})
    artifact = {
        "metadata": metadata,
        "team_vocab": team_vocab,
        "feature_mean": torch.tensor(metadata["feature_mean"], dtype=torch.float32, device=device),
        "feature_std": torch.tensor(metadata["feature_std"], dtype=torch.float32, device=device).clamp_min(1e-6),
        "scoreline_temperature": max(0.05, float(calibration.get("scoreline_temperature", 1.0))),
        "model": model,
    }
    _ARTIFACT_CACHE[cache_key] = artifact
    return artifact


def _unavailable_prediction(
    context: ModelContext,
    model_version: str,
    match: Any,
    artifact_dir: Path,
) -> ModelPrediction:
    mask_reason = mask_reason_for_match(match)
    if mask_reason:
        return masked_prediction(context, MODEL_ID, model_version, match, mask_reason)
    return failed_prediction(
        context=context,
        model_id=MODEL_ID,
        model_version=model_version,
        match=match,
        warning=f"artifacto neural no encontrado: {artifact_dir}",
    )

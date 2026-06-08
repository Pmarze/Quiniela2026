from __future__ import annotations

import torch
from torch.utils.data import Dataset

from quiniela.features.neural_features import NeuralExample


class NeuralScorelineDataset(Dataset):
    def __init__(self, examples: list[NeuralExample], feature_mean: torch.Tensor, feature_std: torch.Tensor) -> None:
        self.examples = examples
        self.feature_mean = feature_mean.float()
        self.feature_std = feature_std.float().clamp_min(1e-6)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        features = (torch.tensor(example.features, dtype=torch.float32) - self.feature_mean) / self.feature_std
        return {
            "team_a": torch.tensor(example.team_a_id, dtype=torch.long),
            "team_b": torch.tensor(example.team_b_id, dtype=torch.long),
            "features": features,
            "scoreline": torch.tensor(example.score_index, dtype=torch.long),
            "outcome": torch.tensor(example.outcome_index, dtype=torch.long),
            "goals": torch.tensor(example.goals, dtype=torch.float32),
            "weight": torch.tensor(example.weight, dtype=torch.float32),
        }


def feature_stats(examples: list[NeuralExample]) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = torch.tensor([example.features for example in examples], dtype=torch.float32)
    return matrix.mean(dim=0), matrix.std(dim=0).clamp_min(1e-6)

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from quiniela.features.hybrid_features import HYBRID_FEATURE_COLUMNS, build_hybrid_examples_previous_day
from quiniela.features.neural_features import build_team_vocabulary
from quiniela.models.common import load_json_config, utc_now
from quiniela.models.neural_hybrid_v2 import NeuralHybridV2
from quiniela.training.neural_dataset import NeuralScorelineDataset, feature_stats
from quiniela.training.neural_trainer import (
    _best_epoch,
    _evaluate,
    _fit_temperature,
    _read_log,
    _run_epoch,
    _save_checkpoint,
    _set_seed,
    _write_log,
    load_historical_records,
)


def train_neural_hybrid_v2(
    db_path: Path,
    config_path: Path,
    output_root: Path,
    device_name: str | None = None,
    folds_only: bool = False,
    final_only: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    config = load_json_config(config_path)
    training = config["training"]
    _set_seed(int(training.get("seed", 42)))
    print(f"[hybrid-v2] leyendo historico desde {db_path}", flush=True)
    records = load_historical_records(db_path)
    if not records:
        raise RuntimeError("No hay partidos historicos disponibles para entrenamiento.")
    max_goals = int(training.get("max_goals", 8))
    team_vocab = build_team_vocabulary(records)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"[hybrid-v2] registros={len(records):,} equipos={len(team_vocab):,} "
        f"device={device} cutoff=previous_day resume={'si' if resume else 'no'}",
        flush=True,
    )
    fold_years = [int(year) for year in training.get("validation_world_cups", [2014, 2018, 2022])]
    metrics: dict[str, Any] = {
        "model_id": config["model_id"],
        "model_version": config["model_version"],
        "created_at_utc": utc_now(),
        "device": str(device),
        "cutoff_strategy": "previous_day",
        "folds": [],
    }

    if not final_only:
        for year in fold_years:
            print(f"[hybrid-v2][fold {year}] construyendo ejemplos previous_day", flush=True)
            train_examples = build_hybrid_examples_previous_day(
                records,
                include_match=lambda row, y=year: row.match_date < f"{y}-01-01",
                team_vocab=team_vocab,
                max_goals=max_goals,
            )
            valid_examples = build_hybrid_examples_previous_day(
                records,
                include_match=lambda row, y=year: row.is_world_cup == 1 and row.match_date.startswith(str(y)),
                team_vocab=team_vocab,
                max_goals=max_goals,
            )
            print(f"[hybrid-v2][fold {year}] train={len(train_examples):,} valid={len(valid_examples):,}", flush=True)
            if not train_examples or not valid_examples:
                metrics["folds"].append({"year": year, "status": "skipped", "reason": "sin ejemplos suficientes"})
                continue
            result = _train_one(
                train_examples=train_examples,
                valid_examples=valid_examples,
                team_vocab=team_vocab,
                config=config,
                output_dir=output_root / f"fold_{year}",
                device=device,
                save_artifact=False,
                resume=resume,
                run_label=f"fold {year}",
            )
            metrics["folds"].append({"year": year, "status": "ok", **result["metrics"]})

    if not folds_only:
        print("[hybrid-v2][final] construyendo ejemplos finales previous_day", flush=True)
        final_examples = build_hybrid_examples_previous_day(
            records,
            include_match=lambda row: True,
            team_vocab=team_vocab,
            max_goals=max_goals,
        )
        split_index = max(1, int(len(final_examples) * 0.9))
        train_examples = final_examples[:split_index]
        valid_examples = final_examples[split_index:] or final_examples[-128:]
        result = _train_one(
            train_examples=train_examples,
            valid_examples=valid_examples,
            team_vocab=team_vocab,
            config=config,
            output_dir=output_root / "latest",
            device=device,
            save_artifact=True,
            resume=resume,
            run_label="final",
        )
        metrics["final"] = result["metrics"]
        metrics["artifact_dir"] = str(output_root / "latest")

    (output_root / "training_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def _train_one(
    train_examples: list[Any],
    valid_examples: list[Any],
    team_vocab: dict[str, int],
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    save_artifact: bool,
    resume: bool,
    run_label: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    max_goals = int(training.get("max_goals", 8))
    mean, std = feature_stats(train_examples)
    train_loader = DataLoader(
        NeuralScorelineDataset(train_examples, mean, std),
        batch_size=int(training.get("batch_size", 1024)),
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        NeuralScorelineDataset(valid_examples, mean, std),
        batch_size=int(training.get("batch_size", 1024)),
        shuffle=False,
        num_workers=0,
    )
    model = NeuralHybridV2(
        team_count=len(team_vocab),
        feature_count=len(HYBRID_FEATURE_COLUMNS),
        team_embedding_dim=int(training.get("team_embedding_dim", 64)),
        hidden_dim=int(training.get("hidden_dim", 512)),
        num_blocks=int(training.get("num_blocks", 4)),
        dropout=float(training.get("dropout", 0.2)),
        max_goals=max_goals,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 0.0007)),
        weight_decay=float(training.get("weight_decay", 0.0001)),
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=bool(training.get("mixed_precision", True)) and device.type == "cuda",
    )
    loss_weights = training.get("loss_weights", {})
    max_epochs = int(training.get("max_epochs", 180))
    patience = int(training.get("patience", 20))
    print_every = max(1, int(training.get("print_every_epochs", 1)))
    checkpoint_every = max(1, int(training.get("checkpoint_every_epochs", 1)))
    checkpoint_last = output_dir / "checkpoint_last.pt"
    checkpoint_best = output_dir / "checkpoint_best.pt"
    log_path = output_dir / "training_log.csv"
    log_rows = _read_log(log_path) if resume else []
    best_loss = math.inf
    stale_epochs = 0
    start_epoch = 1
    if resume and checkpoint_last.exists():
        checkpoint = torch.load(checkpoint_last, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scaler_state") and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_loss = float(checkpoint.get("best_loss", math.inf))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        print(f"[hybrid-v2][{run_label}] reanudando epoca {start_epoch}", flush=True)
    else:
        print(f"[hybrid-v2][{run_label}] inicio fresco", flush=True)

    for epoch in range(start_epoch, max_epochs + 1):
        started = time.perf_counter()
        train_loss = _run_epoch(model, train_loader, optimizer, scaler, device, loss_weights)
        valid = _evaluate(model, valid_loader, device, loss_weights)
        seconds = time.perf_counter() - started
        is_best = valid["loss"] < best_loss
        best_loss = min(best_loss, valid["loss"])
        stale_epochs = 0 if is_best else stale_epochs + 1
        row = {"epoch": epoch, "train_loss": train_loss, **valid, "best_loss": best_loss, "stale_epochs": stale_epochs, "seconds": round(seconds, 3)}
        log_rows.append(row)
        _write_log(log_path, log_rows)
        (output_dir / "metrics_live.json").write_text(
            json.dumps({"run_label": run_label, "epoch": epoch, "max_epochs": max_epochs, **row, "updated_at_utc": utc_now()}, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if epoch % checkpoint_every == 0 or is_best:
            _save_checkpoint(checkpoint_last, model, optimizer, scaler, epoch, best_loss, stale_epochs)
        if is_best:
            _save_checkpoint(checkpoint_best, model, optimizer, scaler, epoch, best_loss, stale_epochs)
        if epoch == start_epoch or epoch % print_every == 0 or is_best or stale_epochs >= patience:
            print(
                f"[hybrid-v2][{run_label}] epoca {epoch}/{max_epochs} {'best' if is_best else 'ok'} "
                f"train_loss={train_loss:.4f} valid_loss={valid['loss']:.4f} "
                f"exact={valid['exact_accuracy']:.3f} matrix_1x2={valid['matrix_outcome_accuracy']:.3f} "
                f"ev_pts={valid['ev_mean_points']:.3f} stale={stale_epochs}/{patience} tiempo={seconds:.1f}s",
                flush=True,
            )
        if stale_epochs >= patience:
            print(f"[hybrid-v2][{run_label}] early stopping por paciencia agotada", flush=True)
            break

    if checkpoint_best.exists():
        checkpoint = torch.load(checkpoint_best, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
    scoreline_temperature = _fit_temperature(model, valid_loader, device, "scoreline_logits", "scoreline")
    outcome_temperature = _fit_temperature(model, valid_loader, device, "outcome_logits", "outcome")
    final_metrics = _evaluate(model, valid_loader, device, loss_weights)
    final_metrics.update(
        {
            "train_examples": len(train_examples),
            "valid_examples": len(valid_examples),
            "best_epoch": _best_epoch(log_rows),
            "scoreline_temperature": scoreline_temperature,
            "outcome_temperature": outcome_temperature,
        }
    )
    (output_dir / "metrics.json").write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if save_artifact:
        torch.save(model.state_dict(), output_dir / "model.pt")
        metadata = {
            "model_id": config["model_id"],
            "model_version": config["model_version"],
            "created_at_utc": utc_now(),
            "cutoff_strategy": "previous_day",
            "feature_columns": HYBRID_FEATURE_COLUMNS,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "team_vocab": team_vocab,
            "max_goals": max_goals,
            "model_params": {
                "team_embedding_dim": int(training.get("team_embedding_dim", 64)),
                "hidden_dim": int(training.get("hidden_dim", 512)),
                "num_blocks": int(training.get("num_blocks", 4)),
                "dropout": float(training.get("dropout", 0.2)),
            },
            "calibration": {
                "scoreline_temperature": scoreline_temperature,
                "outcome_temperature": outcome_temperature,
            },
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[hybrid-v2][{run_label}] artefacto final guardado en {output_dir}", flush=True)
    print(
        f"[hybrid-v2][{run_label}] terminado loss={final_metrics['loss']:.4f} "
        f"exact={final_metrics['exact_accuracy']:.3f} matrix_1x2={final_metrics['matrix_outcome_accuracy']:.3f} "
        f"ev_pts={final_metrics['ev_mean_points']:.3f}",
        flush=True,
    )
    return {"metrics": final_metrics}

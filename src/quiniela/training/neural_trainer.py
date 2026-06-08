from __future__ import annotations

import csv
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from quiniela.features.neural_features import (
    FEATURE_COLUMNS,
    HistoricalMatchRecord,
    build_examples_online,
    build_team_vocabulary,
)
from quiniela.models.common import load_json_config, normalize_team_name, utc_now
from quiniela.models.neural_scoreline_mlp import NeuralScorelineMLP
from quiniela.storage.sqlite_store import SQLiteStore
from quiniela.training.neural_dataset import NeuralScorelineDataset, feature_stats


def train_neural_scoreline(
    db_path: Path,
    config_path: Path,
    output_root: Path,
    device_name: str | None = None,
    folds_only: bool = False,
    final_only: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    config = load_json_config(config_path)
    _set_seed(int(config["training"].get("seed", 42)))
    print(f"[neural] leyendo historico desde {db_path}", flush=True)
    records = load_historical_records(db_path)
    if not records:
        raise RuntimeError("No hay partidos historicos disponibles para entrenamiento.")
    max_goals = int(config["training"].get("max_goals", 8))
    team_vocab = build_team_vocabulary(records)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"[neural] registros={len(records):,} equipos={len(team_vocab):,} device={device} "
        f"resume={'si' if resume else 'no'}",
        flush=True,
    )
    fold_years = [int(year) for year in config["training"].get("validation_world_cups", [2018, 2022])]
    metrics: dict[str, Any] = {
        "model_id": config["model_id"],
        "model_version": config["model_version"],
        "created_at_utc": utc_now(),
        "device": str(device),
        "folds": [],
    }

    if not final_only:
        for year in fold_years:
            print(f"[neural][fold {year}] construyendo ejemplos train/valid sin fuga temporal", flush=True)
            train_examples = build_examples_online(
                records,
                include_match=lambda row, y=year: row.match_date < f"{y}-01-01",
                team_vocab=team_vocab,
                max_goals=max_goals,
            )
            valid_examples = build_examples_online(
                records,
                include_match=lambda row, y=year: row.is_world_cup == 1 and row.match_date.startswith(str(y)),
                team_vocab=team_vocab,
                max_goals=max_goals,
            )
            print(
                f"[neural][fold {year}] train={len(train_examples):,} valid={len(valid_examples):,}",
                flush=True,
            )
            if not train_examples or not valid_examples:
                metrics["folds"].append({"year": year, "status": "skipped", "reason": "sin ejemplos suficientes"})
                continue
            fold_dir = output_root / f"fold_{year}"
            print(f"[neural][fold {year}] entrenando en {fold_dir}", flush=True)
            result = _train_one(
                train_examples=train_examples,
                valid_examples=valid_examples,
                team_vocab=team_vocab,
                config=config,
                output_dir=fold_dir,
                device=device,
                save_artifact=False,
                resume=resume,
                run_label=f"fold {year}",
            )
            metrics["folds"].append({"year": year, "status": "ok", **result["metrics"]})

    if not folds_only:
        print("[neural][final] construyendo ejemplos finales", flush=True)
        final_examples = build_examples_online(
            records,
            include_match=lambda row: True,
            team_vocab=team_vocab,
            max_goals=max_goals,
        )
        split_index = max(1, int(len(final_examples) * 0.9))
        train_examples = final_examples[:split_index]
        valid_examples = final_examples[split_index:] or final_examples[-128:]
        final_dir = output_root / "latest"
        print(
            f"[neural][final] train={len(train_examples):,} valid={len(valid_examples):,} "
            f"artifacto={final_dir}",
            flush=True,
        )
        result = _train_one(
            train_examples=train_examples,
            valid_examples=valid_examples,
            team_vocab=team_vocab,
            config=config,
            output_dir=final_dir,
            device=device,
            save_artifact=True,
            resume=resume,
            run_label="final",
        )
        metrics["final"] = result["metrics"]
        metrics["artifact_dir"] = str(final_dir)

    metrics_path = output_root / "training_summary.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def load_historical_records(db_path: Path) -> list[HistoricalMatchRecord]:
    store = SQLiteStore(db_path)
    store.initialize()
    conn = store.conn
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM v_canonical_historical_matches
            ORDER BY match_date, historical_match_id
            """
        ).fetchall()
        return [_record_from_row(row) for row in rows]
    finally:
        store.close()


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
    train_dataset = NeuralScorelineDataset(train_examples, mean, std)
    valid_dataset = NeuralScorelineDataset(valid_examples, mean, std)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training.get("batch_size", 1024)),
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(training.get("batch_size", 1024)),
        shuffle=False,
        num_workers=0,
    )
    model = NeuralScorelineMLP(
        team_count=len(team_vocab),
        feature_count=len(FEATURE_COLUMNS),
        team_embedding_dim=int(training.get("team_embedding_dim", 24)),
        hidden_dim=int(training.get("hidden_dim", 192)),
        dropout=float(training.get("dropout", 0.2)),
        max_goals=max_goals,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 0.001)),
        weight_decay=float(training.get("weight_decay", 0.0001)),
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=bool(training.get("mixed_precision", True)) and device.type == "cuda",
    )
    loss_weights = training.get("loss_weights", {})
    best_loss = math.inf
    patience = int(training.get("patience", 25))
    stale_epochs = 0
    max_epochs = int(training.get("max_epochs", 300))
    print_every = max(1, int(training.get("print_every_epochs", 1)))
    checkpoint_every = max(1, int(training.get("checkpoint_every_epochs", 1)))
    checkpoint_last = output_dir / "checkpoint_last.pt"
    checkpoint_best = output_dir / "checkpoint_best.pt"
    log_path = output_dir / "training_log.csv"
    log_rows = _read_log(log_path) if resume else []
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
        print(
            f"[neural][{run_label}] reanudando desde epoca {start_epoch} "
            f"(best_loss={best_loss:.4f}, stale={stale_epochs}/{patience})",
            flush=True,
        )
    elif resume:
        print(f"[neural][{run_label}] sin checkpoint previo, iniciando desde cero", flush=True)
    else:
        print(f"[neural][{run_label}] inicio fresco; se ignoraran checkpoints previos", flush=True)

    for epoch in range(start_epoch, max_epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = _run_epoch(model, train_loader, optimizer, scaler, device, loss_weights)
        valid = _evaluate(model, valid_loader, device, loss_weights)
        seconds = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **valid,
            "seconds": round(seconds, 3),
        }
        log_rows.append(row)
        is_best = valid["loss"] < best_loss
        if valid["loss"] < best_loss:
            best_loss = valid["loss"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        row["best_loss"] = best_loss
        row["stale_epochs"] = stale_epochs
        _write_log(log_path, log_rows)
        snapshot_metrics = {
            "run_label": run_label,
            "epoch": epoch,
            "max_epochs": max_epochs,
            "best_loss": best_loss,
            "stale_epochs": stale_epochs,
            "patience": patience,
            **valid,
            "train_loss": train_loss,
            "seconds": round(seconds, 3),
            "updated_at_utc": utc_now(),
        }
        (output_dir / "metrics_live.json").write_text(
            json.dumps(snapshot_metrics, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if epoch % checkpoint_every == 0 or is_best:
            _save_checkpoint(
                path=checkpoint_last,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_loss=best_loss,
                stale_epochs=stale_epochs,
            )
        if is_best:
            _save_checkpoint(
                path=checkpoint_best,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_loss=best_loss,
                stale_epochs=stale_epochs,
            )
        if epoch == start_epoch or epoch % print_every == 0 or is_best or stale_epochs >= patience:
            marker = "best" if is_best else "ok"
            print(
                f"[neural][{run_label}] epoca {epoch}/{max_epochs} {marker} "
                f"train_loss={train_loss:.4f} valid_loss={valid['loss']:.4f} "
                f"exact={valid['exact_accuracy']:.3f} outcome={valid['outcome_accuracy']:.3f} "
                f"stale={stale_epochs}/{patience} tiempo={seconds:.1f}s",
                flush=True,
            )
        if stale_epochs >= patience:
            print(f"[neural][{run_label}] early stopping por paciencia agotada", flush=True)
            break
    if checkpoint_best.exists():
        best_checkpoint = torch.load(checkpoint_best, map_location=device, weights_only=True)
        model.load_state_dict(best_checkpoint["model_state"])
    scoreline_temperature = _fit_temperature(
        model=model,
        loader=valid_loader,
        device=device,
        logits_key="scoreline_logits",
        target_key="scoreline",
    )
    outcome_temperature = _fit_temperature(
        model=model,
        loader=valid_loader,
        device=device,
        logits_key="outcome_logits",
        target_key="outcome",
    )
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
    (output_dir / "metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if save_artifact:
        torch.save(model.state_dict(), output_dir / "model.pt")
        metadata = {
            "model_id": config["model_id"],
            "model_version": config["model_version"],
            "created_at_utc": utc_now(),
            "feature_columns": FEATURE_COLUMNS,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "team_vocab": team_vocab,
            "max_goals": max_goals,
            "model_params": {
                "team_embedding_dim": int(training.get("team_embedding_dim", 24)),
                "hidden_dim": int(training.get("hidden_dim", 192)),
                "dropout": float(training.get("dropout", 0.2)),
            },
            "calibration": {
                "scoreline_temperature": scoreline_temperature,
                "outcome_temperature": outcome_temperature,
            },
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"[neural][{run_label}] artefacto final guardado en {output_dir}", flush=True)
    print(
        f"[neural][{run_label}] terminado loss={final_metrics['loss']:.4f} "
        f"exact={final_metrics['exact_accuracy']:.3f} outcome={final_metrics['outcome_accuracy']:.3f}",
        flush=True,
    )
    return {"metrics": final_metrics}


def _run_epoch(
    model: NeuralScorelineMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    loss_weights: dict[str, float],
) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            output = model(batch["team_a"], batch["team_b"], batch["features"])
            loss = _loss(output, batch, loss_weights)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu()) * len(batch["team_a"])
        count += len(batch["team_a"])
    return total / max(1, count)


def _evaluate(
    model: NeuralScorelineMLP,
    loader: DataLoader,
    device: torch.device,
    loss_weights: dict[str, float],
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    exact = 0
    outcome = 0
    matrix_outcome = 0
    top_points = 0.0
    ev_points = 0.0
    ev_exact = 0
    ev_outcome = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            output = model(batch["team_a"], batch["team_b"], batch["features"])
            loss = _loss(output, batch, loss_weights)
            score_probs = torch.softmax(output["scoreline_logits"], dim=1)
            score_pred = score_probs.argmax(dim=1)
            outcome_pred = output["outcome_logits"].argmax(dim=1)
            max_goals = int(math.sqrt(score_probs.shape[1]) - 1)
            matrix_outcome_pred = _outcome_probs_from_scoreline(score_probs, max_goals).argmax(dim=1)
            quiniela = _quiniela_metrics(score_probs, batch["scoreline"])
            total_loss += float(loss.detach().cpu()) * len(batch["team_a"])
            exact += int((score_pred == batch["scoreline"]).sum().detach().cpu())
            outcome += int((outcome_pred == batch["outcome"]).sum().detach().cpu())
            matrix_outcome += int((matrix_outcome_pred == batch["outcome"]).sum().detach().cpu())
            top_points += float(quiniela["top_points"].detach().cpu())
            ev_points += float(quiniela["ev_points"].detach().cpu())
            ev_exact += int(quiniela["ev_exact"].detach().cpu())
            ev_outcome += int(quiniela["ev_outcome"].detach().cpu())
            count += len(batch["team_a"])
    return {
        "loss": total_loss / max(1, count),
        "exact_accuracy": exact / max(1, count),
        "outcome_accuracy": outcome / max(1, count),
        "matrix_outcome_accuracy": matrix_outcome / max(1, count),
        "top_mean_points": top_points / max(1, count),
        "ev_mean_points": ev_points / max(1, count),
        "ev_exact_accuracy": ev_exact / max(1, count),
        "ev_outcome_accuracy": ev_outcome / max(1, count),
    }


def _loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    score_weight = float(weights.get("scoreline", 0.55))
    outcome_weight = float(weights.get("outcome", 0.25))
    goals_weight = float(weights.get("goals", 0.15))
    calibration_weight = float(weights.get("calibration", 0.05))
    quiniela_weight = float(weights.get("quiniela_reward", 0.0))
    score_loss = F.cross_entropy(output["scoreline_logits"], batch["scoreline"], reduction="none")
    outcome_loss = F.cross_entropy(output["outcome_logits"], batch["outcome"], reduction="none")
    goals_loss = F.smooth_l1_loss(output["goals"], batch["goals"], reduction="none").mean(dim=1)
    score_probs = torch.softmax(output["scoreline_logits"], dim=1)
    max_goals = int(math.sqrt(score_probs.shape[1]) - 1)
    outcome_from_scores = _outcome_probs_from_scoreline(score_probs, max_goals)
    outcome_probs = torch.softmax(output["outcome_logits"], dim=1)
    calibration_loss = F.mse_loss(outcome_probs, outcome_from_scores.detach(), reduction="none").mean(dim=1)
    quiniela_loss = _quiniela_reward_loss(score_probs, batch["scoreline"], max_goals)
    combined = (
        score_weight * score_loss
        + outcome_weight * outcome_loss
        + goals_weight * goals_loss
        + calibration_weight * calibration_loss
        + quiniela_weight * quiniela_loss
    )
    return (combined * batch["weight"]).sum() / batch["weight"].sum().clamp_min(1e-6)


def _quiniela_reward_loss(
    score_probs: torch.Tensor,
    actual_score_index: torch.Tensor,
    max_goals: int,
) -> torch.Tensor:
    reward_matrix = _reward_matrix(max_goals, score_probs.device)
    rewards_for_actual = reward_matrix[:, actual_score_index].transpose(0, 1)
    expected_reward = (score_probs * rewards_for_actual).sum(dim=1)
    return 1.0 - expected_reward / 5.0


def _quiniela_metrics(score_probs: torch.Tensor, actual_score_index: torch.Tensor) -> dict[str, torch.Tensor]:
    max_goals = int(math.sqrt(score_probs.shape[1]) - 1)
    reward_matrix = _reward_matrix(max_goals, score_probs.device)
    candidate_ev = torch.matmul(score_probs, reward_matrix.transpose(0, 1))
    ev_pick = candidate_ev.argmax(dim=1)
    top_pick = score_probs.argmax(dim=1)
    top_points = reward_matrix[top_pick, actual_score_index].sum()
    ev_points = reward_matrix[ev_pick, actual_score_index].sum()
    ev_exact = (ev_pick == actual_score_index).sum()
    ev_outcome = (_score_outcome(ev_pick, max_goals) == _score_outcome(actual_score_index, max_goals)).sum()
    return {
        "top_points": top_points,
        "ev_points": ev_points,
        "ev_exact": ev_exact,
        "ev_outcome": ev_outcome,
    }


def _reward_matrix(max_goals: int, device: torch.device) -> torch.Tensor:
    side = max_goals + 1
    score_count = side * side
    idx = torch.arange(score_count, device=device)
    pred_a = idx // side
    pred_b = idx % side
    actual_a = pred_a.clone()
    actual_b = pred_b.clone()
    pred_diff = pred_a[:, None] - pred_b[:, None]
    actual_diff = actual_a[None, :] - actual_b[None, :]
    pred_outcome = _score_outcome(idx, max_goals)[:, None]
    actual_outcome = _score_outcome(idx, max_goals)[None, :]
    exact = idx[:, None] == idx[None, :]
    same_draw = (pred_outcome == 1) & (actual_outcome == 1)
    same_margin = (pred_outcome != 1) & (actual_outcome != 1) & (pred_diff == actual_diff)
    same_winner = pred_outcome == actual_outcome
    reward = torch.zeros((score_count, score_count), device=device)
    reward = torch.where(same_winner, torch.full_like(reward, 1.0), reward)
    reward = torch.where(same_draw | same_margin, torch.full_like(reward, 3.0), reward)
    reward = torch.where(exact, torch.full_like(reward, 5.0), reward)
    return reward


def _score_outcome(score_index: torch.Tensor, max_goals: int) -> torch.Tensor:
    side = max_goals + 1
    goals_a = score_index // side
    goals_b = score_index % side
    return torch.where(goals_a > goals_b, 0, torch.where(goals_a < goals_b, 2, 1))


def _outcome_probs_from_scoreline(score_probs: torch.Tensor, max_goals: int) -> torch.Tensor:
    side = max_goals + 1
    matrix = score_probs.reshape((-1, side, side))
    home = torch.tril(matrix, diagonal=-1).sum(dim=(1, 2))
    draw = torch.diagonal(matrix, dim1=1, dim2=2).sum(dim=1)
    away = torch.triu(matrix, diagonal=1).sum(dim=(1, 2))
    return torch.stack([home, draw, away], dim=1)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _fit_temperature(
    model: NeuralScorelineMLP,
    loader: DataLoader,
    device: torch.device,
    logits_key: str,
    target_key: str,
) -> float:
    logits = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            output = model(batch["team_a"], batch["team_b"], batch["features"])
            logits.append(output[logits_key].detach())
            targets.append(batch[target_key].detach())
    if not logits:
        return 1.0
    all_logits = torch.cat(logits, dim=0)
    all_targets = torch.cat(targets, dim=0)
    log_temperature = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature).clamp(0.2, 5.0)
        loss = F.cross_entropy(all_logits / temperature, all_targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return round(float(torch.exp(log_temperature).clamp(0.2, 5.0).detach().cpu()), 6)


def _write_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _best_epoch(rows: list[dict[str, Any]]) -> int | None:
    if not rows:
        return None
    best = min(rows, key=lambda row: float(row["loss"]))
    return int(float(best["epoch"]))


def _save_checkpoint(
    path: Path,
    model: NeuralScorelineMLP,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_loss: float,
    stale_epochs: int,
) -> None:
    payload = {
        "epoch": epoch,
        "best_loss": best_loss,
        "stale_epochs": stale_epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
        "saved_at_utc": utc_now(),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _record_from_row(row: sqlite3.Row) -> HistoricalMatchRecord:
    return HistoricalMatchRecord(
        match_id=row["historical_match_id"],
        match_date=row["match_date"],
        team_a_key=row["team_a_canonical_id"] or normalize_team_name(row["team_a_name"]),
        team_b_key=row["team_b_canonical_id"] or normalize_team_name(row["team_b_name"]),
        team_a_name=row["team_a_name"],
        team_b_name=row["team_b_name"],
        home_score=int(row["home_score"]),
        away_score=int(row["away_score"]),
        neutral=row["neutral"],
        tournament=row["tournament"],
        country=row["country"],
        is_world_cup=int(row["is_world_cup"] or 0),
        is_qualifier=int(row["is_qualifier"] or 0),
        is_friendly=int(row["is_friendly"] or 0),
        importance_weight=float(row["importance_weight"] or 1.0),
        recency_weight=float(row["recency_weight"] or 1.0),
        stage=_infer_historical_stage(row),
    )


def _infer_historical_stage(row: sqlite3.Row) -> str | None:
    tournament = normalize_team_name(row["tournament"])
    if not row["is_world_cup"]:
        return None
    if "final" in tournament:
        return "final"
    return "group"

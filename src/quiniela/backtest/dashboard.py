from __future__ import annotations

import html
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quiniela.backtest.runner import _ensure_backtest_schema
from quiniela.storage.sqlite_store import SQLiteStore


@dataclass(frozen=True)
class ValidationDashboardResult:
    output_path: Path
    backtest_run_id: str
    matches: int
    predictions: int
    models: int


def generate_validation_dashboard(
    db_path: Path,
    project_root: Path,
    output_path: Path | None = None,
) -> ValidationDashboardResult:
    store = SQLiteStore(db_path)
    store.initialize()
    conn = store.conn
    try:
        _ensure_backtest_schema(conn)
        run = _load_latest_backtest_run(conn)
        metrics = _load_latest_metrics(conn)
        predictions = _load_latest_predictions(conn)
        scoring = _load_scoring_config(project_root)
        neural_training = _load_neural_training_summary(project_root)
        payload = _build_payload(run, metrics, predictions, scoring, neural_training)
    finally:
        store.close()

    resolved_output = output_path or (project_root / "outputs" / "validation_dashboard" / "index.html")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(_render_html(payload), encoding="utf-8")
    return ValidationDashboardResult(
        output_path=resolved_output,
        backtest_run_id=run["backtest_run_id"],
        matches=run["matches_evaluated"],
        predictions=run["predictions"],
        models=len({row["model_id"] for row in metrics}),
    )


def _load_latest_backtest_run(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM v_latest_backtest_run").fetchone()
    if row is None:
        raise RuntimeError("No hay backtest vigente. Ejecuta scripts/run_backtest.py primero.")
    return dict(row)


def _load_latest_metrics(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM v_latest_backtest_model_metrics
        ORDER BY
          CASE WHEN year = 'all' THEN 0 ELSE 1 END,
          total_quiniela_points DESC,
          exact_hits DESC,
          model_id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _load_latest_predictions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM v_latest_backtest_predictions
        ORDER BY year, match_number, model_id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _load_scoring_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "configs" / "scoring.yaml"
    if not path.exists():
        return {"exact_score": 5, "same_margin_or_draw": 3, "winner": 1}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_neural_training_summary(project_root: Path) -> dict[str, Any] | None:
    summary_path = project_root / "data" / "models" / "neural_scoreline" / "training_summary.json"
    metrics_path = project_root / "data" / "models" / "neural_scoreline" / "latest" / "metrics.json"
    if not summary_path.exists() and not metrics_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    latest = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return {
        "model_id": summary.get("model_id", "neural_scoreline_mlp"),
        "model_version": summary.get("model_version", "0.1.0"),
        "created_at_utc": summary.get("created_at_utc"),
        "device": summary.get("device"),
        "folds": summary.get("folds", []),
        "final": summary.get("final", latest),
        "artifact_dir": summary.get("artifact_dir"),
    }


def _build_payload(
    run: dict[str, Any],
    metrics: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    scoring: dict[str, Any],
    neural_training: dict[str, Any] | None,
) -> dict[str, Any]:
    run_config = json.loads(run.get("config_json") or "{}")
    reference_models = sorted(set(run_config.get("reference_models") or []))
    all_metrics = _aggregate_prediction_metrics(predictions, scoring)
    years = sorted({int(row["year"]) for row in predictions})
    models = sorted({row["model_id"] for row in predictions})
    stages = sorted({row["stage"] for row in predictions})
    best_metric = all_metrics[0] if all_metrics else {}
    return {
        "run": run,
        "metrics": metrics,
        "all_metrics": all_metrics,
        "predictions": predictions,
        "scoring": scoring,
        "years": years,
        "models": models,
        "stages": stages,
        "reference_models": reference_models,
        "neural_training": neural_training,
        "summary": {
            "best_model": best_metric.get("model_id"),
            "best_perspective": best_metric.get("perspective"),
            "best_points": best_metric.get("total_quiniela_points", 0),
            "best_max_possible": best_metric.get("max_possible_points", 0),
            "best_efficiency": best_metric.get("points_efficiency", 0),
            "matches": run["matches_evaluated"],
            "predictions": run["predictions"],
        },
    }


def _aggregate_prediction_metrics(predictions: list[dict[str, Any]], scoring: dict[str, Any]) -> list[dict[str, Any]]:
    max_points_per_match = float(scoring.get("exact_score", 5))
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in predictions:
        for perspective in ("max_points", "most_probable"):
            buckets.setdefault((row["model_id"], perspective), []).append(row)
    metrics = []
    for (model_id, perspective), rows in buckets.items():
        points_key = "actual_points" if perspective == "max_points" else "top_actual_points"
        exact_key = "exact_hit" if perspective == "max_points" else "top_exact_hit"
        total_points = sum(float(row.get(points_key) or 0) for row in rows)
        exact_hits = sum(int(row.get(exact_key) or 0) for row in rows)
        max_possible = len(rows) * max_points_per_match
        metrics.append(
            {
                "model_id": model_id,
                "perspective": perspective,
                "matches_evaluated": len(rows),
                "exact_hits": exact_hits,
                "total_quiniela_points": round(total_points, 6),
                "max_possible_points": round(max_possible, 6),
                "points_efficiency": round(total_points / max_possible, 6) if max_possible else 0,
                "mean_quiniela_points": round(total_points / len(rows), 6) if rows else 0,
            }
        )
    return sorted(
        metrics,
        key=lambda row: (
            -float(row["total_quiniela_points"]),
            -int(row["exact_hits"]),
            str(row["model_id"]),
            str(row["perspective"]),
        ),
    )


def _render_html(payload: dict[str, Any]) -> str:
    safe_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = "Validacion Quiniela2026"
    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7f6;
      --panel: #ffffff;
      --ink: #1e2930;
      --muted: #65717b;
      --line: #d7dedb;
      --berry: #7b2546;
      --teal: #087e8b;
      --gold: #d6a328;
      --green: #247a57;
      --coral: #d95d39;
      --blue: #2f6f9f;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(8,126,139,0.05) 0 1px, transparent 1px),
        linear-gradient(0deg, rgba(123,37,70,0.04) 0 1px, transparent 1px),
        var(--bg);
      background-size: 40px 40px;
    }}

    .shell {{
      min-height: 100vh;
      padding: 16px;
    }}

    .topbar {{
      display: grid;
      grid-template-columns: minmax(320px, 1.2fr) minmax(360px, 1fr);
      gap: 12px;
      margin-bottom: 12px;
    }}

    .title-panel,
    .kpi,
    .panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: 0 14px 34px rgba(30, 41, 48, 0.09);
    }}

    .title-panel {{
      padding: 15px 16px;
    }}

    .eyebrow {{
      margin: 0 0 5px;
      color: var(--teal);
      font-size: 11px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0;
    }}

    h1 {{
      margin: 0;
      color: var(--berry);
      font-size: 30px;
      line-height: 1.08;
      letter-spacing: 0;
    }}

    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }}

    .kpis {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }}

    .kpi {{
      min-height: 82px;
      padding: 11px;
    }}

    .kpi strong {{
      display: block;
      font-size: 24px;
      line-height: 1;
    }}

    .kpi span {{
      display: block;
      margin-top: 8px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0;
    }}

    .layout {{
      display: grid;
      grid-template-columns: minmax(760px, 1.2fr) minmax(460px, 1fr);
      gap: 12px;
      align-items: start;
    }}

    .panel {{
      min-width: 0;
      overflow: hidden;
    }}

    .panel h2 {{
      margin: 0;
      padding: 10px 12px;
      color: #fff;
      background: var(--berry);
      font-size: 14px;
      letter-spacing: 0;
    }}

    .panel.alt h2 {{
      background: var(--teal);
    }}

    .panel-body {{
      padding: 12px;
    }}

    .controls {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }}

    label {{
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0;
    }}

    select {{
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 5px 8px;
      color: var(--ink);
      background: #fff;
      font: inherit;
      font-size: 12px;
      text-transform: none;
    }}

    .ranking {{
      display: grid;
      gap: 8px;
    }}

    .rank-row {{
      display: grid;
      grid-template-columns: 130px 94px minmax(120px, 1fr) repeat(5, minmax(62px, auto));
      gap: 8px;
      align-items: center;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fafbf9;
    }}

    .model-name {{
      min-width: 0;
      overflow: hidden;
      color: var(--berry);
      font-size: 12px;
      font-weight: 900;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .perspective {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 24px;
      padding: 3px 8px;
      border: 1px solid rgba(8,126,139,0.24);
      border-radius: 999px;
      color: var(--teal);
      background: #eef8f8;
      font-size: 10px;
      font-weight: 900;
      text-transform: uppercase;
      white-space: nowrap;
    }}

    .perspective.prob {{
      border-color: rgba(214,163,40,0.34);
      color: #7a5708;
      background: #fff8e5;
    }}

    .reference-badge {{
      display: inline-flex;
      align-items: center;
      min-height: 18px;
      margin-left: 4px;
      padding: 2px 6px;
      border: 1px solid rgba(214,163,40,0.42);
      border-radius: 999px;
      color: #7a5708;
      background: #fff8e5;
      font-size: 9px;
      font-weight: 900;
      text-transform: uppercase;
      white-space: nowrap;
    }}

    .reference-note {{
      margin: 0 0 10px;
      padding: 8px 10px;
      border: 1px solid rgba(214,163,40,0.45);
      border-radius: 8px;
      color: #5f4509;
      background: #fff8e5;
      font-size: 12px;
      line-height: 1.35;
    }}

    .bar-track {{
      height: 12px;
      overflow: hidden;
      border-radius: 999px;
      background: #e8ece7;
    }}

    .bar {{
      height: 100%;
      width: 0;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--teal), var(--gold));
    }}

    .points {{
      text-align: right;
      font-size: 12px;
      font-weight: 900;
      white-space: nowrap;
    }}

    .points small {{
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 800;
    }}

    .rank-stat {{
      text-align: right;
      color: var(--ink);
      font-size: 12px;
      font-weight: 900;
      white-space: nowrap;
    }}

    .rank-stat small {{
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 9px;
      font-weight: 900;
      text-transform: uppercase;
    }}

    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }}

    .metric-card {{
      min-height: 74px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: #f8faf7;
    }}

    .metric-card span {{
      display: block;
      color: var(--muted);
      font-size: 10px;
      font-weight: 900;
      text-transform: uppercase;
    }}

    .metric-card strong {{
      display: block;
      margin-top: 6px;
      font-size: 20px;
    }}

    .table-wrap {{
      min-width: 0;
      width: 100%;
      max-height: 560px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
    }}

    th,
    td {{
      padding: 7px 8px;
      border-bottom: 1px solid var(--line);
      font-size: 12px;
      text-align: right;
      white-space: nowrap;
    }}

    th:first-child,
    td:first-child,
    th:nth-child(2),
    td:nth-child(2) {{
      text-align: left;
    }}

    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      color: var(--muted);
      background: #eef2ef;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}

    .pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 34px;
      min-height: 22px;
      padding: 3px 7px;
      border-radius: 999px;
      color: #fff;
      background: var(--teal);
      font-size: 11px;
      font-weight: 900;
    }}

    .pill.zero {{ background: var(--coral); }}
    .pill.mid {{ background: var(--gold); color: #2e2608; }}
    .pill.high {{ background: var(--green); }}

    .small {{
      color: var(--muted);
      font-size: 11px;
    }}

    @media (max-width: 980px) {{
      .topbar,
      .layout,
      .controls,
      .rank-row {{
        grid-template-columns: 1fr;
      }}
      .kpis,
      .metric-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <section class="title-panel">
        <p class="eyebrow">Backtest walk-forward 2014/2018/2022</p>
        <h1>Validacion de modelos</h1>
        <div class="meta" id="meta"></div>
      </section>
      <section class="kpis" id="kpis"></section>
    </header>

    <main class="layout">
      <section class="panel">
        <h2>Ranking por puntos</h2>
        <div class="panel-body">
          <div class="controls">
            <label>Año
              <select id="yearFilter"></select>
            </label>
            <label>Modelo
              <select id="modelFilter"></select>
            </label>
            <label>Fase
              <select id="stageFilter"></select>
            </label>
            <label>Orden
              <select id="sortFilter">
                <option value="points">Puntos</option>
                <option value="exact">Exactos</option>
                <option value="winner">Ganador</option>
              </select>
            </label>
          </div>
          <div id="referenceNote"></div>
          <div class="ranking" id="ranking"></div>
        </div>
      </section>

      <section class="panel alt">
        <h2>Partidos evaluados</h2>
        <div class="panel-body">
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Partido</th>
                  <th>Modelo</th>
                  <th>Año</th>
                  <th>Fase</th>
                  <th>Real</th>
                  <th>Max puntos</th>
                  <th>Puntos</th>
                  <th>Mas probable</th>
                  <th>Puntos</th>
                  <th>1X2 Max</th>
                  <th>1X2 Prob</th>
                  <th>xG</th>
                </tr>
              </thead>
              <tbody id="predictionRows"></tbody>
            </table>
          </div>
        </div>
      </section>
    </main>
    <section class="panel alt neural-panel" id="neuralPanel" style="display:none; margin-top:12px;">
      <h2>Modelo neural entrenado</h2>
      <div class="panel-body" id="neuralBody"></div>
    </section>
  </div>

  <script>
    const DATA = {safe_json};
    const MAX_POINTS_PER_MATCH = Number(DATA.scoring?.exact_score || 5);

    function init() {{
      renderMeta();
      renderFilters();
      bindFilters();
      renderNeuralTraining();
      renderReferenceNote();
      render();
    }}

    function renderMeta() {{
      const run = DATA.run;
      document.getElementById("meta").innerHTML = `
        <span>run: <strong>${{escapeHtml(run.backtest_run_id)}}</strong></span>
        <span>histórico: <strong>${{escapeHtml(run.history_run_id)}}</strong></span>
        <span>corte: <strong>${{escapeHtml(run.as_of_utc)}}</strong></span>
      `;
      const kpis = [
        [bestLabel(DATA.summary.best_model, DATA.summary.best_perspective), "Mejor fila"],
        [`${{formatNumber(DATA.summary.best_points, 0)}} / ${{formatNumber(DATA.summary.best_max_possible, 0)}}`, "Puntos lider"],
        [formatPercent(DATA.summary.best_efficiency), "Eficiencia lider"],
        [DATA.summary.matches, "Partidos"],
      ];
      document.getElementById("kpis").innerHTML = kpis.map(([value, label]) => `
        <div class="kpi"><strong>${{escapeHtml(value)}}</strong><span>${{escapeHtml(label)}}</span></div>
      `).join("");
    }}

    function renderFilters() {{
      setOptions("yearFilter", ["all", ...DATA.years.map(String)]);
      setOptions("modelFilter", ["all", ...DATA.models]);
      setOptions("stageFilter", ["all", ...DATA.stages]);
    }}

    function setOptions(id, values) {{
      document.getElementById(id).innerHTML = values.map(value => `
        <option value="${{escapeHtml(value)}}">${{escapeHtml(value === "all" ? "Todos" : value)}}</option>
      `).join("");
    }}

    function bindFilters() {{
      ["yearFilter", "modelFilter", "stageFilter", "sortFilter"].forEach(id => {{
        document.getElementById(id).addEventListener("change", render);
      }});
    }}

    function render() {{
      renderRanking();
      renderPredictionsV2();
    }}

    function currentFilters() {{
      return {{
        year: document.getElementById("yearFilter").value,
        model: document.getElementById("modelFilter").value,
        stage: document.getElementById("stageFilter").value,
        sort: document.getElementById("sortFilter").value
      }};
    }}

    function filteredPredictions() {{
      const filters = currentFilters();
      return DATA.predictions.filter(row =>
        (filters.year === "all" || String(row.year) === filters.year) &&
        (filters.model === "all" || row.model_id === filters.model) &&
        (filters.stage === "all" || row.stage === filters.stage)
      );
    }}

    function aggregate(rows) {{
      const byModel = new Map();
      rows.forEach(row => {{
        for (const perspective of ["max_points", "most_probable"]) {{
          const key = `${{row.model_id}}::${{perspective}}`;
          if (!byModel.has(key)) {{
            byModel.set(key, {{
              model_id: row.model_id,
              perspective,
              matches: 0,
              points: 0,
              exact: 0,
              margin: 0,
              winner: 0,
              brier: 0,
              logloss: 0
            }});
          }}
          const item = byModel.get(key);
          const isMaxPoints = perspective === "max_points";
          item.matches += 1;
          item.points += Number(row[isMaxPoints ? "actual_points" : "top_actual_points"] || 0);
          item.exact += Number(row[isMaxPoints ? "exact_hit" : "top_exact_hit"] || 0);
          item.margin += Number(row[isMaxPoints ? "margin_or_draw_hit" : "top_margin_or_draw_hit"] || 0);
          item.winner += Number(row[isMaxPoints ? "winner_hit" : "top_winner_hit"] || 0);
          item.brier += Number(row.brier_1x2 || 0);
          item.logloss += Number(row.log_loss_1x2 || 0);
        }}
      }});
      return [...byModel.values()].map(item => ({{
        ...item,
        max_possible: item.matches * MAX_POINTS_PER_MATCH,
        efficiency: item.matches ? item.points / (item.matches * MAX_POINTS_PER_MATCH) : 0,
        mean_points: item.matches ? item.points / item.matches : 0,
        exact_rate: item.matches ? item.exact / item.matches : 0,
        margin_rate: item.matches ? item.margin / item.matches : 0,
        winner_rate: item.matches ? item.winner / item.matches : 0,
        brier: item.matches ? item.brier / item.matches : 0,
        logloss: item.matches ? item.logloss / item.matches : 0
      }}));
    }}

    function renderRanking() {{
      const filters = currentFilters();
      const rows = aggregate(filteredPredictions());
      const sorters = {{
        points: (a, b) => b.points - a.points || b.mean_points - a.mean_points || b.exact - a.exact,
        exact: (a, b) => b.exact - a.exact || b.exact_rate - a.exact_rate || b.points - a.points,
        winner: (a, b) => b.winner - a.winner || b.winner_rate - a.winner_rate || b.points - a.points
      }};
      rows.sort(sorters[filters.sort]);
      document.getElementById("ranking").innerHTML = rows.map(row => `
        <div class="rank-row">
          <span class="model-name">${{escapeHtml(row.model_id)}}${{referenceBadge(row.model_id)}}</span>
          <span class="perspective ${{row.perspective === "most_probable" ? "prob" : ""}}">${{perspectiveLabel(row.perspective)}}</span>
          <span class="bar-track"><span class="bar" style="width:${{Math.max(4, row.efficiency * 100)}}%"></span></span>
          <span class="rank-stat">${{formatNumber(row.points, 0)}}<small>Puntos</small></span>
          <span class="rank-stat">${{formatNumber(row.max_possible, 0)}}<small>Max</small></span>
          <span class="rank-stat">${{formatPercent(row.efficiency)}}<small>Efic.</small></span>
          <span class="rank-stat">${{row.exact}} / ${{row.matches}}<small>Exactos</small></span>
          <span class="rank-stat">${{formatNumber(row.mean_points, 3)}}<small>Media</small></span>
        </div>
      `).join("") || `<p class="small">Sin datos para estos filtros.</p>`;
    }}

    function renderPredictions() {{
      const rows = filteredPredictions().sort((a, b) =>
        Number(a.year) - Number(b.year) ||
        Number(a.match_number) - Number(b.match_number) ||
        String(a.model_id).localeCompare(String(b.model_id))
      );
      document.getElementById("predictionRows").innerHTML = rows.map(row => `
        <tr>
          <td>${{escapeHtml(row.team_a_name)}} vs ${{escapeHtml(row.team_b_name)}}<br><span class="small">#${{row.match_number}} · ${{escapeHtml(row.match_date)}}</span></td>
          <td>${{escapeHtml(row.model_id)}}${{referenceBadge(row.model_id)}}</td>
          <td>${{row.year}}</td>
          <td>${{escapeHtml(row.stage)}}</td>
          <td><span class="pill high">${{escapeHtml(row.actual_score)}}</span></td>
          <td><span class="${{pointsClass(row.actual_points)}}">${{escapeHtml(row.selected_score || "n/a")}}</span></td>
          <td>${{escapeHtml(row.top_score || "n/a")}}</td>
          <td><strong>${{formatNumber(row.actual_points, 0)}} / ${{formatNumber(MAX_POINTS_PER_MATCH, 0)}}</strong></td>
          <td>${{escapeHtml(row.selected_outcome || "n/a")}} → ${{escapeHtml(row.actual_outcome)}}</td>
          <td>${{formatNumber(row.expected_goals_a, 2)}}-${{formatNumber(row.expected_goals_b, 2)}}</td>
        </tr>
      `).join("") || `<tr><td colspan="10">Sin datos para estos filtros.</td></tr>`;
    }}

    function renderPredictionsV2() {{
      const rows = filteredPredictions().sort((a, b) =>
        Number(a.year) - Number(b.year) ||
        Number(a.match_number) - Number(b.match_number) ||
        String(a.model_id).localeCompare(String(b.model_id))
      );
      document.getElementById("predictionRows").innerHTML = rows.map(row => `
        <tr>
          <td>${{escapeHtml(row.team_a_name)}} vs ${{escapeHtml(row.team_b_name)}}<br><span class="small">#${{row.match_number}} · ${{escapeHtml(row.match_date)}}</span></td>
          <td>${{escapeHtml(row.model_id)}}${{referenceBadge(row.model_id)}}</td>
          <td>${{row.year}}</td>
          <td>${{escapeHtml(row.stage)}}</td>
          <td><span class="pill high">${{escapeHtml(row.actual_score)}}</span></td>
          <td><span class="${{pointsClass(row.actual_points)}}">${{escapeHtml(row.selected_score || "n/a")}}</span></td>
          <td><strong>${{formatNumber(row.actual_points, 0)}} / ${{formatNumber(MAX_POINTS_PER_MATCH, 0)}}</strong></td>
          <td><span class="${{pointsClass(row.top_actual_points)}}">${{escapeHtml(row.top_score || "n/a")}}</span></td>
          <td><strong>${{formatNumber(row.top_actual_points, 0)}} / ${{formatNumber(MAX_POINTS_PER_MATCH, 0)}}</strong></td>
          <td>${{escapeHtml(row.selected_outcome || "n/a")}} -> ${{escapeHtml(row.actual_outcome)}}</td>
          <td>${{escapeHtml(row.top_outcome || "n/a")}} -> ${{escapeHtml(row.actual_outcome)}}</td>
          <td>${{formatNumber(row.expected_goals_a, 2)}}-${{formatNumber(row.expected_goals_b, 2)}}</td>
        </tr>
      `).join("") || `<tr><td colspan="12">Sin datos para estos filtros.</td></tr>`;
    }}

    function renderNeuralTraining() {{
      const neural = DATA.neural_training;
      if (!neural) return;
      const panel = document.getElementById("neuralPanel");
      const final = neural.final || {{}};
      const folds = neural.folds || [];
      panel.style.display = "block";
      document.getElementById("neuralBody").innerHTML = `
        <div class="metric-grid">
          <div class="metric-card"><span>Modelo</span><strong>${{escapeHtml(neural.model_id || "neural_scoreline_mlp")}}</strong></div>
          <div class="metric-card"><span>Final exactos</span><strong>${{formatPercent(final.exact_accuracy)}}</strong></div>
          <div class="metric-card"><span>Final 1X2</span><strong>${{formatPercent(final.outcome_accuracy)}}</strong></div>
        </div>
        <p class="small">Estas metricas vienen del entrenamiento neural y no se mezclan con el ranking walk-forward para evitar fuga de informacion futura.</p>
        <div class="table-wrap" style="max-height:220px;">
          <table>
            <thead>
              <tr><th>Fold</th><th>Estado</th><th>Loss</th><th>Exactos</th><th>1X2</th><th>Train</th><th>Valid</th><th>Best epoch</th></tr>
            </thead>
            <tbody>
              ${{folds.map(row => `
                <tr>
                  <td>${{escapeHtml(row.year || "final")}}</td>
                  <td>${{escapeHtml(row.status || "ok")}}</td>
                  <td>${{formatNumber(row.loss, 4)}}</td>
                  <td>${{formatPercent(row.exact_accuracy)}}</td>
                  <td>${{formatPercent(row.outcome_accuracy)}}</td>
                  <td>${{escapeHtml(row.train_examples || "")}}</td>
                  <td>${{escapeHtml(row.valid_examples || "")}}</td>
                  <td>${{escapeHtml(row.best_epoch || "")}}</td>
                </tr>
              `).join("")}}
              <tr>
                <td><strong>final</strong></td>
                <td>ok</td>
                <td>${{formatNumber(final.loss, 4)}}</td>
                <td>${{formatPercent(final.exact_accuracy)}}</td>
                <td>${{formatPercent(final.outcome_accuracy)}}</td>
                <td>${{escapeHtml(final.train_examples || "")}}</td>
                <td>${{escapeHtml(final.valid_examples || "")}}</td>
                <td>${{escapeHtml(final.best_epoch || "")}}</td>
              </tr>
            </tbody>
          </table>
        </div>
      `;
    }}

    function renderReferenceNote() {{
      const models = DATA.reference_models || [];
      if (!models.length) return;
      document.getElementById("referenceNote").innerHTML = `
        <p class="reference-note">
          Modelos marcados como <strong>Referencia</strong>: ${{models.map(escapeHtml).join(", ")}}.
          Sus resultados se muestran para comparar comportamiento visualmente, pero no son backtest limpio porque el artefacto final puede haber visto informacion posterior.
        </p>
      `;
    }}

    function referenceBadge(modelId) {{
      return (DATA.reference_models || []).includes(modelId) ? ` <span class="reference-badge">Referencia</span>` : "";
    }}

    function bestLabel(modelId, perspective) {{
      if (!modelId) return "n/a";
      return `${{modelId}} · ${{perspectiveLabel(perspective)}}`;
    }}

    function perspectiveLabel(value) {{
      return value === "most_probable" ? "Mas probable" : "Max puntos";
    }}

    function pointsClass(points) {{
      const value = Number(points || 0);
      if (value >= 5) return "pill high";
      if (value >= 3) return "pill mid";
      if (value >= 1) return "pill";
      return "pill zero";
    }}

    function formatNumber(value, digits = 2) {{
      const number = Number(value);
      if (!Number.isFinite(number)) return "n/a";
      return number.toFixed(digits);
    }}

    function formatPercent(value) {{
      const number = Number(value);
      if (!Number.isFinite(number)) return "n/a";
      return `${{(number * 100).toFixed(1)}}%`;
    }}

    function escapeHtml(value) {{
      if (value === null || value === undefined) return "";
      return String(value).replace(/[&<>"']/g, char => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }}[char]));
    }}

    init();
  </script>
</body>
</html>
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Muestra un resumen de la base SQLite del proyecto.")
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "quiniela.db"),
        help="Ruta de la base SQLite.",
    )
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Incluye muestras de partidos y standings.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"No existe la base SQLite: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        print(f"database: {db_path}")
        for table in (
            "ingestion_runs",
            "data_snapshots",
            "teams",
            "stadiums",
            "matches",
            "group_standings",
            "history_ingestion_runs",
            "history_source_files",
            "canonical_historical_matches",
            "model_prediction_runs",
            "model_predictions",
            "tournament_state_runs",
            "state_matches",
            "state_group_tables",
            "state_team_form",
        ):
            count = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            print(f"{table}: {count}")

        print("\ncounts by source")
        for table in ("teams", "stadiums", "matches", "group_standings"):
            print(f"\n{table}")
            query = f"SELECT source_name, COUNT(*) AS n FROM {table} GROUP BY source_name ORDER BY source_name"
            for row in conn.execute(query):
                print(f"  {row['source_name']}: {row['n']}")

        latest_state = conn.execute("SELECT * FROM v_latest_tournament_state").fetchone()
        latest_canonical = conn.execute("SELECT * FROM v_latest_canonical_run").fetchone()
        latest_reconciliation = conn.execute("SELECT * FROM v_latest_reconciliation_run").fetchone()
        latest_history = conn.execute("SELECT * FROM v_latest_history_run").fetchone()
        latest_prediction_batch = conn.execute("SELECT * FROM v_latest_prediction_batch").fetchone()
        if latest_canonical:
            print("\nlatest canonical run")
            print(
                f"  {latest_canonical['canonical_run_id']} "
                f"as_of={latest_canonical['as_of_utc']} "
                f"teams={latest_canonical['teams']} "
                f"matches={latest_canonical['matches']}"
            )
        if latest_reconciliation:
            print("\nlatest reconciliation")
            print(
                f"  {latest_reconciliation['reconciliation_run_id']} "
                f"issues={latest_reconciliation['issues_found']} "
                f"sources={latest_reconciliation['sources_checked']}"
            )
            for row in conn.execute(
                """
                SELECT severity, source_name, issue_type, COUNT(*) AS n
                FROM v_latest_reconciliation_issues
                GROUP BY severity, source_name, issue_type
                ORDER BY source_name, issue_type
                """
            ):
                print(f"  {row['severity']} {row['source_name']} {row['issue_type']}: {row['n']}")
        if latest_state:
            print("\nlatest tournament state")
            print(
                f"  {latest_state['state_id']} "
                f"as_of={latest_state['as_of_utc']} "
                f"completed={latest_state['completed_matches']} "
                f"pending={latest_state['pending_matches']}"
            )
        if latest_history:
            print("\nlatest history run")
            print(
                f"  {latest_history['history_run_id']} "
                f"as_of={latest_history['as_of_utc']} "
                f"files={latest_history['files_downloaded']} "
                f"matches={latest_history['matches_imported']}"
            )
        if latest_prediction_batch:
            print("\nlatest prediction batch")
            print(
                f"  {latest_prediction_batch['prediction_run_id']} "
                f"created={latest_prediction_batch['created_at_utc']}"
            )
            for row in conn.execute(
                """
                SELECT model_id, predictions, successful_predictions, masked_predictions, failed_predictions
                FROM v_latest_model_prediction_runs
                ORDER BY model_id
                """
            ):
                print(
                    f"  {row['model_id']}: "
                    f"ok={row['successful_predictions']} "
                    f"masked={row['masked_predictions']} "
                    f"failed={row['failed_predictions']} "
                    f"total={row['predictions']}"
                )

        if args.samples:
            print("\nsample matches")
            for row in conn.execute(
                """
                SELECT source_match_id, stage, group_name, kickoff_local,
                       team_a_name, team_b_name, stadium_name, status
                FROM v_worldcup26_matches
                ORDER BY CAST(source_match_id AS INTEGER)
                LIMIT 8
                """
            ):
                print(dict(row))

            print("\nsample group standings")
            for row in conn.execute(
                """
                SELECT group_name, team_source_id, team_name, played, points,
                       goals_for, goals_against, goal_difference
                FROM v_worldcup26_group_standings
                ORDER BY group_name, CAST(team_source_id AS INTEGER)
                LIMIT 8
                """
            ):
                print(dict(row))

            if latest_state:
                state_id = latest_state["state_id"]
                print("\nsample state matches")
                for row in conn.execute(
                    """
                    SELECT match_number, stage, group_name, kickoff_local,
                           team_a_name, team_b_name, status, home_score, away_score
                    FROM state_matches
                    WHERE state_id = ?
                    ORDER BY match_number
                    LIMIT 8
                    """,
                    (state_id,),
                ):
                    print(dict(row))

                print("\nsample state group table")
                for row in conn.execute(
                    """
                    SELECT group_name, rank_sort, team_name, played, points,
                           goals_for, goals_against, goal_difference
                    FROM state_group_tables
                    WHERE state_id = ?
                    ORDER BY group_name, rank_sort
                    LIMIT 8
                    """,
                    (state_id,),
                ):
                    print(dict(row))

            if latest_history:
                print("\nsample historical training matches")
                for row in conn.execute(
                    """
                    SELECT match_date, team_a_name, team_b_name, home_score, away_score,
                           tournament, importance_weight, recency_weight
                    FROM v_model_training_matches
                    ORDER BY match_date DESC
                    LIMIT 8
                    """
                ):
                    print(dict(row))

            if latest_prediction_batch:
                print("\nsample model predictions")
                for row in conn.execute(
                    """
                    SELECT model_id, match_number, team_a_name, team_b_name,
                           expected_goals_a, expected_goals_b,
                           p_team_a_win, p_draw, p_team_b_win,
                           selected_score, selected_expected_points
                    FROM v_latest_model_predictions
                    WHERE status = 'ok'
                    ORDER BY model_id, match_number
                    LIMIT 8
                    """
                ):
                    print(dict(row))

                print("\nmasked model predictions")
                for row in conn.execute(
                    """
                    SELECT model_id, mask_reason, COUNT(*) AS n
                    FROM v_latest_model_predictions
                    WHERE status = 'masked'
                    GROUP BY model_id, mask_reason
                    ORDER BY model_id, mask_reason
                    """
                ):
                    print(dict(row))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

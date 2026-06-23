from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quiniela.data.snapshot import SnapshotRecord


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def initialize(self) -> None:
        try:
            self.conn.executescript(SCHEMA_SQL)
        except sqlite3.OperationalError as e:
            if "already exists" not in str(e):
                raise
            # Race condition: another process initialized concurrently — safe to ignore
        self._run_lightweight_migrations()
        self.conn.commit()

    def _run_lightweight_migrations(self) -> None:
        _ensure_columns(
            self.conn,
            "state_matches",
            {
                "canonical_match_id": "TEXT",
                "team_a_canonical_id": "TEXT",
                "team_b_canonical_id": "TEXT",
                "kickoff_local_iso": "TEXT",
                "kickoff_timezone": "TEXT",
                "kickoff_guatemala": "TEXT",
            },
        )
        _ensure_columns(
            self.conn,
            "canonical_matches",
            {
                "team_a_primary_source_id": "TEXT",
                "team_b_primary_source_id": "TEXT",
            },
        )
        _ensure_columns(
            self.conn,
            "model_prediction_runs",
            {
                "masked_predictions": "INTEGER NOT NULL DEFAULT 0",
            },
        )
        _ensure_columns(
            self.conn,
            "model_predictions",
            {
                "is_evaluation_candidate": "INTEGER NOT NULL DEFAULT 1",
                "mask_reason": "TEXT",
            },
        )

    def start_run(self, run_id: str, as_of_utc: str) -> None:
        now = _utc_now()
        self.conn.execute(
            """
            INSERT INTO ingestion_runs (run_id, as_of_utc, started_at_utc, status)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, as_of_utc, now, "running"),
        )
        self.conn.commit()

    def finish_run(self, run_id: str, status: str, notes: str) -> None:
        self.conn.execute(
            """
            UPDATE ingestion_runs
            SET completed_at_utc = ?, status = ?, notes = ?
            WHERE run_id = ?
            """,
            (_utc_now(), status, notes, run_id),
        )
        self.conn.commit()

    def insert_snapshot(self, snapshot: SnapshotRecord) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO data_snapshots (
                snapshot_id, run_id, source_name, resource_name, url, as_of_utc,
                downloaded_at_utc, content_sha256, content_type, byte_count,
                raw_path, metadata_path, http_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.snapshot_id,
                snapshot.run_id,
                snapshot.source_name,
                snapshot.resource_name,
                snapshot.url,
                snapshot.as_of_utc,
                snapshot.downloaded_at_utc,
                snapshot.content_sha256,
                snapshot.content_type,
                snapshot.byte_count,
                snapshot.raw_path,
                snapshot.metadata_path,
                snapshot.http_status,
            ),
        )
        self.conn.commit()

    def upsert_teams(self, rows: list[dict[str, Any]], run_id: str) -> None:
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO teams (
                source_name, source_team_id, fifa_code, name, group_name,
                payload_json, updated_run_id, updated_at_utc
            )
            VALUES (:source_name, :source_team_id, :fifa_code, :name, :group_name,
                :payload_json, :updated_run_id, :updated_at_utc)
            """,
            [_with_run(row, run_id) for row in rows if row.get("source_team_id") or row.get("name")],
        )
        self.conn.commit()

    def upsert_stadiums(self, rows: list[dict[str, Any]], run_id: str) -> None:
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO stadiums (
                source_name, source_stadium_id, name, fifa_name, city, country,
                capacity, payload_json, updated_run_id, updated_at_utc
            )
            VALUES (:source_name, :source_stadium_id, :name, :fifa_name, :city,
                :country, :capacity, :payload_json, :updated_run_id, :updated_at_utc)
            """,
            [_with_run(row, run_id) for row in rows if row.get("source_stadium_id") or row.get("name")],
        )
        self.conn.commit()

    def upsert_matches(self, rows: list[dict[str, Any]], run_id: str) -> None:
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO matches (
                source_name, source_match_id, match_number, stage, group_name,
                matchday, kickoff_local, kickoff_utc, team_a_source_id,
                team_b_source_id, team_a_name, team_b_name, home_team_label,
                away_team_label, stadium_source_id, home_score, away_score,
                status, finished, payload_json, updated_run_id, updated_at_utc
            )
            VALUES (
                :source_name, :source_match_id, :match_number, :stage, :group_name,
                :matchday, :kickoff_local, :kickoff_utc, :team_a_source_id,
                :team_b_source_id, :team_a_name, :team_b_name, :home_team_label,
                :away_team_label, :stadium_source_id, :home_score, :away_score,
                :status, :finished, :payload_json, :updated_run_id, :updated_at_utc
            )
            """,
            [_with_run(row, run_id) for row in rows if row.get("source_match_id")],
        )
        self.conn.commit()

    def upsert_group_standings(self, rows: list[dict[str, Any]], run_id: str) -> None:
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO group_standings (
                source_name, group_name, team_source_id, team_name, played,
                wins, draws, losses, points, goals_for, goals_against,
                goal_difference, payload_json, updated_run_id, updated_at_utc
            )
            VALUES (
                :source_name, :group_name, :team_source_id, :team_name, :played,
                :wins, :draws, :losses, :points, :goals_for, :goals_against,
                :goal_difference, :payload_json, :updated_run_id, :updated_at_utc
            )
            """,
            [_with_run(row, run_id) for row in rows if row.get("team_source_id") or row.get("team_name")],
        )
        self.conn.commit()


def print_database_summary(db_path: Path) -> None:
    if not db_path.exists():
        print(f"No existe la base SQLite: {db_path}")
        return
    conn = sqlite3.connect(db_path)
    try:
        print(f"database: {db_path}")
        for table in (
            "ingestion_runs",
            "data_snapshots",
            "canonical_build_runs",
            "canonical_teams",
            "canonical_matches",
            "reconciliation_runs",
            "reconciliation_issues",
            "history_ingestion_runs",
            "history_source_files",
            "canonical_historical_matches",
            "model_prediction_runs",
            "model_predictions",
            "teams",
            "stadiums",
            "matches",
            "group_standings",
            "tournament_state_runs",
            "state_matches",
            "state_group_tables",
            "state_team_form",
        ):
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table}: {count}")
    finally:
        conn.close()


def _with_run(row: dict[str, Any], run_id: str) -> dict[str, Any]:
    values = dict(row)
    values["updated_run_id"] = run_id
    values["updated_at_utc"] = _utc_now()
    return values


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_columns(conn: sqlite3.Connection, table_name: str, columns: dict[str, str]) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})")}
    for column_name, column_type in columns.items():
        if column_name not in existing:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id TEXT PRIMARY KEY,
    as_of_utc TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    status TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS data_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    resource_name TEXT NOT NULL,
    url TEXT NOT NULL,
    as_of_utc TEXT NOT NULL,
    downloaded_at_utc TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    content_type TEXT,
    byte_count INTEGER,
    raw_path TEXT NOT NULL,
    metadata_path TEXT NOT NULL,
    http_status INTEGER,
    FOREIGN KEY (run_id) REFERENCES ingestion_runs(run_id)
);

CREATE TABLE IF NOT EXISTS teams (
    source_name TEXT NOT NULL,
    source_team_id TEXT NOT NULL,
    fifa_code TEXT,
    name TEXT,
    group_name TEXT,
    payload_json TEXT,
    updated_run_id TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY (source_name, source_team_id)
);

CREATE TABLE IF NOT EXISTS stadiums (
    source_name TEXT NOT NULL,
    source_stadium_id TEXT NOT NULL,
    name TEXT,
    fifa_name TEXT,
    city TEXT,
    country TEXT,
    capacity INTEGER,
    payload_json TEXT,
    updated_run_id TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY (source_name, source_stadium_id)
);

CREATE TABLE IF NOT EXISTS matches (
    source_name TEXT NOT NULL,
    source_match_id TEXT NOT NULL,
    match_number INTEGER,
    stage TEXT,
    group_name TEXT,
    matchday TEXT,
    kickoff_local TEXT,
    kickoff_utc TEXT,
    team_a_source_id TEXT,
    team_b_source_id TEXT,
    team_a_name TEXT,
    team_b_name TEXT,
    home_team_label TEXT,
    away_team_label TEXT,
    stadium_source_id TEXT,
    home_score INTEGER,
    away_score INTEGER,
    status TEXT,
    finished INTEGER DEFAULT 0,
    payload_json TEXT,
    updated_run_id TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY (source_name, source_match_id)
);

CREATE INDEX IF NOT EXISTS idx_matches_stage ON matches(stage);
CREATE INDEX IF NOT EXISTS idx_matches_group ON matches(group_name);
CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status);
CREATE INDEX IF NOT EXISTS idx_matches_kickoff ON matches(kickoff_utc, kickoff_local);

CREATE TABLE IF NOT EXISTS group_standings (
    source_name TEXT NOT NULL,
    group_name TEXT NOT NULL,
    team_source_id TEXT NOT NULL,
    team_name TEXT,
    played INTEGER,
    wins INTEGER,
    draws INTEGER,
    losses INTEGER,
    points INTEGER,
    goals_for INTEGER,
    goals_against INTEGER,
    goal_difference INTEGER,
    payload_json TEXT,
    updated_run_id TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY (source_name, group_name, team_source_id)
);

CREATE INDEX IF NOT EXISTS idx_group_standings_group ON group_standings(group_name);

CREATE TABLE IF NOT EXISTS canonical_build_runs (
    canonical_run_id TEXT PRIMARY KEY,
    source_run_id TEXT NOT NULL,
    as_of_utc TEXT NOT NULL,
    primary_source_name TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    teams INTEGER NOT NULL,
    matches INTEGER NOT NULL,
    status TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY (source_run_id) REFERENCES ingestion_runs(run_id)
);

CREATE TABLE IF NOT EXISTS canonical_teams (
    canonical_team_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    fifa_code TEXT,
    group_name TEXT,
    primary_source_name TEXT NOT NULL,
    primary_source_team_id TEXT NOT NULL,
    aliases_json TEXT,
    canonical_run_id TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    FOREIGN KEY (canonical_run_id) REFERENCES canonical_build_runs(canonical_run_id)
);

CREATE INDEX IF NOT EXISTS idx_canonical_teams_group ON canonical_teams(group_name);
CREATE INDEX IF NOT EXISTS idx_canonical_teams_fifa ON canonical_teams(fifa_code);

CREATE TABLE IF NOT EXISTS canonical_matches (
    canonical_match_id TEXT PRIMARY KEY,
    match_number INTEGER NOT NULL,
    stage TEXT,
    group_name TEXT,
    matchday TEXT,
    primary_source_name TEXT NOT NULL,
    primary_source_match_id TEXT NOT NULL,
    kickoff_local_raw TEXT,
    kickoff_local_iso TEXT,
    kickoff_utc TEXT,
    kickoff_timezone TEXT,
    kickoff_guatemala TEXT,
    team_a_canonical_id TEXT,
    team_b_canonical_id TEXT,
    team_a_primary_source_id TEXT,
    team_b_primary_source_id TEXT,
    team_a_name TEXT,
    team_b_name TEXT,
    team_a_fifa_code TEXT,
    team_b_fifa_code TEXT,
    stadium_source_id TEXT,
    stadium_name TEXT,
    stadium_city TEXT,
    stadium_country TEXT,
    home_score INTEGER,
    away_score INTEGER,
    status TEXT,
    source_status TEXT,
    is_completed INTEGER NOT NULL DEFAULT 0,
    canonical_run_id TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    FOREIGN KEY (canonical_run_id) REFERENCES canonical_build_runs(canonical_run_id)
);

CREATE INDEX IF NOT EXISTS idx_canonical_matches_number ON canonical_matches(match_number);
CREATE INDEX IF NOT EXISTS idx_canonical_matches_stage ON canonical_matches(stage);
CREATE INDEX IF NOT EXISTS idx_canonical_matches_group ON canonical_matches(group_name);
CREATE INDEX IF NOT EXISTS idx_canonical_matches_kickoff ON canonical_matches(kickoff_utc);

CREATE TABLE IF NOT EXISTS reconciliation_runs (
    reconciliation_run_id TEXT PRIMARY KEY,
    canonical_run_id TEXT NOT NULL,
    as_of_utc TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    primary_source_name TEXT NOT NULL,
    sources_checked INTEGER NOT NULL,
    issues_found INTEGER NOT NULL,
    status TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY (canonical_run_id) REFERENCES canonical_build_runs(canonical_run_id)
);

CREATE TABLE IF NOT EXISTS reconciliation_issues (
    reconciliation_run_id TEXT NOT NULL,
    issue_id TEXT NOT NULL,
    severity TEXT NOT NULL,
    source_name TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    canonical_match_id TEXT,
    source_match_id TEXT,
    message TEXT NOT NULL,
    payload_json TEXT,
    created_at_utc TEXT NOT NULL,
    PRIMARY KEY (reconciliation_run_id, issue_id),
    FOREIGN KEY (reconciliation_run_id) REFERENCES reconciliation_runs(reconciliation_run_id)
);

CREATE INDEX IF NOT EXISTS idx_reconciliation_issues_type ON reconciliation_issues(issue_type);
CREATE INDEX IF NOT EXISTS idx_reconciliation_issues_source ON reconciliation_issues(source_name);

CREATE TABLE IF NOT EXISTS history_ingestion_runs (
    history_run_id TEXT PRIMARY KEY,
    as_of_utc TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    status TEXT NOT NULL,
    sources_checked INTEGER NOT NULL DEFAULT 0,
    files_downloaded INTEGER NOT NULL DEFAULT 0,
    matches_imported INTEGER NOT NULL DEFAULT 0,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS history_source_files (
    history_run_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    resource_name TEXT NOT NULL,
    url TEXT NOT NULL,
    downloaded_at_utc TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    byte_count INTEGER NOT NULL,
    row_count INTEGER,
    raw_path TEXT NOT NULL,
    status TEXT NOT NULL,
    PRIMARY KEY (history_run_id, source_id, resource_name),
    FOREIGN KEY (history_run_id) REFERENCES history_ingestion_runs(history_run_id)
);

CREATE INDEX IF NOT EXISTS idx_history_source_files_source ON history_source_files(source_id, resource_name);

CREATE TABLE IF NOT EXISTS canonical_historical_matches (
    historical_match_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_match_key TEXT NOT NULL,
    match_date TEXT NOT NULL,
    team_a_name TEXT NOT NULL,
    team_b_name TEXT NOT NULL,
    team_a_canonical_id TEXT,
    team_b_canonical_id TEXT,
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    tournament TEXT,
    city TEXT,
    country TEXT,
    neutral INTEGER,
    result_1x2 TEXT NOT NULL,
    goal_difference INTEGER NOT NULL,
    total_goals INTEGER NOT NULL,
    is_world_cup INTEGER NOT NULL DEFAULT 0,
    is_qualifier INTEGER NOT NULL DEFAULT 0,
    is_friendly INTEGER NOT NULL DEFAULT 0,
    importance_weight REAL NOT NULL DEFAULT 1.0,
    recency_weight REAL NOT NULL DEFAULT 1.0,
    history_run_id TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    PRIMARY KEY (history_run_id, historical_match_id),
    FOREIGN KEY (history_run_id) REFERENCES history_ingestion_runs(history_run_id)
);

CREATE INDEX IF NOT EXISTS idx_historical_matches_id ON canonical_historical_matches(historical_match_id);
CREATE INDEX IF NOT EXISTS idx_historical_matches_date ON canonical_historical_matches(match_date);
CREATE INDEX IF NOT EXISTS idx_historical_matches_source ON canonical_historical_matches(source_id);
CREATE INDEX IF NOT EXISTS idx_historical_matches_team_a ON canonical_historical_matches(team_a_canonical_id);
CREATE INDEX IF NOT EXISTS idx_historical_matches_team_b ON canonical_historical_matches(team_b_canonical_id);
CREATE INDEX IF NOT EXISTS idx_historical_matches_tournament ON canonical_historical_matches(tournament);

CREATE TABLE IF NOT EXISTS model_prediction_runs (
    prediction_run_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    as_of_utc TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    training_data_version TEXT,
    input_snapshot_id TEXT,
    tournament_state_id TEXT,
    predictions INTEGER NOT NULL DEFAULT 0,
    successful_predictions INTEGER NOT NULL DEFAULT 0,
    failed_predictions INTEGER NOT NULL DEFAULT 0,
    masked_predictions INTEGER NOT NULL DEFAULT 0,
    output_json_path TEXT,
    output_csv_path TEXT,
    status TEXT NOT NULL,
    notes TEXT,
    PRIMARY KEY (prediction_run_id, model_id)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_run_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    match_id TEXT NOT NULL,
    source_match_id TEXT,
    match_number INTEGER,
    team_a_name TEXT,
    team_b_name TEXT,
    kickoff_utc TEXT,
    tournament_state_id TEXT,
    expected_goals_a REAL,
    expected_goals_b REAL,
    p_team_a_win REAL,
    p_draw REAL,
    p_team_b_win REAL,
    score_matrix_json TEXT,
    top_score TEXT,
    top_score_probability REAL,
    selected_score TEXT,
    selected_expected_points REAL,
    status TEXT NOT NULL,
    is_evaluation_candidate INTEGER NOT NULL DEFAULT 1,
    mask_reason TEXT,
    warnings TEXT,
    created_at_utc TEXT NOT NULL,
    PRIMARY KEY (prediction_run_id, model_id, match_id),
    FOREIGN KEY (prediction_run_id, model_id) REFERENCES model_prediction_runs(prediction_run_id, model_id)
);

CREATE INDEX IF NOT EXISTS idx_model_predictions_match ON model_predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_model_predictions_model ON model_predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_model_predictions_source_match ON model_predictions(source_match_id);

CREATE TABLE IF NOT EXISTS tournament_state_runs (
    state_id TEXT PRIMARY KEY,
    source_run_id TEXT NOT NULL,
    as_of_utc TEXT NOT NULL,
    source_name TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    total_matches INTEGER NOT NULL,
    completed_matches INTEGER NOT NULL,
    pending_matches INTEGER NOT NULL,
    group_matches_completed INTEGER NOT NULL,
    teams INTEGER NOT NULL,
    groups INTEGER NOT NULL,
    status TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY (source_run_id) REFERENCES ingestion_runs(run_id)
);

CREATE TABLE IF NOT EXISTS state_matches (
    state_id TEXT NOT NULL,
    source_match_id TEXT NOT NULL,
    canonical_match_id TEXT,
    match_number INTEGER,
    stage TEXT,
    group_name TEXT,
    matchday TEXT,
    kickoff_local TEXT,
    kickoff_utc TEXT,
    kickoff_local_iso TEXT,
    kickoff_timezone TEXT,
    kickoff_guatemala TEXT,
    team_a_source_id TEXT,
    team_a_canonical_id TEXT,
    team_a_name TEXT,
    team_a_fifa_code TEXT,
    team_b_source_id TEXT,
    team_b_canonical_id TEXT,
    team_b_name TEXT,
    team_b_fifa_code TEXT,
    stadium_source_id TEXT,
    stadium_name TEXT,
    stadium_city TEXT,
    stadium_country TEXT,
    home_score INTEGER,
    away_score INTEGER,
    status TEXT,
    source_status TEXT,
    is_completed INTEGER NOT NULL DEFAULT 0,
    winner_1x2 TEXT,
    goal_difference INTEGER,
    team_a_points INTEGER,
    team_b_points INTEGER,
    PRIMARY KEY (state_id, source_match_id),
    FOREIGN KEY (state_id) REFERENCES tournament_state_runs(state_id)
);

CREATE INDEX IF NOT EXISTS idx_state_matches_state ON state_matches(state_id);
CREATE INDEX IF NOT EXISTS idx_state_matches_status ON state_matches(state_id, status);
CREATE INDEX IF NOT EXISTS idx_state_matches_group ON state_matches(state_id, group_name);

CREATE TABLE IF NOT EXISTS state_group_tables (
    state_id TEXT NOT NULL,
    group_name TEXT NOT NULL,
    team_source_id TEXT NOT NULL,
    team_name TEXT,
    fifa_code TEXT,
    rank_sort INTEGER,
    played INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    draws INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    points INTEGER NOT NULL,
    goals_for INTEGER NOT NULL,
    goals_against INTEGER NOT NULL,
    goal_difference INTEGER NOT NULL,
    PRIMARY KEY (state_id, group_name, team_source_id),
    FOREIGN KEY (state_id) REFERENCES tournament_state_runs(state_id)
);

CREATE INDEX IF NOT EXISTS idx_state_group_tables_group ON state_group_tables(state_id, group_name, rank_sort);

CREATE TABLE IF NOT EXISTS state_team_form (
    state_id TEXT NOT NULL,
    team_source_id TEXT NOT NULL,
    team_name TEXT,
    fifa_code TEXT,
    group_name TEXT,
    matches_played INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    draws INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    goals_for INTEGER NOT NULL,
    goals_against INTEGER NOT NULL,
    goal_difference INTEGER NOT NULL,
    last_match_number INTEGER,
    form_last5 TEXT,
    PRIMARY KEY (state_id, team_source_id),
    FOREIGN KEY (state_id) REFERENCES tournament_state_runs(state_id)
);

CREATE INDEX IF NOT EXISTS idx_state_team_form_group ON state_team_form(state_id, group_name);

DROP VIEW IF EXISTS v_worldcup26_matches;
CREATE VIEW v_worldcup26_matches AS
-- Selects the best row per match_number across all sources.
-- Priority: prefer rows that have actual scores, then worldcup26_ir > openfootball > rezarahiminia.
WITH ranked AS (
    SELECT
        m.*,
        ROW_NUMBER() OVER (
            PARTITION BY m.match_number
            ORDER BY
                CASE WHEN m.home_score IS NOT NULL AND m.away_score IS NOT NULL THEN 0 ELSE 1 END,
                CASE m.source_name
                    WHEN 'worldcup26_ir'            THEN 1
                    WHEN 'openfootball_worldcup_json' THEN 2
                    WHEN 'rezarahiminia_static_csv'  THEN 3
                    ELSE 9
                END,
                m.updated_at_utc DESC
        ) AS _rn
    FROM matches m
    WHERE m.match_number IS NOT NULL
      AND m.match_number BETWEEN 1 AND 104
)
SELECT
    m.source_match_id,
    m.match_number,
    m.stage,
    m.group_name,
    m.matchday,
    m.kickoff_local,
    m.kickoff_utc,
    m.team_a_source_id,
    COALESCE(m.team_a_name, ta.name, m.home_team_label) AS team_a_name,
    ta.fifa_code AS team_a_fifa_code,
    m.team_b_source_id,
    COALESCE(m.team_b_name, tb.name, m.away_team_label) AS team_b_name,
    tb.fifa_code AS team_b_fifa_code,
    m.home_team_label,
    m.away_team_label,
    m.stadium_source_id,
    s.name AS stadium_name,
    s.fifa_name AS stadium_fifa_name,
    s.city AS stadium_city,
    s.country AS stadium_country,
    m.home_score,
    m.away_score,
    m.status,
    m.finished,
    m.updated_run_id,
    m.updated_at_utc
FROM ranked m
LEFT JOIN teams ta
    ON ta.source_name = m.source_name
   AND ta.source_team_id = m.team_a_source_id
LEFT JOIN teams tb
    ON tb.source_name = m.source_name
   AND tb.source_team_id = m.team_b_source_id
LEFT JOIN stadiums s
    ON s.source_name = m.source_name
   AND s.source_stadium_id = m.stadium_source_id
WHERE m._rn = 1;

DROP VIEW IF EXISTS v_worldcup26_group_standings;
CREATE VIEW v_worldcup26_group_standings AS
SELECT
    gs.group_name,
    gs.team_source_id,
    COALESCE(gs.team_name, t.name) AS team_name,
    t.fifa_code,
    gs.played,
    gs.wins,
    gs.draws,
    gs.losses,
    gs.points,
    gs.goals_for,
    gs.goals_against,
    gs.goal_difference,
    gs.updated_run_id,
    gs.updated_at_utc
FROM group_standings gs
LEFT JOIN teams t
    ON t.source_name = gs.source_name
   AND t.source_team_id = gs.team_source_id
WHERE gs.source_name = 'worldcup26_ir';

DROP VIEW IF EXISTS v_latest_completed_run;
CREATE VIEW v_latest_completed_run AS
SELECT *
FROM ingestion_runs
WHERE status = 'completed'
ORDER BY completed_at_utc DESC
LIMIT 1;

DROP VIEW IF EXISTS v_latest_canonical_run;
CREATE VIEW v_latest_canonical_run AS
SELECT *
FROM canonical_build_runs
WHERE status = 'completed'
ORDER BY created_at_utc DESC
LIMIT 1;

DROP VIEW IF EXISTS v_canonical_teams;
CREATE VIEW v_canonical_teams AS
SELECT ct.*
FROM canonical_teams ct
JOIN v_latest_canonical_run lcr
  ON lcr.canonical_run_id = ct.canonical_run_id;

DROP VIEW IF EXISTS v_canonical_matches;
CREATE VIEW v_canonical_matches AS
SELECT cm.*
FROM canonical_matches cm
JOIN v_latest_canonical_run lcr
  ON lcr.canonical_run_id = cm.canonical_run_id;

DROP VIEW IF EXISTS v_latest_reconciliation_run;
CREATE VIEW v_latest_reconciliation_run AS
SELECT *
FROM reconciliation_runs
WHERE status = 'completed'
ORDER BY created_at_utc DESC
LIMIT 1;

DROP VIEW IF EXISTS v_latest_reconciliation_issues;
CREATE VIEW v_latest_reconciliation_issues AS
SELECT ri.*
FROM reconciliation_issues ri
JOIN v_latest_reconciliation_run lrr
  ON lrr.reconciliation_run_id = ri.reconciliation_run_id;

DROP VIEW IF EXISTS v_latest_history_run;
CREATE VIEW v_latest_history_run AS
SELECT *
FROM history_ingestion_runs
WHERE status = 'completed'
ORDER BY completed_at_utc DESC
LIMIT 1;

DROP VIEW IF EXISTS v_canonical_historical_matches;
CREATE VIEW v_canonical_historical_matches AS
SELECT chm.*
FROM canonical_historical_matches chm
JOIN v_latest_history_run lhr
  ON lhr.history_run_id = chm.history_run_id;

DROP VIEW IF EXISTS v_model_training_matches;
CREATE VIEW v_model_training_matches AS
SELECT
    historical_match_id,
    source_id,
    match_date,
    team_a_name,
    team_b_name,
    team_a_canonical_id,
    team_b_canonical_id,
    home_score,
    away_score,
    result_1x2,
    goal_difference,
    total_goals,
    tournament,
    city,
    country,
    neutral,
    is_world_cup,
    is_qualifier,
    is_friendly,
    importance_weight,
    recency_weight,
    importance_weight * recency_weight AS model_weight
FROM v_canonical_historical_matches
WHERE match_date < DATE('now');

DROP VIEW IF EXISTS v_team_rating_inputs;
CREATE VIEW v_team_rating_inputs AS
SELECT
    historical_match_id,
    match_date,
    team_a_name AS home_team_name,
    team_b_name AS away_team_name,
    team_a_canonical_id AS home_team_canonical_id,
    team_b_canonical_id AS away_team_canonical_id,
    home_score,
    away_score,
    result_1x2,
    tournament,
    neutral,
    importance_weight * recency_weight AS model_weight
FROM v_model_training_matches;

DROP VIEW IF EXISTS v_latest_prediction_batch;
CREATE VIEW v_latest_prediction_batch AS
SELECT prediction_run_id, MAX(created_at_utc) AS created_at_utc
FROM model_prediction_runs
WHERE status = 'completed'
GROUP BY prediction_run_id
ORDER BY created_at_utc DESC
LIMIT 1;

DROP VIEW IF EXISTS v_latest_model_prediction_runs;
CREATE VIEW v_latest_model_prediction_runs AS
SELECT mpr.*
FROM model_prediction_runs mpr
JOIN v_latest_prediction_batch lpb
  ON lpb.prediction_run_id = mpr.prediction_run_id;

DROP VIEW IF EXISTS v_latest_model_predictions;
CREATE VIEW v_latest_model_predictions AS
SELECT mp.*
FROM model_predictions mp
JOIN v_latest_prediction_batch lpb
  ON lpb.prediction_run_id = mp.prediction_run_id;

DROP VIEW IF EXISTS v_latest_evaluable_model_predictions;
CREATE VIEW v_latest_evaluable_model_predictions AS
SELECT *
FROM v_latest_model_predictions
WHERE status = 'ok'
  AND is_evaluation_candidate = 1;

DROP VIEW IF EXISTS v_latest_tournament_state;
CREATE VIEW v_latest_tournament_state AS
SELECT *
FROM tournament_state_runs
WHERE status = 'completed'
ORDER BY created_at_utc DESC
LIMIT 1;

DROP VIEW IF EXISTS v_latest_state_matches;
CREATE VIEW v_latest_state_matches AS
SELECT sm.*
FROM state_matches sm
JOIN v_latest_tournament_state lts
  ON lts.state_id = sm.state_id;

DROP VIEW IF EXISTS v_latest_state_group_tables;
CREATE VIEW v_latest_state_group_tables AS
SELECT sgt.*
FROM state_group_tables sgt
JOIN v_latest_tournament_state lts
  ON lts.state_id = sgt.state_id;

DROP VIEW IF EXISTS v_latest_state_team_form;
CREATE VIEW v_latest_state_team_form AS
SELECT stf.*
FROM state_team_form stf
JOIN v_latest_tournament_state lts
  ON lts.state_id = stf.state_id;
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quiniela.data.http_client import fetch_url
from quiniela.data.normalizers import normalize_payload
from quiniela.data.snapshot import write_snapshot
from quiniela.storage.sqlite_store import SQLiteStore


@dataclass
class PipelineResult:
    run_id: str
    as_of_utc: str
    snapshots_written: int = 0
    errors: list[str] = field(default_factory=list)


def run_download_pipeline(
    config_path: Path,
    db_path: Path,
    project_root: Path,
    as_of_utc: str | None = None,
    source_filter: set[str] | None = None,
) -> PipelineResult:
    as_of = as_of_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    run_id = f"run_{_compact_timestamp(as_of)}_{uuid.uuid4().hex[:8]}"
    result = PipelineResult(run_id=run_id, as_of_utc=as_of)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    store = SQLiteStore(db_path)
    store.initialize()
    store.start_run(run_id=run_id, as_of_utc=as_of)

    try:
        for source in config.get("sources", []):
            source_name = source.get("source_name")
            if not source.get("enabled", True):
                continue
            if source_filter and source_name not in source_filter:
                continue
            for resource in source.get("resources", []):
                try:
                    _download_resource(
                        store=store,
                        project_root=project_root,
                        run_id=run_id,
                        as_of_utc=as_of,
                        source_name=source_name,
                        resource=resource,
                    )
                    result.snapshots_written += 1
                except Exception as exc:  # noqa: BLE001 - keep pipeline alive per-source.
                    result.errors.append(f"{source_name}/{resource.get('resource_name')}: {exc}")
        store.finish_run(run_id=run_id, status="failed" if result.errors else "completed", notes="")
    except Exception:
        store.finish_run(run_id=run_id, status="failed", notes="pipeline exception")
        raise
    finally:
        store.close()

    return result


def _download_resource(
    store: SQLiteStore,
    project_root: Path,
    run_id: str,
    as_of_utc: str,
    source_name: str,
    resource: dict[str, Any],
) -> None:
    resource_name = resource["resource_name"]
    url = resource["url"]
    normalizer = resource.get("normalizer", "")
    response = fetch_url(url)
    if response.status >= 400:
        raise RuntimeError(f"HTTP {response.status}")

    snapshot = write_snapshot(
        project_root=project_root,
        run_id=run_id,
        as_of_utc=as_of_utc,
        source_name=source_name,
        resource_name=resource_name,
        url=url,
        content_type=response.content_type,
        body=response.body,
        http_status=response.status,
    )
    store.insert_snapshot(snapshot)

    batch = normalize_payload(normalizer, response.body, source_name=source_name)
    store.upsert_teams(batch.teams, run_id=run_id)
    store.upsert_stadiums(batch.stadiums, run_id=run_id)
    store.upsert_matches(batch.matches, run_id=run_id)
    store.upsert_group_standings(batch.group_standings, run_id=run_id)


def _compact_timestamp(value: str) -> str:
    return (
        value.replace("-", "")
        .replace(":", "")
        .replace(".", "")
        .replace("+0000", "Z")
        .replace("+00:00", "Z")
    )


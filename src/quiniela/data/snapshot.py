from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    run_id: str
    source_name: str
    resource_name: str
    url: str
    as_of_utc: str
    downloaded_at_utc: str
    content_sha256: str
    content_type: str
    byte_count: int
    raw_path: str
    metadata_path: str
    http_status: int


def write_snapshot(
    project_root: Path,
    run_id: str,
    as_of_utc: str,
    source_name: str,
    resource_name: str,
    url: str,
    content_type: str,
    body: bytes,
    http_status: int,
) -> SnapshotRecord:
    downloaded_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    digest = hashlib.sha256(body).hexdigest()
    snapshot_id = f"{source_name}_{resource_name}_{digest[:12]}"
    as_of_dir = _safe_path_part(as_of_utc)
    snapshot_dir = project_root / "data" / "raw" / "snapshots" / source_name / as_of_dir
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    suffix = _payload_suffix(content_type, resource_name)
    raw_path = snapshot_dir / f"{resource_name}.{suffix}.gz"
    metadata_path = snapshot_dir / f"{resource_name}.metadata.json"

    with gzip.open(raw_path, "wb") as fh:
        fh.write(body)

    metadata = {
        "snapshot_id": snapshot_id,
        "run_id": run_id,
        "source_name": source_name,
        "resource_name": resource_name,
        "url": url,
        "as_of_utc": as_of_utc,
        "downloaded_at_utc": downloaded_at,
        "content_sha256": f"sha256:{digest}",
        "content_type": content_type,
        "byte_count": len(body),
        "raw_path": str(raw_path.relative_to(project_root)),
        "http_status": http_status,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return SnapshotRecord(
        snapshot_id=snapshot_id,
        run_id=run_id,
        source_name=source_name,
        resource_name=resource_name,
        url=url,
        as_of_utc=as_of_utc,
        downloaded_at_utc=downloaded_at,
        content_sha256=f"sha256:{digest}",
        content_type=content_type,
        byte_count=len(body),
        raw_path=str(raw_path.relative_to(project_root)),
        metadata_path=str(metadata_path.relative_to(project_root)),
        http_status=http_status,
    )


def _safe_path_part(value: str) -> str:
    return (
        value.replace(":", "")
        .replace("-", "")
        .replace("+", "")
        .replace(".", "")
        .replace("Z", "Z")
    )


def _payload_suffix(content_type: str, resource_name: str) -> str:
    lowered = content_type.lower()
    if "csv" in lowered or resource_name.endswith("_csv"):
        return "csv"
    if "json" in lowered:
        return "json"
    return "txt"


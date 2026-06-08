from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FILES = [
    "model.pt",
    "metadata.json",
    "metrics.json",
    "metrics_live.json",
    "training_log.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publica un modelo entrenado localmente en model_registry para compartirlo por Git/LFS."
    )
    parser.add_argument("--model-id", required=True, help="ID del modelo, por ejemplo neural_hybrid_v2.")
    parser.add_argument("--version", required=True, help="Version compartida, por ejemplo v2026-06-07.")
    parser.add_argument("--source-dir", required=True, help="Carpeta local con model.pt y metadata.json.")
    parser.add_argument(
        "--registry-root",
        default=str(PROJECT_ROOT / "model_registry"),
        help="Carpeta raiz del registro versionado.",
    )
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Incluye checkpoint_best.pt/checkpoint_last.pt si existen. Por defecto no se copian.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Nota corta para README del modelo publicado.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir
    if not source_dir.exists():
        raise RuntimeError(f"No existe source_dir: {source_dir}")

    registry_root = Path(args.registry_root)
    if not registry_root.is_absolute():
        registry_root = PROJECT_ROOT / registry_root
    target_dir = registry_root / args.model_id / args.version
    target_dir.mkdir(parents=True, exist_ok=True)

    files = list(DEFAULT_FILES)
    if args.include_checkpoints:
        files.extend(["checkpoint_best.pt", "checkpoint_last.pt"])

    copied: list[dict[str, Any]] = []
    for filename in files:
        source_path = source_dir / filename
        if not source_path.exists():
            continue
        target_path = target_dir / filename
        shutil.copy2(source_path, target_path)
        copied.append(file_record(target_path, target_dir))

    parent_summary = source_dir.parent / "training_summary.json"
    if parent_summary.exists():
        target_summary = target_dir / "training_summary.json"
        shutil.copy2(parent_summary, target_summary)
        copied.append(file_record(target_summary, target_dir))

    if not (target_dir / "model.pt").exists() or not (target_dir / "metadata.json").exists():
        raise RuntimeError("El modelo publicado debe incluir model.pt y metadata.json.")

    manifest = {
        "model_id": args.model_id,
        "version": args.version,
        "published_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_dir": str(source_dir),
        "notes": args.notes,
        "files": sorted(copied, key=lambda item: item["path"]),
    }
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_readme(target_dir, manifest)
    update_registry_index(registry_root)

    print(f"published: {target_dir}")
    print(f"files: {len(copied)}")
    return 0


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_readme(target_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# {manifest['model_id']} {manifest['version']}",
        "",
        manifest.get("notes") or "Modelo publicado para uso compartido.",
        "",
        "## Archivos",
        "",
    ]
    for item in manifest["files"]:
        mb = item["bytes"] / (1024 * 1024)
        lines.append(f"- `{item['path']}` ({mb:.2f} MB)")
    lines.extend(
        [
            "",
            "## Uso",
            "",
            "Este directorio se referencia desde `configs/models.yaml` como `artifact_dir`.",
            "Los pesos `*.pt` se versionan con Git LFS.",
            "",
        ]
    )
    (target_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def update_registry_index(registry_root: Path) -> None:
    entries = []
    for manifest_path in sorted(registry_root.glob("*/*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries.append(
            {
                "model_id": manifest["model_id"],
                "version": manifest["version"],
                "path": str(manifest_path.parent.relative_to(registry_root)).replace("\\", "/"),
                "published_at_utc": manifest.get("published_at_utc"),
            }
        )
    (registry_root / "registry.json").write_text(
        json.dumps({"models": entries}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Google Sheets document URL", re.compile(r"https://docs\.google\.com/spreadsheets/d/[A-Za-z0-9_-]{20,}", re.I)),
    ("Google Sheets export path", re.compile(r"spreadsheets/d/[A-Za-z0-9_-]{20,}", re.I)),
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_-]{20,}")),
    ("GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}")),
    ("OpenAI key", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    ("private key block", re.compile(r"BEGIN [A-Z ]*PRIVATE KEY")),
    ("local Windows user path", re.compile(r"C:\\Users\\[^\\\s\"']+", re.I)),
]


def _load_payload(html: str) -> dict:
    match = re.search(r"const DATA = (\{.*?\});\s*\n", html, re.DOTALL)
    if not match:
        raise ValueError("No se encontro el bloque const DATA.")
    return json.loads(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida que un dashboard HTML sea seguro para publicarse."
    )
    parser.add_argument(
        "html",
        nargs="?",
        default="docs/index.html",
        help="Ruta del dashboard HTML generado.",
    )
    parser.add_argument(
        "--require-no-friends",
        action="store_true",
        help="Falla si DATA.friends contiene participantes.",
    )
    args = parser.parse_args()

    html_path = Path(args.html)
    raw = html_path.read_text(encoding="utf-8")
    payload = _load_payload(raw)

    errors: list[str] = []
    friends = payload.get("friends") or []
    if args.require_no_friends and friends:
        errors.append("DATA.friends contiene participantes.")
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(raw):
            errors.append(f"Posible secreto en HTML: {label}.")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"publish dashboard ok: {html_path}")
    print(f"matches: {len(payload.get('matches') or [])}")
    print(f"friends: {len(friends)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

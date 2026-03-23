from __future__ import annotations

"""Validate and sync notebook markdown files in UTF-8.

Usage:
  python scripts/notebook_utf8_guard.py --check
  python scripts/notebook_utf8_guard.py --sync-md
"""

import argparse
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
PREFIXES = ("00_", "01_")


def find_target_ipynb() -> list[Path]:
    files: list[Path] = []
    for p in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        if p.name.startswith(PREFIXES) and not p.name.isascii():
            files.append(p)
    return files


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def validate_file(path: Path) -> tuple[bool, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    q_count = text.count("?")
    cjk = has_cjk(text)

    if not cjk:
        return False, f"{path.name}: no CJK characters found"
    if q_count > 30:
        return False, f"{path.name}: suspicious question mark count = {q_count}"
    return True, f"{path.name}: ok"


def sync_md_from_ipynb(path: Path) -> Path:
    nb = json.loads(path.read_text(encoding="utf-8"))
    sections: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        text = "".join(cell.get("source", [])).rstrip()
        if text:
            sections.append(text)

    md_text = "\n\n---\n\n".join(sections) + "\n"
    md_path = path.with_suffix(".md")
    md_path.write_text(md_text, encoding="utf-8")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sync-md", action="store_true")
    args = parser.parse_args()

    targets = find_target_ipynb()
    if not targets:
        print("No target notebooks found")
        return 1

    failed = False
    for p in targets:
        ok, msg = validate_file(p)
        print(msg)
        if not ok:
            failed = True

    if args.sync_md and not failed:
        for p in targets:
            md_path = sync_md_from_ipynb(p)
            print(f"synced: {md_path.name}")

    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

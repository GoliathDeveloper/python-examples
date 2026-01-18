#!/usr/bin/env python3
"""create_summary_anchors.py

Automatically generate ``_summary.md`` files for every folder that contains Markdown
files.  The generated file contains a simple list of the folder’s immediate
children (sub‑folders and Markdown documents) and can be used as a
folder‑level anchor for the RLM‑style retrieval workflow.

Usage:
    python3 create_summary_anchors.py [--root <path>]

If ``--root`` is omitted, the current working directory is used.
"""

import argparse
import sys
from pathlib import Path


def generate_summary(folder: Path) -> str:
    """Return a Markdown summary for *folder*.

    The summary lists sub‑folders and Markdown files (excluding
    ``_summary.md``).  If the folder contains no Markdown files, the
    function returns an empty string.
    """
    items = []
    for child in sorted(folder.iterdir()):
        if child.is_dir():
            items.append(f"- {child.name}/ (folder)")
        elif child.suffix == ".md" and child.name != "_summary.md":
            items.append(f"- {child.stem} (document)")
    if not items:
        return ""
    header = f"# {folder.name}\n\nThis folder contains the following items:\n"
    return header + "\n".join(items) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate _summary.md files for Markdown folders")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Root directory to scan (default: current working directory)")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    created = 0
    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue
        # Skip hidden directories
        if folder.name.startswith("."):
            continue
        # Check if folder contains any Markdown files
        md_files = list(folder.glob("*.md"))
        if not md_files:
            continue
        summary_path = folder / "_summary.md"
        if summary_path.exists():
            # Skip existing summaries to avoid accidental overwrite
            continue
        content = generate_summary(folder)
        if not content:
            continue
        summary_path.write_text(content, encoding="utf-8")
        print(f"Created {summary_path}")
        created += 1

    print(f"\nFinished. {created} _summary.md files created.")


if __name__ == "__main__":
    main()

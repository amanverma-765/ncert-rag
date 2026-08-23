"""Fetch the NCERT textbooks this pipeline reads, using ncert-cli."""

import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# codes from `ncert --list --class 10`; filenames from ncert_cli.naming
BOOKS = {
    "maths": ("jemh1", "class_10_mathematics.pdf"),
    "science": ("jesc1", "class_10_science.pdf"),
}


def download_books() -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ncert",
            *(code for code, _ in BOOKS.values()),
            "-o",
            str(DATA_DIR),
            "--no-prelims",
        ],
        check=True,
    )
    paths = {name: DATA_DIR / filename for name, (_, filename) in BOOKS.items()}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"ncert-cli did not produce: {', '.join(missing)}")
    return paths

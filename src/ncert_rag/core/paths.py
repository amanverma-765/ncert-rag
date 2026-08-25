from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
BOOKS_DIR = DATA_DIR / "books"
DB_PATH = DATA_DIR / "corpus.db"

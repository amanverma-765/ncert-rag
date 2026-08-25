"""Build the corpus: PDFs in, searchable SQLite out.

Idempotent per book: one whose PDFs hash the same as last time is skipped,
so a rebuild after touching one book does not re-parse the other sixteen.
"""

import hashlib
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from ncert_rag.core.models import BookSpec
from ncert_rag.core.paths import BOOKS_DIR
from ncert_rag.core.registry import BOOKS
from ncert_rag.ingest.chunk import from_pages, from_sections
from ncert_rag.ingest.download import chapter_files, provision
from ncert_rag.ingest.extract import page_lines, page_texts
from ncert_rag.ingest.parse.cutoff import before, page_cut, pages_before
from ncert_rag.ingest.parse.exercises import find as find_exercises
from ncert_rag.ingest.parse.profile import find_marks, induce
from ncert_rag.ingest.parse.resolver import book_offset, chapter_number
from ncert_rag.ingest.parse.sections import split
from ncert_rag.services import embedder
from ncert_rag.store import db, vectors

_PROFILE_SAMPLE = 4  # chapters used to induce the heading gate


def _digest(files: list[Path]) -> str:
    h = hashlib.sha256()
    for path in files:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _build_book(conn: sqlite3.Connection, book: BookSpec) -> list[int]:
    """Parse one book into chunks and exercises. Returns printed chapter numbers."""
    files = chapter_files(BOOKS_DIR / book.slug)
    chapters = [page_lines(path) for path in files]
    profile = induce(chapters[:_PROFILE_SAMPLE])

    numbers = []
    for position, (path, lines) in enumerate(zip(files, chapters, strict=True), 1):
        number = chapter_number(find_marks(lines, profile), position)
        numbers.append(number)

        questions, exercise_page = find_exercises(lines, number)
        db.add_exercises(conn, book.slug, number, questions)

        # the exercises stay out of the index, and both chunk sets lose the
        # same pages so neither arm gains from the cut
        cut = page_cut(lines, exercise_page)
        sections = split(before(lines, cut), profile, book.slug, number)
        pages = pages_before(page_texts(path), cut)

        db.add_chunks(conn, from_sections(sections))
        db.add_chunks(conn, from_pages(book.slug, number, pages))

    return numbers


def _embed_pending(conn: sqlite3.Connection) -> int:
    """Embed every chunk that has no vector yet."""
    rows = conn.execute(
        "SELECT c.id, c.text FROM chunks c "
        "LEFT JOIN embeddings e ON e.chunk_id = c.id WHERE e.chunk_id IS NULL "
        "ORDER BY c.id"
    ).fetchall()
    if not rows:
        return 0

    ids = [row["id"] for row in rows]
    texts = [row["text"] for row in rows]
    for start in range(0, len(ids), 512):
        window = slice(start, start + 512)
        vectors.save(conn, ids[window], embedder.encode_documents(texts[window]))
        conn.commit()
        print(f"  embedded {min(start + 512, len(ids))}/{len(ids)}")
    return len(ids)


def build(books: Iterable[BookSpec] = BOOKS, rebuild: bool = False) -> None:
    books = list(books)
    provision(books)
    conn = db.connect()

    for book in books:
        files = chapter_files(BOOKS_DIR / book.slug)
        digest = _digest(files)
        if not rebuild and db.stored_digest(conn, book.slug) == digest:
            print(f"{book.slug}: unchanged, skipping")
            continue

        db.clear_book(conn, book.slug)
        numbers = _build_book(conn, book)
        db.add_book(conn, book, digest)
        conn.commit()

        offset = book_offset(numbers)
        note = f"chapters {numbers[0]}-{numbers[-1]}"
        if offset is None:
            note += " (numbering inconsistent, using file order where unresolved)"
        elif offset != 0:
            note += f" (offset {offset:+d} from filenames)"
        print(f"{book.slug}: {len(numbers)} files, {note}")

    print("Embedding...")
    print(f"  {_embed_pending(conn)} new vectors")
    _report(conn, books)
    conn.close()


def _report(conn: sqlite3.Connection, books: list[BookSpec]) -> None:
    """Invariants worth seeing every build."""
    print("\nCorpus")
    for source in ("parsed", "raw"):
        n = conn.execute(
            "SELECT COUNT(*) n FROM chunks WHERE source = ?", (source,)
        ).fetchone()["n"]
        print(f"  {source:>6} chunks: {n}")

    total_q = conn.execute("SELECT COUNT(*) n FROM exercises").fetchone()["n"]
    print(f"  exercise questions: {total_q}")

    for book in books:
        found = conn.execute(
            "SELECT COUNT(DISTINCT chapter) n FROM chunks WHERE book = ?",
            (book.slug,),
        ).fetchone()["n"]
        if found != book.chapters:
            print(f"  ! {book.slug}: {found} chapters parsed, {book.chapters} expected")

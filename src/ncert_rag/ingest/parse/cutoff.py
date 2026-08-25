"""Where a chapter's searchable content stops.

NCERT prints the exercise list inside the chapter. Indexing it lets an eval
built from those questions match each question against its own printed copy,
scoring a retriever for finding the question rather than the explanation. Both
chunk sets cut at the same page so neither is advantaged.
"""

from ncert_rag.ingest.extract import Line

# A cut that would take most of the chapter means the run was body text, not an
# exercise list.
_MIN_KEEP = 0.3


def page_cut(lines: list[Line], exercise_page: int | None) -> int | None:
    """First page to drop, or None to keep the whole chapter."""
    if exercise_page is None or exercise_page <= 1 or not lines:
        return None
    total = max(line.page for line in lines)
    return exercise_page if (exercise_page - 1) / total >= _MIN_KEEP else None


def before(lines: list[Line], cut: int | None) -> list[Line]:
    return lines if cut is None else [line for line in lines if line.page < cut]


def pages_before(pages: list[str], cut: int | None) -> list[str]:
    return pages if cut is None else pages[: cut - 1]

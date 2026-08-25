"""Cut a chapter into its printed sections.

Marks arrive in reading order but not always in numeric order: chemistry
repeats the current section number in a margin box on every page, so the raw
stream reads 6.1, 6.2, 6.2, 6.1, 6.3. Only marks that advance the sequence open
a new section; the repeats are furniture.
"""

from collections import Counter

from ncert_rag.core.models import Section
from ncert_rag.ingest.extract import Line
from ncert_rag.ingest.parse.profile import SECTION, HeadingProfile, Mark, find_marks

# a chapter with fewer real headings than this is treated as unnumbered prose
_MIN_MARKS = 2


def _drop_furniture(lines: list[Line], threshold: float = 0.4) -> list[Line]:
    """Remove running heads, but never a section mark: those open sections."""
    pages = len({line.page for line in lines})
    if pages < 3:
        return lines

    counts = Counter(line.text for line in lines if len(line.text) <= 60)
    limit = max(2, threshold * pages)
    return [
        line for line in lines if SECTION.match(line.text) or counts[line.text] < limit
    ]


def _advancing(marks: list[Mark], chapter: int) -> list[Mark]:
    """Keep only this chapter's marks, in strictly increasing order."""
    kept: list[Mark] = []
    last: tuple[int, ...] = ()
    for mark in marks:
        if mark.prefix != chapter or mark.key <= last:
            continue
        kept.append(mark)
        last = mark.key
    return kept


def _text(lines: list[Line], start: int, end: int) -> str:
    return "\n".join(line.text for line in lines[start:end]).strip()


def split(
    lines: list[Line], profile: HeadingProfile, book: str, chapter: int
) -> list[Section]:
    """Cut one chapter's lines into sections.

    A chapter whose headings defeat the profile still yields one whole-chapter
    section so its text stays searchable; it just loses section citations.
    """
    lines = _drop_furniture(lines)
    kept = _advancing(find_marks(lines, profile), chapter)

    if len(kept) < _MIN_MARKS:
        body = _text(lines, 0, len(lines))
        page = lines[0].page if lines else 1
        return [Section(book, chapter, None, "", body, page)] if body else []

    sections: list[Section] = []

    # text before the first heading: the chapter opener
    opening = _text(lines, 0, kept[0].index)
    if len(opening) > 200:
        sections.append(
            Section(book, chapter, None, "Introduction", opening, lines[0].page)
        )

    ends = [mark.index for mark in kept[1:]] + [len(lines)]
    for mark, end in zip(kept, ends, strict=True):
        body = _text(lines, mark.index, end)
        if body:
            sections.append(
                Section(book, chapter, mark.number, mark.title, body, mark.page)
            )

    return sections

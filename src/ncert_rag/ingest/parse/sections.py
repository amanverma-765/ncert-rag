"""Cut a chapter into its printed sections.

Marks arrive in reading order but not always in numeric order: chemistry
repeats the current section number in a margin box on every page, so the raw
stream reads 6.1, 6.2, 6.2, 6.1, 6.3. Only marks that advance the sequence open
a new section; the repeats are furniture.
"""

from collections import Counter

from ncert_rag.core.models import Section
from ncert_rag.ingest.clean import clean_text, is_page_number
from ncert_rag.ingest.extract import Line
from ncert_rag.ingest.parse.profile import SECTION, HeadingProfile, Mark, find_marks

# a chapter with fewer real headings than this is treated as unnumbered prose
_MIN_MARKS = 2


def _drop_furniture(lines: list[Line], threshold: float = 0.4) -> list[Line]:
    """Remove running heads and page numbers, but never a section mark.

    Page numbers are dropped by position, exactly as the raw path drops them,
    because they never repeat often enough to look like furniture on their own.
    """
    pages = len({line.page for line in lines})
    if pages < 3:
        return lines

    counts = Counter(line.text for line in lines if len(line.text) <= 60)
    limit = max(2, threshold * pages)

    per_page: dict[int, list[int]] = {}
    for index, line in enumerate(lines):
        per_page.setdefault(line.page, []).append(index)
    position = {
        index: (place, len(group))
        for group in per_page.values()
        for place, index in enumerate(group)
    }

    kept = []
    for index, line in enumerate(lines):
        if SECTION.match(line.text):
            kept.append(line)
            continue
        place, total = position[index]
        if counts[line.text] >= limit or is_page_number(line.text, place, total):
            continue
        kept.append(line)
    return kept


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
    """Join a run of lines and clean the result as a block.

    `page_lines` cleans each line on its own, where the newline-dependent rules
    -- rejoining a word broken across lines, reattaching a drop cap -- cannot
    match. The join is the first point at which they can, and `clean_text` is
    idempotent for everything else it does, so running it again here is what
    keeps this text identical to the raw path's.
    """
    return clean_text("\n".join(line.text for line in lines[start:end])).strip()


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
        return (
            [Section(book=book, chapter=chapter, number=None, text=body, page=page)]
            if body
            else []
        )

    sections: list[Section] = []

    # text before the first heading: the chapter opener
    opening = _text(lines, 0, kept[0].index)
    if len(opening) > 200:
        sections.append(
            Section(
                book=book,
                chapter=chapter,
                number=None,
                text=opening,
                page=lines[0].page,
            )
        )

    ends = [mark.index for mark in kept[1:]] + [len(lines)]
    for mark, end in zip(kept, ends, strict=True):
        body = _text(lines, mark.index, end)
        if body:
            sections.append(
                Section(
                    book=book,
                    chapter=chapter,
                    number=mark.number,
                    text=body,
                    page=mark.page,
                )
            )

    return sections

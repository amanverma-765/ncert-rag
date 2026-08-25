"""Two chunkings of the same text: section-aware, and fixed windows.

Both use the same window size and the same cleaned text, so the eval can
attribute any difference between them to the boundaries alone.

Windows are counted in words rather than tokens. Close enough at this size, and
it keeps a tokenizer out of the build.
"""

from collections.abc import Iterator

from ncert_rag.core.models import Chunk, Section

WORDS = 350
OVERLAP = 60


def _windows(words: list[str]) -> Iterator[tuple[int, list[str]]]:
    """(start index, window) pairs covering the whole word list."""
    if not words:
        return
    start = 0
    while True:
        end = min(start + WORDS, len(words))
        yield start, words[start:end]
        if end >= len(words):
            return
        start = end - OVERLAP


def from_sections(sections: list[Section]) -> list[Chunk]:
    """Chunk within section boundaries; a window never spans two sections."""
    return [
        Chunk(
            book=section.book,
            chapter=section.chapter,
            section=section.number,
            page=section.page,
            text=" ".join(window),
            source="parsed",
        )
        for section in sections
        for _start, window in _windows(section.text.split())
    ]


def from_pages(book: str, chapter: int, pages: list[str]) -> list[Chunk]:
    """Chunk the chapter as one flat stream, structure ignored."""
    words: list[str] = []
    page_of: list[int] = []  # page each word came from, for citations
    for number, text in enumerate(pages, start=1):
        page_words = text.split()
        words.extend(page_words)
        page_of.extend([number] * len(page_words))

    return [
        Chunk(
            book=book,
            chapter=chapter,
            section=None,
            page=page_of[start],
            text=" ".join(window),
            source="raw",
        )
        for start, window in _windows(words)
    ]

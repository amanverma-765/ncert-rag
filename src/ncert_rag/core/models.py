"""Value types shared across the pipeline. No internal imports live here."""

from dataclasses import dataclass
from typing import Literal

# 'clean' parses reliably; 'fragmented' shreds equations into <=3-char lines
# (maths books run 60-71% such lines) and leans on window-chunk fallback.
Tier = Literal["clean", "fragmented"]

# 'parsed' chunks respect section boundaries; 'raw' are fixed windows over the
# same cleaned text. Both live in one table so the arms share a corpus.
ChunkSource = Literal["parsed", "raw"]


@dataclass(frozen=True, slots=True)
class BookSpec:
    slug: str  # directory name under data/books/
    code: str  # ncert-cli catalog code, e.g. "kebo1"
    klass: int
    subject: str
    tier: Tier
    chapters: int  # expected count, checked as a build invariant


@dataclass(frozen=True, slots=True)
class Section:
    book: str
    chapter: int
    number: str | None  # "2.1"; None when the chapter carries no numbering
    title: str
    text: str
    page: int  # page the section starts on


@dataclass(frozen=True, slots=True)
class Chunk:
    book: str
    chapter: int
    section: str | None
    page: int
    text: str
    source: ChunkSource


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    chunk_id: int
    book: str
    chapter: int
    section: str | None
    page: int
    text: str
    score: float

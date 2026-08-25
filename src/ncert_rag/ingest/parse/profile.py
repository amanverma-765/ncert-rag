"""Work out how one book marks its section headings.

NCERT uses at least four layout dialects across the corpus: biology puts the
number on its own line two points above body size, chemistry keeps it inline at
body size and relies on weight alone, class 10 science sets headings large but
NOT bold (and overprints them several times), CS and maths sit somewhere
between. One hardcoded font gate cannot serve all of them, so induce the gate
per book by trying a few and keeping whichever yields the most coherent
section sequence.
"""

import re
from collections import Counter
from dataclasses import dataclass

from ncert_rag.ingest.extract import Line

# "2.1" / "2.1.3", optionally followed by a title on the same line. The
# negative lookahead rejects decimals in running text: 9.109382 stops at 9.10
# and then sees another digit, 0.25 survives this but dies on the chapter-number
# check downstream.
SECTION = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,2})(?![\d.])[ \t]*(.*)$")


@dataclass(frozen=True, slots=True)
class HeadingProfile:
    body_size: float
    min_size: float
    require_bold: bool


@dataclass(frozen=True, slots=True)
class Mark:
    """A detected section heading and where it sits in the line stream."""

    number: str
    title: str
    page: int
    index: int

    @property
    def prefix(self) -> int:
        return int(self.number.split(".")[0])

    @property
    def key(self) -> tuple[int, ...]:
        return tuple(int(p) for p in self.number.split("."))


def body_size(lines: list[Line]) -> float:
    """The size most characters are set in."""
    weight: Counter[float] = Counter()
    for line in lines:
        weight[line.size] += len(line.text)
    return weight.most_common(1)[0][0] if weight else 10.5


def find_marks(lines: list[Line], profile: HeadingProfile) -> list[Mark]:
    """Every heading candidate the profile admits, in reading order."""
    marks: list[Mark] = []
    for i, line in enumerate(lines):
        if line.size < profile.min_size or (profile.require_bold and not line.bold):
            continue
        m = SECTION.match(line.text)
        if not m:
            continue
        number, title = m.group(1), m.group(2).strip()
        if not title:
            title = _title_after(lines, i, line.size)
        marks.append(Mark(number=number, title=title, page=line.page, index=i))
    return _dedupe(marks)


def _title_after(lines: list[Line], i: int, size: float) -> str:
    """Biology-style layout: the number is alone, the title is the next line."""
    for nxt in lines[i + 1 : i + 3]:
        if abs(nxt.size - size) < 0.6 and not SECTION.match(nxt.text):
            return nxt.text
    return ""


def _dedupe(marks: list[Mark]) -> list[Mark]:
    """Collapse overprinted repeats, keeping the longest title seen.

    Class 10 science draws each heading several times at slight offsets, so the
    same number arrives as '1.1 CHEMIC', '1.1 CHEMIC', '1.1 CHEMICAL EQUA'.
    """
    out: list[Mark] = []
    for mark in marks:
        if out and out[-1].number == mark.number and mark.index - out[-1].index < 6:
            if len(mark.title) > len(out[-1].title):
                out[-1] = mark
            continue
        out.append(mark)
    return out


def _coherence(marks: list[Mark]) -> int:
    """How many marks agree on a chapter and climb in order.

    A gate that lets in noise scores badly because the stray numbers neither
    share the majority chapter prefix nor keep ascending.
    """
    if not marks:
        return 0
    prefix = Counter(m.prefix for m in marks).most_common(1)[0][0]
    kept = [m for m in marks if m.prefix == prefix]

    run, last = 0, ()
    for mark in kept:
        if mark.key >= last:
            run += 1
            last = mark.key
    return run


def induce(chapters: list[list[Line]]) -> HeadingProfile:
    """Pick the font gate that best explains this book's section numbering."""
    sizes = [body_size(lines) for lines in chapters if lines]
    body = Counter(sizes).most_common(1)[0][0] if sizes else 10.5

    # stricter gates first, so a tie keeps the one less likely to admit prose
    gates = [
        HeadingProfile(body, body + 0.9, True),
        HeadingProfile(body, body + 0.9, False),
        HeadingProfile(body, body, True),
        HeadingProfile(body, body, False),
    ]
    return max(
        gates,
        key=lambda g: sum(_coherence(find_marks(lines, g)) for lines in chapters),
    )

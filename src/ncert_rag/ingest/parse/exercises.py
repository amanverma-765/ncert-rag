"""Lift end-of-chapter exercise questions, which are the eval's ground truth.

Each question came from a known chapter, so a retriever that surfaces that
chapter is right and one that doesn't is wrong, with no hand labelling.

The books disagree on format: biology writes '1.How is...', class 10 science
puts '12.' alone above its text, chemistry numbers questions '6.9' by chapter,
and some chapters carry no EXERCISES header at all. What they share is a run of
ascending question numbers near the end, so that is what we look for rather
than any particular header.
"""

import re

from ncert_rag.ingest.extract import Line

_ORDINAL = re.compile(r"^(\d{1,2})\.\s*(.*)$")  # "12." or "1. List some"
_BY_CHAPTER = re.compile(r"^(\d{1,2})\.(\d{1,2})\s*(.*)$")  # chemistry's "6.9"

_MIN_RUN = 4  # fewer ascending numbers than this is a coincidence, not a list
_MIN_WORDS = 6
_MAX_CHARS = 400

# an ascending run also matches answer keys, matching-column tables and lists of
# worked equations. A real question asks something: it ends in '?' or opens with
# an instruction verb.
_ASKS = re.compile(
    r"(?i)^(what|why|how|when|where|which|who|whose|explain|define|describe|name|"
    r"write|give|state|list|differentiate|distinguish|discuss|compare|account|"
    r"identify|mention|justify|suggest|outline|illustrate|comment|answer|choose)\b"
)

# the print-run stamp bleeds in from the page footer
_STAMP = re.compile(r"Reprint\s*\d{4}.*$")

# below this share of letters the text is shredded formulas, not prose
_MIN_ALPHA = 0.7


def _is_question(text: str) -> bool:
    alpha = sum(c.isalpha() or c.isspace() for c in text)
    if alpha / len(text) < _MIN_ALPHA:
        return False
    return text.endswith("?") or bool(_ASKS.match(text))


def _starts(lines: list[Line], chapter: int) -> list[tuple[int, int, str]]:
    """(line index, question number, text on the same line) for every candidate."""
    found = []
    for i, line in enumerate(lines):
        by_chapter = _BY_CHAPTER.match(line.text)
        if by_chapter:
            # only the chapter's own numbering counts; 3.14 in prose does not
            if int(by_chapter.group(1)) == chapter:
                found.append((i, int(by_chapter.group(2)), by_chapter.group(3).strip()))
            continue
        ordinal = _ORDINAL.match(line.text)
        if ordinal:
            found.append((i, int(ordinal.group(1)), ordinal.group(2).strip()))
    return found


def _runs(starts: list[tuple[int, int, str]]) -> list[list[tuple[int, int, str]]]:
    """Split candidates into maximal runs of consecutive numbers."""
    runs: list[list[tuple[int, int, str]]] = []
    for start in starts:
        if runs and start[1] == runs[-1][-1][1] + 1:
            runs[-1].append(start)
        else:
            runs.append([start])
    return [run for run in runs if len(run) >= _MIN_RUN]


def _questions(lines: list[Line], run: list[tuple[int, int, str]]) -> list[str]:
    """Text of each numbered item, from its number to the next one."""
    bounds = [i for i, _n, _t in run[1:]] + [len(lines)]
    out = []
    for (index, _number, inline), end in zip(run, bounds, strict=True):
        body = " ".join([inline] + [line.text for line in lines[index + 1 : end]])
        body = _STAMP.sub("", re.sub(r"\s+", " ", body)).strip()
        if (
            len(body.split()) >= _MIN_WORDS
            and len(body) <= _MAX_CHARS
            and _is_question(body)
        ):
            out.append(body)
    return out


def find(lines: list[Line], chapter: int) -> tuple[list[str], int | None]:
    """This chapter's exercise questions, and the page they start on.

    A chapter can hold several ascending runs: the exercises, an answer key,
    a matching-column table. Keep whichever run yields the most text that reads
    like a question, preferring the later one when they tie, since exercises
    come after worked examples.

    The page matters as much as the questions: those pages must stay out of the
    searchable corpus or an eval built from these questions would be scoring
    each one against its own printed copy.
    """
    best: list[str] = []
    page: int | None = None
    for run in _runs(_starts(lines, chapter)):
        found = _questions(lines, run)
        if len(found) >= len(best):
            best, page = found, lines[run[0][0]].page
    return best, (page if best else None)

"""Pull text out of a chapter PDF, by page and by line.

`page_texts` feeds the raw chunks; `page_lines` carries the font metrics the
heading detection needs.
"""

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from ncert_rag.ingest.clean import clean_text, strip_running_heads

_BOLD_MARKERS = ("Demi", "Bold", "Black", "Heavy")


@dataclass(frozen=True, slots=True)
class Line:
    text: str
    size: float
    bold: bool
    page: int


def page_texts(path: Path) -> list[str]:
    """Cleaned text per page, furniture removed."""
    with pymupdf.open(path) as doc:
        pages = [clean_text(page.get_text()) for page in doc]
    return strip_running_heads(pages)


def page_lines(path: Path) -> list[Line]:
    """Every non-empty line with the font metrics needed for heading detection.

    Size is the largest span on the line and bold is true if any span is, so a
    heading whose number and title differ slightly still reads as one unit.
    """
    out: list[Line] = []
    with pymupdf.open(path) as doc:
        for pno, page in enumerate(doc, start=1):
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = clean_text("".join(s["text"] for s in spans))
                    if not text:
                        continue
                    out.append(
                        Line(
                            text=text,
                            size=max(round(s["size"], 1) for s in spans),
                            bold=any(
                                m in s["font"] for s in spans for m in _BOLD_MARKERS
                            ),
                            page=pno,
                        )
                    )
    return out

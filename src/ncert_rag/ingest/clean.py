"""Text normalization applied identically to both chunk sets.

Pure functions on already-extracted text. Running both sides through the same
cleaning keeps chunking the only difference between them.
"""

import re
from collections import Counter

# PyMuPDF emits real ligature codepoints. NFKC would fix them but also flatten
# x² to x2, which the maths books cannot afford.
_LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"})

# "I N T R O D U C T I O N": headings set with letter tracking. Four letters
# minimum so ordinary prose ("I a m") is never a candidate.
_TRACKED = re.compile(r"\b(?:[A-Z] ){3,}[A-Z]\b")

# a word broken across a line: "photo-\nsynthesis". Only when the next line
# starts lowercase, so "well-\nKnown" style compounds survive.
_HYPHEN_BREAK = re.compile(r"(\w)-\n([a-z])")

# NCERT opens sections with a drop cap, which extracts as a lone capital on its
# own line ahead of the rest of the word
_DROP_CAP = re.compile(r"^([A-Z])\n(?=[a-z])", re.MULTILINE)

_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")


def clean_text(text: str) -> str:
    """Normalize one page's extracted text."""
    text = text.translate(_LIGATURES).replace("\xa0", " ")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _DROP_CAP.sub(r"\1", text)
    text = _TRACKED.sub(lambda m: m.group().replace(" ", ""), text)
    # collapse runs of spaces/tabs but keep newlines: section splitting needs them
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_running_heads(pages: list[str], threshold: float = 0.4) -> list[str]:
    """Drop the running header/footer lines that repeat across a chapter.

    A short line appearing on 40%+ of pages is furniture (book title, chapter
    name, page numbers), not content.
    """
    if len(pages) < 3:
        return pages

    counts = Counter(
        line.strip()
        for page in pages
        for line in page.splitlines()
        if 0 < len(line.strip()) <= 60
    )
    furniture = {
        line for line, n in counts.items() if n >= max(2, threshold * len(pages))
    }

    return [
        "\n".join(
            line
            for line in page.splitlines()
            if line.strip() not in furniture and not _PAGE_NUMBER.match(line)
        ).strip()
        for page in pages
    ]

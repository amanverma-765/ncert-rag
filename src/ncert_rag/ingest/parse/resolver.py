"""Recover the chapter number a file actually prints.

Part II volumes restart their filenames at chapter_01 while NCERT keeps
numbering on from Part I, so class_12_chemistry_chemistry_ii/chapter_01.pdf is
really chapter 6. The section marks inside the file give it away: a chapter
whose sections read 6.1, 6.2, 6.3 is chapter 6 whatever the filename says.
"""

from collections import Counter

from ncert_rag.ingest.parse.profile import Mark


def chapter_number(marks: list[Mark], fallback: int) -> int:
    """Majority vote over the section marks' leading component."""
    if not marks:
        return fallback
    number, votes = Counter(m.prefix for m in marks).most_common(1)[0]
    # a lone stray mark is not evidence; require it to actually dominate
    return number if votes >= max(2, len(marks) // 3) else fallback


def book_offset(numbers: list[int]) -> int | None:
    """The constant gap between printed number and file position, if there is one.

    Returns None when the book disagrees with itself, which means the marks are
    too noisy to trust and callers should fall back to file order.
    """
    if not numbers:
        return None
    gaps = Counter(number - i for i, number in enumerate(numbers, start=1))
    gap, votes = gaps.most_common(1)[0]
    # tolerate one odd chapter (an appendix, or one that defeated the profile)
    return gap if votes >= len(numbers) - 1 else None

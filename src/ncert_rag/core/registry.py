"""The books under evaluation: classes 10-12 STEM, English editions.

Physics is left out until formula reconstruction exists. Maths is included but
marked `fragmented`, since its equations shred on extraction and the eval reports
that tier separately rather than pretending it parses like prose.

Slugs are the directory names ncert-cli writes; codes come from `ncert --list`.
Chapter counts are the build-time invariant: a mismatch means a download went
wrong or NCERT re-cut the book.
"""

from ncert_rag.core.models import BookSpec

BOOKS: tuple[BookSpec, ...] = (
    BookSpec("class_10_science", "jesc1", 10, "Science", "clean", 13),
    BookSpec("class_10_mathematics", "jemh1", 10, "Mathematics", "fragmented", 14),
    BookSpec("class_11_biology", "kebo1", 11, "Biology", "clean", 19),
    BookSpec("class_11_biotechnology", "kebt1", 11, "Biotechnology", "clean", 12),
    BookSpec(
        "class_11_chemistry_chemistry_part_i", "kech1", 11, "Chemistry", "clean", 6
    ),
    BookSpec(
        "class_11_chemistry_chemistry_part_ii", "kech2", 11, "Chemistry", "clean", 3
    ),
    BookSpec("class_11_computer_science", "kecs1", 11, "Computer Science", "clean", 11),
    BookSpec(
        "class_11_informatics_practices",
        "keip1",
        11,
        "Informatics Practices",
        "clean",
        8,
    ),
    BookSpec("class_11_mathematics", "kemh1", 11, "Mathematics", "fragmented", 14),
    BookSpec("class_12_biology", "lebo1", 12, "Biology", "clean", 13),
    BookSpec("class_12_biotechnology", "lebt1", 12, "Biotechnology", "clean", 13),
    BookSpec("class_12_chemistry_chemistry_i", "lech1", 12, "Chemistry", "clean", 5),
    BookSpec("class_12_chemistry_chemistry_ii", "lech2", 12, "Chemistry", "clean", 5),
    BookSpec("class_12_computer_science", "lecs1", 12, "Computer Science", "clean", 13),
    BookSpec(
        "class_12_informatics_practices",
        "leip1",
        12,
        "Informatics Practices",
        "clean",
        7,
    ),
    BookSpec(
        "class_12_mathematics_mathematics_part_i",
        "lemh1",
        12,
        "Mathematics",
        "fragmented",
        6,
    ),
    BookSpec(
        "class_12_mathematics_mathematics_part_ii",
        "lemh2",
        12,
        "Mathematics",
        "fragmented",
        7,
    ),
)

BY_SLUG = {book.slug: book for book in BOOKS}

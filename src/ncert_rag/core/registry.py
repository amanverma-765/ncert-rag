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
    BookSpec(
        slug="class_10_science",
        code="jesc1",
        klass=10,
        subject="Science",
        tier="clean",
        chapters=13,
    ),
    BookSpec(
        slug="class_10_mathematics",
        code="jemh1",
        klass=10,
        subject="Mathematics",
        tier="fragmented",
        chapters=14,
    ),
    BookSpec(
        slug="class_11_biology",
        code="kebo1",
        klass=11,
        subject="Biology",
        tier="clean",
        chapters=19,
    ),
    BookSpec(
        slug="class_11_biotechnology",
        code="kebt1",
        klass=11,
        subject="Biotechnology",
        tier="clean",
        chapters=12,
    ),
    BookSpec(
        slug="class_11_chemistry_chemistry_part_i",
        code="kech1",
        klass=11,
        subject="Chemistry",
        tier="clean",
        chapters=6,
    ),
    BookSpec(
        slug="class_11_chemistry_chemistry_part_ii",
        code="kech2",
        klass=11,
        subject="Chemistry",
        tier="clean",
        chapters=3,
    ),
    BookSpec(
        slug="class_11_computer_science",
        code="kecs1",
        klass=11,
        subject="Computer Science",
        tier="clean",
        chapters=11,
    ),
    BookSpec(
        slug="class_11_informatics_practices",
        code="keip1",
        klass=11,
        subject="Informatics Practices",
        tier="clean",
        chapters=8,
    ),
    BookSpec(
        slug="class_11_mathematics",
        code="kemh1",
        klass=11,
        subject="Mathematics",
        tier="fragmented",
        chapters=14,
    ),
    BookSpec(
        slug="class_12_biology",
        code="lebo1",
        klass=12,
        subject="Biology",
        tier="clean",
        chapters=13,
    ),
    BookSpec(
        slug="class_12_biotechnology",
        code="lebt1",
        klass=12,
        subject="Biotechnology",
        tier="clean",
        chapters=13,
    ),
    BookSpec(
        slug="class_12_chemistry_chemistry_i",
        code="lech1",
        klass=12,
        subject="Chemistry",
        tier="clean",
        chapters=5,
    ),
    BookSpec(
        slug="class_12_chemistry_chemistry_ii",
        code="lech2",
        klass=12,
        subject="Chemistry",
        tier="clean",
        chapters=5,
    ),
    BookSpec(
        slug="class_12_computer_science",
        code="lecs1",
        klass=12,
        subject="Computer Science",
        tier="clean",
        chapters=13,
    ),
    BookSpec(
        slug="class_12_informatics_practices",
        code="leip1",
        klass=12,
        subject="Informatics Practices",
        tier="clean",
        chapters=7,
    ),
    BookSpec(
        slug="class_12_mathematics_mathematics_part_i",
        code="lemh1",
        klass=12,
        subject="Mathematics",
        tier="fragmented",
        chapters=6,
    ),
    BookSpec(
        slug="class_12_mathematics_mathematics_part_ii",
        code="lemh2",
        klass=12,
        subject="Mathematics",
        tier="fragmented",
        chapters=7,
    ),
)

BY_SLUG = {book.slug: book for book in BOOKS}

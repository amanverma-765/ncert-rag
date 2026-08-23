from ncert_cli.models import Book
from ncert_cli.naming import output_filename

from ncert_rag.download import BOOKS

# what ncert.nic.in calls these books, as shown by `ncert --list --class 10`
TITLES = {
    "jemh1": ("Mathematics", "Mathematics"),
    "jesc1": ("Science", "Science"),
}


def test_book_filenames_match_ncert_cli():
    """Catches an upstream rename before it becomes a missing-file error."""
    for code, expected in BOOKS.values():
        subject, title = TITLES[code]
        book = Book(
            title=title, code=code, chapters=0, class_number=10, subject=subject
        )
        assert output_filename(book) == expected

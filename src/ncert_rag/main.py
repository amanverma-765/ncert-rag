from ncert_rag.download import download_books
from ncert_rag.parser import parse_pdf


def main() -> None:
    books = download_books()
    averages = [parse_pdf(books["maths"]) for _ in range(100)]
    print(sum(averages) / len(averages))


if __name__ == "__main__":
    main()

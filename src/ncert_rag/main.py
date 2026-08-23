import os
from pathlib import Path

from ncert_rag.parser import parse_pdf

MATH_PDF = Path(os.getcwd()) / "data" / "NCERT_Class10_Mathematics.pdf"
SCIENCE_PDF = Path(os.getcwd()) / "data" / "NCERT_Class10_Science.pdf"


def main() -> None:
    averages = []
    for _ in range(100):
        v = parse_pdf(MATH_PDF)
        averages.append(v)

    super_average = sum(averages) / len(averages)
    print(super_average)


if __name__ == "__main__":
    main()

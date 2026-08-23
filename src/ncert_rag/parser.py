import random
from pathlib import Path

import tiktoken
from pymupdf import pymupdf

encoding = tiktoken.encoding_for_model("gpt-5.6")


def parse_pdf(path: Path) -> float | int:
    doc = pymupdf.open(path)
    page_nums = random.sample(range(len(doc)), 5)

    token_lengths = []

    for num in page_nums:
        text = doc[num].get_text()
        token_count = len(encoding.encode(text))
        token_lengths.append(token_count)

        print(f"Page {num}: {token_count} tokens")

    average_tokens = sum(token_lengths) / len(token_lengths)

    print(f"Average tokens: {average_tokens:.0f}")
    return average_tokens

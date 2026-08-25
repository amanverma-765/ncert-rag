from ncert_rag.core.models import Section
from ncert_rag.ingest.chunk import OVERLAP, WORDS, from_pages, from_sections


def section(number: str, words: int) -> Section:
    return Section("book", 1, number, "T", " ".join(f"w{i}" for i in range(words)), 3)


def test_short_section_stays_one_chunk():
    chunks = from_sections([section("1.1", 20)])
    assert len(chunks) == 1
    assert chunks[0].section == "1.1"
    assert chunks[0].source == "parsed"


def test_windows_never_span_two_sections():
    chunks = from_sections([section("1.1", 10), section("1.2", 10)])
    assert [c.section for c in chunks] == ["1.1", "1.2"]


def test_long_section_splits_with_overlap():
    chunks = from_sections([section("1.1", WORDS * 2)])
    assert len(chunks) > 1
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-OVERLAP:] == second[:OVERLAP]
    assert all(c.section == "1.1" for c in chunks)


def test_every_chunk_has_text():
    chunks = from_sections([section("1.1", WORDS + 5), section("1.2", 0)])
    assert all(c.text.strip() for c in chunks)


def test_raw_chunks_carry_the_page_the_window_starts_on():
    pages = [" ".join(f"a{i}" for i in range(WORDS)), "b0 b1 b2"]
    chunks = from_pages("book", 7, pages)
    assert chunks[0].page == 1
    assert all(c.source == "raw" and c.section is None for c in chunks)
    assert all(c.chapter == 7 for c in chunks)


def test_raw_chunking_covers_every_word():
    pages = [" ".join(f"w{i}" for i in range(WORDS * 2))]
    seen = {word for chunk in from_pages("b", 1, pages) for word in chunk.text.split()}
    assert len(seen) == WORDS * 2

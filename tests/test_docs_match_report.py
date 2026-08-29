"""README.md and EVALUATION.md copy REPORT.md's tables; this fails when a copy drifts.

Both documents restate generated numbers so a reader never has to open the
report to see them. Copies go stale silently -- that is how a fragmented-tier
paragraph ended up quoting figures from a corpus two rebuilds old -- so every
copied row is compared against the generated one here.

Only tables are guarded. Prose figures ("+6.0 points", "three questions") are
written by hand and reviewed by hand.
"""

from pathlib import Path

import pytest

from evals.run import REPORT_PATH
from ncert_rag.retrieve import ARMS

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
EVALUATION = ROOT / "evals" / "EVALUATION.md"

# Heading holding each accuracy table, per document. The serving-cost table
# names the same arms with different columns, so tables are matched by the
# section they sit in rather than by their rows.
TABLES = [
    ("All questions", "All questions (n=282)"),
    ("Clean tier", "Clean tier, the 13 non-mathematics books (n=239)"),
    ("Fragmented tier", "Fragmented tier, the four mathematics books (n=43)"),
]

R_AT_5 = 1  # cells are [R@1, R@5, R@10, MRR, n]
N = 4


def _sections(path: Path) -> dict[str, str]:
    """Markdown split into {heading: body}, so same-named tables stay apart."""
    out: dict[str, str] = {}
    heading, body = "", []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            out[heading] = "\n".join(body)
            heading, body = line.lstrip("#").strip(), []
        else:
            body.append(line)
    out[heading] = "\n".join(body)
    return out


def _rows(body: str) -> dict[str, list[str]]:
    """{arm: cells} for every table row whose first cell names an arm."""
    rows = {}
    for line in body.splitlines():
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) > 1 and cells[0] in ARMS:
            rows[cells[0]] = cells[1:]
    return rows


def _report(heading: str) -> dict[str, list[str]]:
    rows = _rows(_sections(REPORT_PATH)[heading])
    # A renamed heading would hand every assertion an empty dict and pass.
    assert len(rows) == len(ARMS), f"REPORT.md '{heading}' lost rows: {sorted(rows)}"
    return rows


@pytest.mark.parametrize(("in_report", "in_evaluation"), TABLES)
def test_evaluation_tables_match_report(in_report: str, in_evaluation: str) -> None:
    assert _rows(_sections(EVALUATION)[in_evaluation]) == _report(in_report)


def test_readme_full_comparison_matches_report() -> None:
    assert _rows(_sections(README)["The full comparison"]) == _report("All questions")


@pytest.mark.parametrize("tier", ["clean", "fragmented"])
def test_readme_tier_summary_matches_report(tier: str) -> None:
    """The README's tier strip is R@5 for three arms, pulled from two tables."""
    report = _report(f"{tier.capitalize()} tier")
    expected = [report[arm][R_AT_5] for arm in ("bm25", "hybrid", "expansion_hybrid")]
    expected.append(report["bm25"][N])

    body = _sections(README)["The two tiers"]
    summary = {
        cells[0]: cells[1:]
        for line in body.splitlines()
        if (cells := [c.strip().strip("`") for c in line.strip().strip("|").split("|")])
        and cells[0] in ("clean", "fragmented")
    }
    assert summary[tier] == expected

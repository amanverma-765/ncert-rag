"""The report is the only record of arms that need a model proxy to produce."""

import pytest

from evals import run


def _report(tmp_path, monkeypatch, body: str):
    path = tmp_path / "REPORT.md"
    path.write_text(body)
    monkeypatch.setattr(run, "REPORT_PATH", path)
    return path


def test_refuses_to_drop_arms_the_report_already_holds(tmp_path, monkeypatch):
    _report(tmp_path, monkeypatch, "| bm25 | 75.2% |\n| expansion_hybrid | 92.2% |\n")
    with pytest.raises(SystemExit, match="expansion_hybrid"):
        run._check_overwrite(["bm25"], force=False)


def test_allows_a_run_that_covers_every_arm_present(tmp_path, monkeypatch):
    _report(tmp_path, monkeypatch, "| bm25 | 75.2% |\n| vector | 84.0% |\n")
    run._check_overwrite(["bm25", "vector", "hybrid"], force=False)


def test_force_overrides_the_guard(tmp_path, monkeypatch):
    _report(tmp_path, monkeypatch, "| bm25 | 75.2% |\n| expansion_hybrid | 92.2% |\n")
    run._check_overwrite(["bm25"], force=True)


def test_no_report_yet_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "REPORT_PATH", tmp_path / "absent.md")
    run._check_overwrite(["bm25"], force=False)

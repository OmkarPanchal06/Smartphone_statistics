"""End-to-end check of the Streamlit dashboard via its AppTest harness."""

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit.testing.v1", reason="streamlit not installed")

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(scope="module")
def app():
    at = streamlit.AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    return at


def test_dashboard_runs_without_exceptions(app):
    assert not app.exception, app.exception


def test_dashboard_shows_every_analysis_tab(app):
    labels = [t.label for t in app.tabs]
    assert len(labels) == 5
    lowered = " ".join(labels).lower()
    for keyword in ("battery life", "claim tests", "prediction", "preferences"):
        assert keyword in lowered


def test_dashboard_reports_key_statistics(app):
    texts = [m.label for m in app.metric] + [m.value for m in app.metric]
    joined = " ".join(str(t) for t in texts)
    assert "R²" in joined
    assert "Cramér's V" in joined
    assert "χ² statistic" in joined


def test_claim_verdicts_are_shown(app):
    # The bundled sample rejects every default claim at alpha = 0.05.
    assert len(app.error) >= 1


def test_dashboard_runs_in_legacy_mode(monkeypatch):
    # Simulates an older Streamlit (no alt=, no alert titles, no icons).
    monkeypatch.setenv("SMARTPHONE_STATS_LEGACY_UI", "1")
    at = streamlit.AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 5

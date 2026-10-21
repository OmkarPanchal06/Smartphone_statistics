import pytest

from smartphone_stats.cli import main


@pytest.mark.parametrize("command", ["describe", "estimate", "claim", "predict", "preference", "all"])
def test_every_command_runs_and_reports(command, capsys):
    assert main([command]) == 0
    out = capsys.readouterr().out
    assert "==" in out
    assert len(out.splitlines()) > 3


def test_estimate_for_single_brand(capsys):
    main(["estimate", "--brand", "Cosmos", "--confidence", "0.99"])
    out = capsys.readouterr().out
    assert "Cosmos" in out
    assert "All brands" not in out


def test_claim_with_custom_value(capsys):
    main(["claim", "--brand", "Apex", "--claim", "12.5", "--alpha", "0.01"])
    out = capsys.readouterr().out
    assert "Apex" in out and "claim=12.5" in out


def test_claim_custom_value_needs_brand():
    with pytest.raises(SystemExit):
        main(["claim", "--claim", "12.5"])


def test_predict_with_new_point(capsys):
    main(["predict", "--charging", "60", "--score", "1100"])
    out = capsys.readouterr().out
    assert "Prediction for charging=60 min, score=1100" in out


def test_unknown_brand_fails_cleanly():
    with pytest.raises(SystemExit):
        main(["estimate", "--brand", "Nope"])


def test_unknown_brand_in_predict_fails_cleanly(capsys):
    with pytest.raises(SystemExit, match="unknown brand"):
        main(["predict", "--charging", "60", "--score", "1100", "--brand", "Nope"])

"""Regression checks for isolated evaluations, without decoding source photos."""
import json
import sys

import pytest

from scripts import evaluate_delivery_batch as script


def fixture_case(tmp_path):
    originals = tmp_path / "originals"
    originals.mkdir()
    selected = tmp_path / "manual"
    selected.mkdir()
    (selected / "0.JPG").touch()
    log = tmp_path / "log.json"
    payload = {"source_folders": [str(originals)], "settings": {},
               "manual_ratings": {}, "photos": [
                   {"folder": str(originals), "file": f"{i}.ARW",
                    "group": i, "rating": 1 if i == 0 else 0}
                   for i in range(2)]}
    log.write_text(json.dumps(payload), encoding="utf-8")
    return log, selected, originals


def test_baseline_is_exact_copy_and_cannot_be_overwritten(tmp_path):
    log, selected, _ = fixture_case(tmp_path)
    output = tmp_path / "evaluation"
    result = script.evaluate_case("test", log, selected, output, learn=False)
    assert (output / "baseline.json").read_bytes() == log.read_bytes()
    assert result["metrics"]["retained_precision"] == 1
    with pytest.raises(FileExistsError):
        script.evaluate_case("test", log, selected, output, learn=False)


def test_missing_originals_refuse_visual_training_before_decode(tmp_path, monkeypatch):
    log, selected, _ = fixture_case(tmp_path)
    def forbidden(*_args, **_kwargs):
        pytest.fail("Incomplete originals must not be decoded or trained")
    monkeypatch.setattr(script.calibration, "_feature_rows", forbidden)
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="2 originals missing"):
        script.evaluate_case("test", log, selected, output, learn=True)
    assert json.loads((output / "unavailable-originals.json").read_text())["count"] == 2
    assert not (output / "candidate-profile.json").exists()


@pytest.mark.parametrize("target", ["source", "manual"])
def test_cli_rejects_output_inside_source_or_reference(tmp_path, monkeypatch, target):
    log, selected, originals = fixture_case(tmp_path)
    output = (originals if target == "source" else selected) / "evaluation"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--case", "test", str(log), str(selected),
                                     "--output", str(output)])
    with pytest.raises(SystemExit):
        script.main()
    assert not output.exists()

"""Review packets are derived diagnostics, never a second image pipeline."""
import hashlib
import json

import numpy as np
import pytest

from scripts.build_selection_review_packet import build_packet, build_capture_packet


def _case(tmp_path):
    originals = tmp_path / "originals"
    manual = tmp_path / "manual"
    originals.mkdir()
    manual.mkdir()
    for name in ("a1", "b0"):
        (manual / f"{name}.JPG").touch()
    photos = [
        {"folder": str(originals), "file": "a0.ARW", "group": 1},
        {"folder": str(originals), "file": "a1.ARW", "group": 1},
        {"folder": str(originals), "file": "b0.ARW", "group": 2},
        {"folder": str(originals), "file": "c0.ARW", "group": 3},
    ]
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"photos": photos}), encoding="utf-8")
    ratings = (1, 0, 0, 1)
    positives = (False, True, True, False)
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({
        "complete_evaluated_case": True,
        "evaluated_baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
        "photos": [{"file": photo["file"], "evaluated": True, "ai_rating": rating,
                    "human_retained": positive, "score": 0.5}
                   for photo, rating, positive in zip(photos, ratings, positives)],
    }), encoding="utf-8")
    return originals, manual, baseline, evaluation


def test_review_packet_classifies_groups_and_uses_injected_preview(tmp_path):
    originals, manual, baseline, evaluation = _case(tmp_path)
    output = tmp_path / "review"
    seen = []

    def load_preview(path):
        seen.append(path.name)
        return np.zeros((20, 20, 3), dtype=np.uint8)

    report = build_packet(baseline, evaluation, manual, output, per_category=1,
                          load_preview=load_preview)
    assert {key: len(value) for key, value in report["samples"].items()} == {
        "same_group_different_frame": 1,
        "human_group_missed": 1,
        "ai_group_not_delivered": 1,
    }
    assert sorted(seen) == ["a0.ARW", "a1.ARW", "b0.ARW", "c0.ARW"]
    assert len(list(output.glob("*.jpg"))) == 3
    assert "原因：____" in (output / "请标注原因.md").read_text(encoding="utf-8")
    assert not list(originals.iterdir())


def test_review_packet_rejects_partial_or_misaligned_reference_without_writes(tmp_path):
    originals, manual, baseline, evaluation = _case(tmp_path)
    output = tmp_path / "review"
    data = json.loads(evaluation.read_text(encoding="utf-8"))
    data["photos"][0]["evaluated"] = False
    evaluation.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="不完整"):
        build_packet(baseline, evaluation, manual, output)
    assert not output.exists()
    data["photos"][0]["evaluated"] = True
    evaluation.write_text(json.dumps(data), encoding="utf-8")
    (manual / "wrong.JPG").touch()
    with pytest.raises(ValueError, match="不一致"):
        build_packet(baseline, evaluation, manual, output)
    assert not output.exists()
    (manual / "wrong.JPG").unlink()
    with pytest.raises(ValueError, match="原片目录外"):
        build_packet(baseline, evaluation, manual, originals / "report")
    with pytest.raises(ValueError, match="原片目录外"):
        build_packet(baseline, evaluation, manual, manual / "report")
    with pytest.raises(ValueError, match="1–24"):
        build_packet(baseline, evaluation, manual, output, per_category=25)


def test_capture_packet_matches_paths_not_duplicate_names(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    manual = tmp_path / "manual"
    manual.mkdir()
    (manual / "a_副本.JPG").write_bytes(b"export")
    photos, capture_rows, stats = [], [], {}
    for folder, rating, human in (("morning", 1, False), ("afternoon", 0, True)):
        source = tmp_path / folder
        source.mkdir()
        path = source / "a.ARW"
        path.write_bytes(b"raw")
        photos.append({"folder": str(source), "file": path.name, "group": 1, "score": 1.})
        capture_rows.append({"path": str(path), "group": 1, "ai_rating": rating, "human_retained": human})
    for path in [*(tmp_path / folder / "a.ARW" for folder in ("morning", "afternoon")),
                 manual / "a_副本.JPG"]:
        stats[str(path)] = [path.stat().st_size, path.stat().st_mtime_ns]
    baseline, reference, inventory = [evidence / n for n in ("baseline.json", "reference.json", "inventory.json")]
    baseline.write_text(json.dumps({"photos": photos, "manual_ratings": {}}))
    reference.write_text(json.dumps({"baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
                                     "photos": capture_rows}))
    inventory.write_text(json.dumps(stats))
    seen = []

    def preview(path):
        seen.append(path)
        return np.zeros((20, 20, 3), np.uint8)

    report = build_capture_packet(baseline, reference, inventory, manual, tmp_path / "review",
                                  load_preview=preview)
    assert not report["samples"]["same_group_different_frame"]
    assert len(report["samples"]["human_group_missed"]) == 1
    assert len(report["samples"]["ai_group_not_delivered"]) == 1
    assert len(set(seen)) == len(seen) == report["unique_decodes"] == 2
    (manual / "extra.JPG").touch()
    with pytest.raises(ValueError, match="file list changed"):
        build_capture_packet(baseline, reference, inventory, manual, tmp_path / "refused", load_preview=preview)
    assert not (tmp_path / "refused").exists()

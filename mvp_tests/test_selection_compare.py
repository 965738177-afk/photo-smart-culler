from pathlib import Path
import json
import sys

import pytest

from event_cull.selection_compare import compare_selection, group_allocation_diagnostic


def batch():
    return {"photos": [{"file": f"{i}.ARW", "folder": "/originals", "group": 1,
                        "rating": rating} for i, rating in enumerate((1, 0, 3, 0))]}


def test_finished_folder_reports_retained_metrics_not_unlabelled_retouch_metrics(tmp_path):
    (tmp_path / "0.JPG").write_bytes(b"image")
    (tmp_path / "1.ARW").write_bytes(b"image")
    report = compare_selection(batch(), tmp_path, confirmed_complete=True)
    assert report["metrics"]["retained_precision"] == .5
    assert report["metrics"]["retained_recall"] == .5
    assert report["metrics"]["retouch_precision"] is None
    assert report["differences"]["ai_only_not_delivered"] == 1


def test_group_diagnostic_separates_missed_allocations_from_frame_slots():
    groups = {
        "a": {"ai_target": ["a1"], "human_target": ["a2"]},
        "b": {"ai_target": [], "human_target": ["b1", "b2"]},
        "c": {"ai_target": ["c1", "c2"], "human_target": []},
    }
    result = group_allocation_diagnostic(groups)
    assert result["human_photos_in_unselected_groups"] == 2
    assert result["ai_photos_in_nonhuman_groups"] == 2
    assert result["same_group_slot_max_exact"] == 1
    assert result["same_group_slot_headroom"] == 1
    assert result["same_group_slot_max_recall"] == pytest.approx(1 / 3)


def test_group_ceiling_not_reported_for_incomplete_reference(tmp_path, monkeypatch):
    from scripts import compare_event_selection as script
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_report_payload(tmp_path, [1, 0])), encoding="utf-8")
    selected = tmp_path / "manual"
    selected.mkdir()
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["compare", str(baseline), str(selected), str(output), "--no-previews"])
    script.main()
    report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert "same_group_slot_max_exact" not in report["group_metrics"]


def test_unfinished_folder_never_labels_absence_as_negative(tmp_path):
    (tmp_path / "0.JPG").touch()
    result = compare_selection(batch(), tmp_path)
    assert result["metrics"] is None
    assert result["photos"][2]["human_retained"] is None


def test_same_counter_from_two_original_folders_is_not_guessed(tmp_path):
    (tmp_path / "0.JPG").touch()
    payload = batch()
    payload["photos"].append({"file": "0.ARW", "folder": "/other-card", "group": 2, "rating": 3})
    report = compare_selection(payload, tmp_path, confirmed_complete=True)
    assert report["ambiguous_stems"] == ["0"]
    assert report["photos"][0]["ai_rating"] is None
    assert report["photos"][-1]["human_retained"] is None


def test_raw_jpg_export_pair_counts_as_one_capture(tmp_path):
    (tmp_path / "0.JPG").touch()
    (tmp_path / "0.ARW").touch()
    report = compare_selection(batch(), tmp_path, confirmed_complete=True)
    assert report["human_unique_stems"] == 1
    assert not report["ambiguous_stems"]


def test_renamed_or_foreign_export_blocks_misleading_metrics(tmp_path):
    (tmp_path / "renamed.JPG").touch()
    report = compare_selection(batch(), tmp_path, confirmed_complete=True)
    assert report["unmatched_human_stems"] == ["renamed"]
    assert report["metrics"] is None
    assert all(row["human_retained"] is None for row in report["photos"])


def test_manual_star_edits_are_not_compared_as_original_ai_advice(tmp_path):
    (tmp_path / "0.JPG").touch()
    payload = batch()
    payload["manual_ratings"] = {"/originals/0.ARW": 1}
    payload["photos"][2]["rating"] = 2
    report = compare_selection(payload, tmp_path, confirmed_complete=True)
    assert report["photos"][0]["ai_rating"] is None
    assert report["photos"][2]["ai_rating"] is None


def test_bad_data_is_not_treated_as_empty_result(tmp_path):
    with pytest.raises(ValueError):
        compare_selection({"photos": None}, tmp_path)
    with pytest.raises(ValueError):
        compare_selection(batch(), tmp_path / "missing")


def test_retouch_objective_only_counts_three_stars_not_all_kept_frames(tmp_path):
    (tmp_path / "0.JPG").touch()
    (tmp_path / "2.ARW").touch()
    report = compare_selection(batch(), tmp_path, objective="retouch", confirmed_complete=True)
    assert report["metrics"]["both_featured"] == 1
    assert report["metrics"]["retouch_precision"] == 1
    assert report["metrics"]["retouch_recall"] == .5
    assert "retained_precision" not in report["metrics"]
    assert all(row["human_retained"] is None for row in report["photos"])
    assert report["photos"][0]["human_retouch_candidate"] is True
    assert report["groups"]["1"]["ai_target"] == ["/originals/2.ARW"]


def test_partial_retouch_reference_reports_positive_hits_without_negative_labels(tmp_path):
    (tmp_path / "2.JPG").touch()
    report = compare_selection(batch(), tmp_path, objective="retouch")
    assert report["metrics"] is None
    assert report["positive_reference"]["exact_hits"] == 1
    assert report["positive_reference"]["positive_coverage"] == 1
    assert report["photos"][0]["human_retouch_candidate"] is None
    assert "ai_featured_not_in_reference" not in report["differences"]


def test_direct_reference_ignores_rendered_export_copy_and_xmp(tmp_path):
    (tmp_path / "2.ARW").touch()
    (tmp_path / "2.XMP").touch()
    export = tmp_path / "rendered"
    export.mkdir()
    (export / "2.JPG").touch()
    (export / "foreign.JPG").touch()
    report = compare_selection(batch(), tmp_path, objective="retouch", recursive=False)
    assert report["human_unique_stems"] == 1
    assert not report["ambiguous_stems"]
    assert not report["unmatched_human_stems"]
    assert report["positive_reference"]["exact_hits"] == 1
    recursive = compare_selection(batch(), tmp_path, objective="retouch")
    assert recursive["ambiguous_stems"] == ["2"]


def test_retouch_reference_cannot_turn_manual_rating_into_ai_advice(tmp_path):
    (tmp_path / "2.JPG").touch()
    payload = batch()
    payload["manual_ratings"] = {"/originals/2.ARW": 3}
    report = compare_selection(payload, tmp_path, objective="retouch")
    assert report["positive_reference"]["unscored_positive_stems"] == ["2"]
    assert report["positive_reference"]["exact_hits"] == 0
    assert report["positive_reference"]["positive_coverage"] is None


def test_unknown_comparison_objective_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="未知"):
        compare_selection(batch(), tmp_path, objective="quality")


def test_reference_matcher_scans_supported_files_once(tmp_path, monkeypatch):
    from event_cull import selection_compare
    calls = []
    def collect(folder, *, recursive):
        calls.append((folder, recursive))
        return (folder / "2.ARW",)
    monkeypatch.setattr(selection_compare, "exported_selection_files", collect)
    result = compare_selection(batch(), tmp_path, objective="retouch", recursive=False)
    assert calls == [(tmp_path, False)]
    assert result["positive_reference"]["exact_hits"] == 1


def _report_payload(tmp_path, ratings):
    return {"software_version": "2.1.25", "source_folders": [str(tmp_path / "originals")],
            "settings": {"target_keep_count": 1, "target_keep_max_count": 2},
            "photos": [{"file": f"{i}.ARW", "folder": str(tmp_path / "originals"),
                        "group": 1, "rating": rating, "score": 100,
                        "reasons": "明显虚焦或抖动" if rating == 0 else "原片保留"}
                       for i, rating in enumerate(ratings)]}


def test_report_can_describe_zero_retained_without_formatting_none_as_percentage(tmp_path, monkeypatch):
    from scripts import compare_event_selection as script
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_report_payload(tmp_path, [0])), encoding="utf-8")
    selected = tmp_path / "manual"
    selected.mkdir()
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["compare", str(baseline), str(selected), str(output), "--confirmed-complete"])
    script.main()
    summary = (output / "对比摘要.md").read_text(encoding="utf-8")
    assert "不适用" in summary
    assert "场景名额与组内帧误差" in summary
    assert (output / "逐组差异清单.md").is_file()


def test_report_preview_failure_preserves_metadata_and_records_unseen_samples(tmp_path, monkeypatch):
    from scripts import compare_event_selection as script
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_report_payload(tmp_path, [1, 0])), encoding="utf-8")
    selected = tmp_path / "manual"
    selected.mkdir()
    (selected / "1.JPG").touch()
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["compare", str(baseline), str(selected), str(output), "--confirmed-complete"])
    def unreadable(_path):
        raise OSError("source disappeared")
    monkeypatch.setattr(script, "_load_preview", unreadable)
    script.main()
    report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert report["metrics"]["human_only"] == 1
    assert report["visual_errors"]
    assert all(error["error"] == "source disappeared" for error in report["visual_errors"])


def test_original_comparison_without_previews_never_decodes(tmp_path, monkeypatch):
    from scripts import compare_event_selection as script
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_report_payload(tmp_path, [1, 0])), encoding="utf-8")
    selected = tmp_path / "manual"
    selected.mkdir()
    (selected / "1.JPG").touch()
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["compare", str(baseline), str(selected), str(output), "--confirmed-complete", "--no-previews"])
    monkeypatch.setattr(script, "_load_preview", lambda *_: pytest.fail("metadata-only report must not decode"))
    script.main()
    report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert report["metrics"]["human_only"] == 1
    assert report["previews_requested"] is False
    assert report["visual_samples"] == report["visual_errors"] == []
    assert "没有读取或解码照片" in (output / "对比摘要.md").read_text(encoding="utf-8")


def test_app_log_nested_target_settings_are_supported(tmp_path, monkeypatch):
    from scripts import compare_event_selection as script
    payload = _report_payload(tmp_path, [1, 0])
    payload["settings"] = {"target_selection_count": {"min": 1500, "max": 2560}}
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(payload), encoding="utf-8")
    selected = tmp_path / "manual"
    selected.mkdir()
    (selected / "0.JPG").touch()
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["compare", str(baseline), str(selected), str(output),
                                     "--confirmed-complete", "--no-previews"])
    script.main()
    assert "1500–2560" in (output / "对比摘要.md").read_text(encoding="utf-8")


def test_retouch_report_without_previews_never_decodes_or_trains(tmp_path, monkeypatch):
    from scripts import compare_retouch_reference as script
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_report_payload(tmp_path, [1, 3])), encoding="utf-8")
    selected = tmp_path / "manual"
    selected.mkdir()
    (selected / "1.ARW").touch()
    output = tmp_path / "retouch-report"
    monkeypatch.setattr(sys, "argv", ["compare", str(baseline), str(selected), str(output), "--direct"])
    def forbidden(*args, **kwargs):
        pytest.fail("Reference metadata comparison must not decode images")
    monkeypatch.setattr(script, "_load_preview", forbidden)
    monkeypatch.setattr(script, "exported_selection_files", forbidden)
    script.main()
    report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert report["positive_reference"]["exact_hits"] == 1
    assert report["metrics"] is None
    assert report["training_performed"] is False
    assert report["reference_ai_rating_counts"] == {"3": 1}


def test_retouch_report_rejects_output_inside_selected_photos(tmp_path, monkeypatch):
    from scripts import compare_retouch_reference as script
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_report_payload(tmp_path, [3])), encoding="utf-8")
    selected = tmp_path / "manual"
    selected.mkdir()
    output = selected / "report"
    monkeypatch.setattr(sys, "argv", ["compare", str(baseline), str(selected), str(output)])
    with pytest.raises(SystemExit):
        script.main()
    assert not output.exists()


def test_retouch_report_preserves_existing_report(tmp_path, monkeypatch):
    from scripts import compare_retouch_reference as script
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_report_payload(tmp_path, [3])), encoding="utf-8")
    selected = tmp_path / "manual"
    selected.mkdir()
    output = tmp_path / "report"
    output.mkdir()
    frozen = output / "comparison.json"
    frozen.write_text("immutable", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["compare", str(baseline), str(selected), str(output)])
    with pytest.raises(SystemExit):
        script.main()
    assert frozen.read_text(encoding="utf-8") == "immutable"

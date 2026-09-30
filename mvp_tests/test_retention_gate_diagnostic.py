"""Known answers can diagnose gates but cannot qualify a release."""
from dataclasses import asdict
from pathlib import Path

import pytest

from event_cull import engine
from scripts.diagnose_retention_gates import diagnose_case


def test_partial_and_preculled_refs_refused_before_any_label_read(monkeypatch):
    from scripts import diagnose_retention_gates as diagnostic
    monkeypatch.setattr(diagnostic,"compare_selection",lambda *a,**k:pytest.fail("Must reject invalid reference role first"))
    for case in ({"source_preculled":True,"retention_labels_complete":True},
                 {"retention_labels_complete":False},
                 {"retention_labels_complete":True,"role":"curated_positive_control"}):
        with pytest.raises(ValueError,match="Preculled/partial"):
            diagnose_case(case,{})


def test_perfect_hypothetical_result_is_never_model_accuracy_or_release(monkeypatch,tmp_path):
    from scripts import diagnose_retention_gates as diagnostic
    monkeypatch.setattr(engine,"_load_preview",lambda *a,**k:pytest.fail("No image decoding in gate diagnosis"))
    case = {"case_id":"fixture","originals":str(tmp_path),"selected_originals":str(tmp_path/"selected"),
            "original_count":2,"human_retained_count":1,"retention_labels_complete":True}
    rows = [{"folder":str(tmp_path),"file":f"image{i}.ARW","group":i,"rating":1,
             "score":100.,"frame_sharpness":200.,"highlight_clip":0.,"shadow_clip":0.,
             "perceptual_hash":i,"reasons":"","median_luminance":110.} for i in range(2)]
    compared = {"metrics":{},"human_unique_stems":1,"ambiguous_stems":[],"unmatched_human_stems":[],
                "photos":[{"human_retained":True,"ai_rating":1},{"human_retained":False,"ai_rating":1}]}
    monkeypatch.setattr(diagnostic,"compare_selection",lambda *a,**k:compared)
    payload = {"settings":asdict(engine.AnalysisOptions(scene_type="婚礼",target_keep_count=1,target_keep_max_count=1)),
               "photos":rows}
    report = diagnose_case(case,payload)
    assert report["hypothetical_metrics"]["precision"]==1.
    assert report["selection_accuracy"] is None and report["passed_acceptance"] is False
    assert report["release_approved"] is False and report["answers_used_as_scores"] is True
    assert report["additional_image_decodes"]==0 and report["model_training_performed"] is False
    assert rows[0]["group"]==0 and rows[1]["rating"]==1  # Only fresh local results are rated.
    rows[0]["frame_sharpness"] = 0.
    blocked = diagnose_case(case,payload)
    assert blocked["hard_gate_human_retained_blocked"]==1
    assert blocked["hypothetical_metrics"]["both_retained"]==0


def test_incomplete_case_refused_before_reading_labels(monkeypatch,tmp_path):
    from scripts import diagnose_retention_gates as diagnostic
    monkeypatch.setattr(diagnostic,"compare_selection",lambda *a,**k:pytest.fail("Incomplete baseline must be refused"))
    case={"originals":str(tmp_path),"original_count":2,"retention_labels_complete":True}
    with pytest.raises(ValueError,match="Incomplete"):
        diagnose_case(case,{"photos":[]})

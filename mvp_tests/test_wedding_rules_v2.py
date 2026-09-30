"""Rule boundary regressions; observations are fixtures, not measured accuracy."""
import copy
import json
from pathlib import Path
import threading

import numpy as np
import pytest

from event_cull import engine, wedding_content as content, wedding_rules as policy
from event_cull.app import EventCullApp


def photo(name="frame.jpg", **changes):
    values = dict(score=90, sharpness=150, subject_sharpness=150,
                  highlight_clip=0, shadow_clip=0, perceptual_hash=1,
                  frontal_face_count=5, open_eye_face_count=5, smiling_face_count=5,
                  face_coverage=.2, eye_detection_mode="scene_group")
    values.update(changes)
    return engine.PhotoResult(Path(name), **values)


def evidence(context="formal_group", **changes):
    checks = dict.fromkeys(content.CHECK_NAMES, "pass")
    checks.update(changes)
    return dict(status="ok", content_type=context, section=5, checks=checks,
                observations=["多人并排，部分眼部未知"], rule_ids=["W048"])


def rank(items, **changes):
    options = engine.AnalysisOptions(scene_type="婚礼", target_keep_count=len(items), **changes)
    engine._assign_tiered_ratings([items], options)
    return options


def test_bundled_catalog_has_all_240_unique_stable_ids():
    rules = policy.rules()
    assert len(rules) == 240
    assert [rule.identifier for rule in rules] == [f"W{i:03}" for i in range(1, 241)]
    assert {rule.section for rule in rules} == set(range(1, 25))
    assert policy.PROGRAM_RULE_IDS <= {rule.identifier for rule in rules}


@pytest.mark.parametrize("section", range(1, 25))
def test_every_rule_section_can_be_routed_to_model(section):
    prompt = policy.prompt_rules({section})
    assert len(prompt.splitlines()) == 10
    assert prompt.startswith(f"W{(section - 1) * 10 + 1:03}")


def test_unknown_group_eyes_are_review_only_not_best_or_backup():
    item = photo(open_eye_face_count=4)
    rank([item])
    assert item.suggested_rating == item.max_auto_rating == 1
    assert "W022" in item.rule_ids


def test_group_without_complete_smile_evidence_is_backup():
    item = photo(smiling_face_count=3)
    rank([item])
    assert item.suggested_rating < 3
    assert "W043" in item.rule_ids


@pytest.mark.parametrize("context", ["纪实 / 仪式 / 互动", "新人 / 艺术肖像", "动态抓拍", "静物 / 场地"])
def test_explicit_non_formal_content_does_not_require_smiling_or_open_eyes(context):
    item = photo(open_eye_face_count=0, smiling_face_count=0, eye_penalty=24)
    rank([item], wedding_content_type=context)
    assert item.suggested_rating == 3
    assert item.eye_penalty == 0
    assert not item.featured_blocked


def test_optional_content_model_documentary_classification_respects_eye_exemption():
    item = photo(open_eye_face_count=0, smiling_face_count=0,
                 content_evidence=evidence("documentary"))
    rank([item], content_model_enabled=True)
    assert item.suggested_rating == 3
    assert item.content_type == "documentary"


@pytest.mark.parametrize("check", content.CHECK_NAMES)
def test_model_uncertain_evidence_cannot_become_automatic_three_star(check):
    item = photo(content_evidence=evidence(**{check: "unknown"}))
    rank([item], content_model_enabled=True)
    assert item.suggested_rating == 1
    assert item.featured_blocked


@pytest.mark.parametrize("check", content.CHECK_NAMES)
def test_model_negative_evidence_stays_review_not_hard_delete(check):
    item = photo(content_evidence=evidence(**{check: "fail"}))
    rank([item], content_model_enabled=True)
    assert item.suggested_rating == 1
    assert item.path == Path("frame.jpg")


def test_failed_model_does_not_turn_a_technically_good_photo_into_reject():
    item = photo(content_evidence={"status": "error", "error": "timed out"})
    rank([item], content_model_enabled=True)
    assert item.suggested_rating > 0
    assert any("未完成" in reason for reason in item.reasons)


def test_wedding_default_does_not_refill_duplicate_minimum():
    items = [photo(f"frame{i}.jpg") for i in range(8)]
    rank(items, max_duplicate_picks_per_group=1)
    assert sum(item.suggested_rating > 0 for item in items) == 1
    assert "目标差 7 张" in policy.selection_summary(items, 8)


def test_explicit_fill_checkbox_can_still_refill_duplicate_minimum():
    items = [photo(f"frame{i}.jpg") for i in range(8)]
    rank(items, max_duplicate_picks_per_group=1, fill_with_similar=True)
    assert all(item.suggested_rating > 0 for item in items)


def test_must_keep_label_does_not_select_entire_wedding_burst():
    items = [photo(f"C4/交接{i}.jpg") for i in range(8)]
    rank(items, max_duplicate_picks_per_group=1)
    assert sum(item.suggested_rating > 0 for item in items) == 1


def test_rules_never_override_manual_rating():
    item = photo(open_eye_face_count=0)
    rank([item])
    EventCullApp._restore_manual_ratings([item], {item.path: 3}, "zh")
    assert item.suggested_rating == 3


def test_cache_base_reasons_are_not_mutated_by_ranking(tmp_path):
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"fixture")
    options = engine.AnalysisOptions(scene_type="婚礼")
    item = photo(str(path))
    signature = engine._cache_signature(path, options)
    record = engine._cache_record(item, signature)
    engine._assign_tiered_ratings([[item]], options)
    assert record["reasons"] == []
    assert engine._result_from_cache(path, [path], record, signature).rule_ids == []


@pytest.mark.parametrize("malformed", [None, [], {"status": "ok"},
    {"status": "ok", "checks": {}}, {"status": "ok", "checks": "pass"}])
def test_malformed_cached_content_is_discarded(tmp_path, malformed):
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"fixture")
    options = engine.AnalysisOptions(scene_type="婚礼")
    signature = engine._cache_signature(path, options)
    record = engine._cache_record(photo(str(path)), signature)
    record["content_evidence"] = malformed
    assert engine._result_from_cache(path, [path], record, signature) is None


def test_local_model_reuses_single_decode(monkeypatch, tmp_path):
    path = tmp_path / "raw.ARW"
    path.write_bytes(b"fixture")
    image = np.full((80, 120, 3), 127, dtype=np.uint8)
    decoded, inspected = [], []
    monkeypatch.setattr(engine, "_load_preview", lambda source: decoded.append(source) or image)
    monkeypatch.setattr(engine, "_face_metrics", lambda *_args: ([], 150, 5, 5, 5, 127))
    class Model:
        def inspect(self, preview, context):
            inspected.append(preview)
            return evidence()
    engine._analyze(path, [path], engine.AnalysisOptions(scene_type="婚礼"), Model())
    assert decoded == [path]
    assert len(inspected) == 1 and inspected[0] is image


def test_default_analysis_does_not_connect_model(monkeypatch, tmp_path):
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(engine, "_analyze", lambda p, pairs, options: photo(str(p)))
    monkeypatch.setattr(engine, "_load_analysis_cache", lambda: {})
    monkeypatch.setattr(engine, "_write_analysis_cache", lambda _cache: None)
    def forbidden(*_args):
        raise AssertionError("default path must not contact local or cloud models")
    monkeypatch.setattr(engine, "LocalWeddingContentModel", forbidden)
    assert len(engine.analyze_paths([path], engine.AnalysisOptions(scene_type="婚礼"))) == 1


def stub_local_service(monkeypatch, response=None):
    calls = []
    def request(path, payload=None, **_kwargs):
        calls.append((path, payload))
        if path == "/api/tags":
            return {"models": [{"name": content.MODEL_NAMES[0], "digest": "testdigest"}]}
        value = ({"content_type": "formal_group", "section": 5}
                 if payload["format"] is content.ROUTE_SCHEMA else (response or evidence()))
        return {"message": {"content": json.dumps(value, ensure_ascii=False)}}
    monkeypatch.setattr(content, "_json_request", request)
    return calls


def test_local_model_routes_rules_and_validates_observations(monkeypatch):
    calls = stub_local_service(monkeypatch)
    client = content.LocalWeddingContentModel(content.MODEL_NAMES[0])
    result = client.inspect(np.zeros((80, 100, 3), dtype=np.uint8))
    assert result["status"] == "ok"
    assert len(calls) == 3
    assert calls[1][1]["messages"][1]["images"] == calls[2][1]["messages"][1]["images"]
    prompt = calls[2][1]["messages"][1]["content"]
    assert "W041" in prompt and "W240" in prompt
    assert "testdigest" in client.fingerprint and policy.RULE_VERSION in client.fingerprint


@pytest.mark.parametrize("invalid", [
    {"checks": {}},
    {"checks": dict.fromkeys(content.CHECK_NAMES, True)},
    {"checks": dict.fromkeys(content.CHECK_NAMES, "pass"), "observations": ["x" * 161], "rule_ids": []},
    {"checks": dict.fromkeys(content.CHECK_NAMES, "pass"), "observations": [], "rule_ids": ["W999"]},
    {"checks": dict.fromkeys(content.CHECK_NAMES, "pass"), "observations": [], "rule_ids": ["W071"]},
])
def test_invalid_model_output_falls_back_not_stars(monkeypatch, invalid):
    stub_local_service(monkeypatch, invalid)
    client = content.LocalWeddingContentModel(content.MODEL_NAMES[0])
    assert client.inspect(np.zeros((80, 100, 3), dtype=np.uint8))["status"] == "error"


def test_missing_local_model_does_not_send_any_images(monkeypatch):
    calls = []
    monkeypatch.setattr(content, "_json_request", lambda path, **kwargs: calls.append(path) or {"models": []})
    client = content.LocalWeddingContentModel(content.MODEL_NAMES[0])
    assert not client.ready
    assert client.inspect(np.zeros((80, 100, 3), dtype=np.uint8))["status"] == "unavailable"
    assert calls == ["/api/tags"]


def test_cloud_model_name_is_rejected_without_network(monkeypatch):
    monkeypatch.setattr(content, "_json_request", lambda *_args, **_kwargs: pytest.fail("network forbidden"))
    assert not content.LocalWeddingContentModel("qwen3-vl:cloud").ready


def test_cancel_waiting_model_worker_does_not_wait_for_inference(monkeypatch):
    stub_local_service(monkeypatch)
    cancel = threading.Event()
    client = content.LocalWeddingContentModel(content.MODEL_NAMES[0], cancel)
    client.lock.acquire()
    cancel.set()
    try:
        with pytest.raises(content.ContentModelCancelled):
            client.inspect(np.zeros((80, 100, 3), dtype=np.uint8))
    finally:
        client.lock.release()


def test_two_service_failures_stop_repeated_timeout_loop(monkeypatch):
    calls = stub_local_service(monkeypatch)
    client = content.LocalWeddingContentModel(content.MODEL_NAMES[0])
    def timeout(*args, **kwargs):
        calls.append("timeout")
        raise TimeoutError("fixture timeout")
    monkeypatch.setattr(content, "_json_request", timeout)
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    assert client.inspect(image)["status"] == "error"
    assert client.inspect(image)["status"] == "error"
    assert client.inspect(image)["status"] == "unavailable"
    assert calls.count("timeout") == 2


def test_model_errors_are_not_persisted_to_feature_cache(monkeypatch, tmp_path):
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"fixture")
    model = type("Model", (), {"fingerprint": "fixture-model"})()
    monkeypatch.setattr(engine, "_analyze", lambda p, *_args: photo(str(p), content_evidence={"status": "error"}))
    item, key, record = engine._analyze_cached(path, [path], engine.AnalysisOptions(), {}, model)
    assert record is None


def test_log_records_rule_and_content_evidence():
    item = photo(content_evidence=evidence())
    rank([item], content_model_enabled=True)
    record = EventCullApp._log_record(item, 1)
    assert record["rule_version"] == policy.RULE_VERSION
    assert "W048" in record["rule_ids"]
    assert json.loads(record["content_evidence"])["status"] == "ok"

"""Diagnostic integrity, not evidence of wedding selection accuracy."""
import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.benchmark_burst_review import comparison_canvas, review_group, visual_controls, face_details


def rows(tmp_path, count=4):
    result = []
    for index in range(count):
        path = tmp_path/f"private-{index}.jpg"
        path.write_bytes(b"fixture")
        result.append({"folder":str(tmp_path), "file":path.name, "group":10})
    return result


class Runtime:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def checkpoint(self):
        pass

    def chat(self, encoded, prompt, schema, *, max_tokens):
        assert "private" not in prompt and max_tokens in {80, 240}
        image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
        self.calls.append((image, prompt, schema))
        return self.answer(len(self.calls))


def test_order_changes_pixels_but_not_prompt_and_previews_are_decoded_once(tmp_path):
    records = rows(tmp_path)
    calls = []
    def loader(path, *, maximum_dimension):
        calls.append(path)
        assert maximum_dimension == 448
        return np.full((100, 100, 3), 30+30*len(calls), np.uint8)
    runtime = Runtime(lambda number: {"ordered_frames":[0, 1], "evidence_clear":True})
    result = review_group(records, runtime, permutation_check=True, loader=loader)
    assert len(calls) == 4 and result["decoded_frames"] == 4
    assert len(runtime.calls) == 2 and runtime.calls[0][1:] == runtime.calls[1][1:]
    assert runtime.calls[0][0][180, 160, 0] < runtime.calls[1][0][180, 160, 0]
    assert result["reviews"][1]["canonical_ordered_frames"] == [3, 2]
    assert result["same_selected_set"] is False and result["same_top1"] is False
    assert result["selection_is_informative"] is True


def test_true_permutation_equivariance_is_measured_by_original_identity(tmp_path):
    runtime = Runtime(lambda number: {"ordered_frames":[0, 1] if number == 1 else [3, 2],
                                      "evidence_clear":False})
    result = review_group(rows(tmp_path), runtime, permutation_check=True,
                          loader=lambda *_a, **_k: np.zeros((10, 10, 3), np.uint8))
    assert result["same_selected_set"] is True and result["same_top1"] is True
    assert result["reply"]["evidence_clear"] is False  # Not a quality certificate.


def test_selecting_every_frame_is_not_informative_set_consistency(tmp_path):
    runtime = Runtime(lambda _: {"ordered_frames":[0, 1, 2], "evidence_clear":True})
    result = review_group(rows(tmp_path, 3), runtime, permutation_check=True,
                          loader=lambda *_a, **_k: np.zeros((10, 10, 3), np.uint8))
    assert result["same_selected_set"] is True and result["same_top1"] is False
    assert result["selection_is_informative"] is False


@pytest.mark.parametrize("reply", [
    {"ordered_frames":[True], "evidence_clear":True},
    {"ordered_frames":[0, 0], "evidence_clear":True},
    {"ordered_frames":[4], "evidence_clear":True},
    {"ordered_frames":[0, 1, 2, 3], "evidence_clear":True},
    {"ordered_frames":[], "evidence_clear":1},
])
def test_malformed_reply_is_not_silently_accepted(tmp_path, reply):
    with pytest.raises(ValueError, match="Invalid local burst"):
        review_group(rows(tmp_path), Runtime(lambda _:reply),
                     loader=lambda *_a, **_k:np.zeros((10, 10, 3), np.uint8))


def test_duplicate_and_stale_inputs_fail_instead_of_comparing_wrong_photos(tmp_path):
    records = rows(tmp_path)
    runtime = Runtime(lambda _: {"ordered_frames":[], "evidence_clear":False})
    with pytest.raises(ValueError, match="Duplicate"):
        review_group([records[0], records[0]], runtime)
    def changed_loader(path, **_kwargs):
        path.write_bytes(b"changed fixture")
        return np.zeros((10, 10, 3), np.uint8)
    with pytest.raises(ValueError, match="source changed"):
        review_group(records, runtime, loader=changed_loader)


def test_invalid_permutation_or_array_is_refused():
    image = np.zeros((10, 10, 3), np.uint8)
    with pytest.raises(ValueError, match="complete permutation"):
        comparison_canvas([image, image], [0, 0])
    with pytest.raises(ValueError, match="Invalid comparison"):
        comparison_canvas([np.zeros((10, 10))], [0])


def test_visual_controls_use_known_pixels_not_photo_decoding(monkeypatch):
    from scripts import benchmark_burst_review as experiment
    monkeypatch.setattr(experiment, "_load_preview", lambda *_a, **_k:pytest.fail("no photo decoding"))
    observed = []
    class ColorRuntime:
        def checkpoint(self):
            pass
        def chat(self, encoded, prompt, schema, *, max_tokens):
            image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
            bgr = image[160, 160]
            color = "black" if bgr.max() < 5 else "white" if bgr.min() > 250 else (
                "red" if bgr[2] > 250 else "blue")
            observed.append(schema["properties"]["dominant_color"]["enum"])
            return {"dominant_color":color, "people":"none"}
    result = visual_controls(ColorRuntime())
    assert len(result) == 8 and all(record["correct"] for record in result)
    assert observed[0] == list(reversed(observed[1]))


def test_control_failure_is_reported_not_relabelled_as_correct():
    runtime = Runtime(lambda _: {"dominant_color":"unknown", "people":"unknown"})
    result = visual_controls(runtime)
    assert not any(record["correct"] for record in result)


def test_detail_crops_reuse_frozen_boxes_without_any_detector():
    from src.core.image_features import human_analysis as human
    image = np.full((200, 200, 3), (10, 20, 240), dtype=np.uint8)
    observation = {"status":"ok", "pipeline":human.PIPELINE_ID,
                   "faces":[{"bbox":[.2, .2, .5, .5]}]}
    crops = face_details(image, observation)
    assert len(crops) == 1 and np.array_equal(crops[0][0, 0], [10, 20, 240])
    assert face_details(image, {**observation, "faces":[]}) == []
    with pytest.raises(ValueError, match="Frozen face"):
        face_details(image, None)


def test_detailed_evidence_is_requested_once_per_order_without_another_decode(tmp_path):
    from src.core.image_features import human_analysis as human
    records = rows(tmp_path, 3)
    for record in records:
        record["person_evidence"] = {"status":"ok", "pipeline":human.PIPELINE_ID,
                                    "faces":[{"bbox":[.2, .2, .8, .8]}]}
    decoded = []
    def loader(path, *, maximum_dimension):
        assert maximum_dimension == 1600
        decoded.append(path)
        return np.zeros((200, 200, 3), dtype=np.uint8)
    runtime = Runtime(lambda _: {"ordered_frames":[], "evidence_clear":False,
                                 "visible_differences":"Cannot establish an expression difference."})
    result = review_group(records, runtime, permutation_check=True, detail_view=True,
                          evidence_first=True, loader=loader)
    assert len(decoded) == 3 and result["face_crops"] == 3
    assert runtime.calls[0][0].shape == (992, 1920, 3)
    assert "visible_differences" in runtime.calls[0][2]["required"]
    assert result["reply"]["evidence_clear"] is False

from pathlib import Path

from scripts.compare_selection_by_capture import base_stem, matching_candidates


def capture(path, timestamp="2026:10:11 09:00:43", subsecond="487"):
    return {"SourceFile": path, "DateTimeOriginal": timestamp,
            "SubSecTimeOriginal": subsecond, "Model": "ILCE-7M4"}


def test_duplicate_name_from_other_time_is_never_guessed():
    morning = capture("/morning/DSC08849.ARW")
    afternoon = capture("/afternoon/DSC08849.ARW", "2026:10:11 17:18:50", "425")
    delivery = capture("/manual/DSC08849_副本.jpg", "2026:10:11 17:18:50", "425")
    assert base_stem(Path(delivery["SourceFile"])) == "dsc08849"
    assert matching_candidates(delivery, [morning, afternoon]) == [afternoon]


def test_same_second_but_different_subseconds_remains_distinct():
    assert not matching_candidates(capture("/manual/a.jpg", subsecond="488"), [capture("/raw/a.ARW")])


def test_missing_capture_time_does_not_match_by_filename_alone():
    delivery = capture("/manual/a.jpg")
    delivery.pop("DateTimeOriginal")
    assert not matching_candidates(delivery, [capture("/raw/a.ARW")])


def test_two_equal_candidates_remain_ambiguous():
    a, b = capture("/one/a.ARW"), capture("/two/a.ARW")
    assert len(matching_candidates(capture("/manual/a.jpg"), [a, b])) == 2


def test_conflicting_available_camera_serials_prevent_match():
    a, b = capture("/manual/a.jpg"), capture("/raw/a.ARW")
    a["SerialNumber"], b["SerialNumber"] = "one", "two"
    assert not matching_candidates(a, [b])

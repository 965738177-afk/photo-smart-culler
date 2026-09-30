from pathlib import Path
import pytest

from event_cull.case_inventory import inventory_case,verify_reference_copies,verify_capture_metadata
from event_cull.source_scan import scan_source_images


def files(tmp_path):
    source,selected=tmp_path/"source",tmp_path/"selected"
    source.mkdir();selected.mkdir()
    for name,content in (("a.ARW",b"RAW a"),("a.JPG",b"camera a"),("b.ARW",b"RAW b"),("b.JPG",b"camera b")):
        (source/name).write_bytes(content)
    (selected/"a.JPG").write_bytes(b"camera a")
    return source,selected


def test_pairs_are_single_capture_and_unknown_is_not_negative(tmp_path):
    source,selected=files(tmp_path)
    report=inventory_case(scan_source_images([source]),list(selected.iterdir()),case_id="wedding-one")
    assert report["logical_photos"]==2 and report["physical_source_files"]==4
    assert report["reference_unique_captures"]==1
    assert sorted((row["human_retained"] for row in report["records"]),key=str)==[None,True]
    assert all(row["ai_rating"] is None for row in report["records"])
    assert not report["training_performed"] and not report["model_inference_performed"]
    assert not report["full_selection_acceptance_eligible"]
    complete=inventory_case(scan_source_images([source]),list(selected.iterdir()),case_id="wedding-one",confirmed_complete=True)
    assert sum(row["human_retained"] is False for row in complete["records"])==1
    assert complete["full_selection_acceptance_eligible"]


def test_case_identity_prevents_cross_wedding_filename_collision(tmp_path):
    source,selected=files(tmp_path)
    first=inventory_case(scan_source_images([source]),list(selected.iterdir()),case_id="first")
    second=inventory_case(scan_source_images([source]),list(selected.iterdir()),case_id="second")
    assert not {r["identity"] for r in first["records"]}&{r["identity"] for r in second["records"]}


def test_preculled_source_is_positive_control_including_missing_exports(tmp_path):
    source,selected=files(tmp_path)
    report=inventory_case(scan_source_images([source]),list(selected.iterdir()),case_id="curated",preculled_source=True)
    assert all(row["human_retained"] is True for row in report["records"])
    assert all(row["human_label_basis"]=="user_confirmed_preculled_source" for row in report["records"])
    assert report["reference_unique_captures"]==1 and report["logical_photos"]==2
    assert not report["full_selection_acceptance_eligible"]
    assert not report["reference_confirmed_complete"]
    with pytest.raises(ValueError,match="Preculled"):
        inventory_case(scan_source_images([source]),list(selected.iterdir()),case_id="curated",preculled_source=True,confirmed_complete=True)


def test_reference_scan_ignores_hidden_metadata_files(tmp_path):
    from event_cull.calibration import exported_selection_files
    (tmp_path/"a.JPG").touch();(tmp_path/"._a.JPG").touch()
    assert exported_selection_files(tmp_path)==(tmp_path/"a.JPG",)


def test_reference_bytes_compare_to_jpeg_buddy_without_decoder(monkeypatch,tmp_path):
    from event_cull import engine
    monkeypatch.setattr(engine,"_load_preview",lambda *a,**k:pytest.fail("identity checks must not decode"))
    source,selected=files(tmp_path)
    report=inventory_case(scan_source_images([source]),list(selected.iterdir()),case_id="wedding")
    verified=verify_reference_copies(report)
    assert verified["same_bytes"]==1 and verified["different_bytes"]==0 and verified["images_decoded"]==0
    assert verified["records"][0]["source_companion"].endswith("a.JPG")
    assert (source/"a.ARW").read_bytes()==b"RAW a" and (source/"a.JPG").read_bytes()==b"camera a"


def test_changed_reference_refused_and_different_edited_bytes_explicit(tmp_path):
    source,selected=files(tmp_path)
    report=inventory_case(scan_source_images([source]),list(selected.iterdir()),case_id="wedding")
    (selected/"a.JPG").write_bytes(b"edited JPEG with other bytes")
    with pytest.raises(ValueError,match="changed"):
        verify_reference_copies(report)
    report=inventory_case(scan_source_images([source]),list(selected.iterdir()),case_id="wedding")
    assert verify_reference_copies(report)["different_bytes"]==1


def test_identity_hashing_is_cancellable(tmp_path):
    source,selected=files(tmp_path)
    report=inventory_case(scan_source_images([source]),list(selected.iterdir()),case_id="wedding")
    def stop():
        raise InterruptedError()
    with pytest.raises(InterruptedError):
        verify_reference_copies(report,checkpoint=stop)


def test_unmatched_or_ambiguous_names_refused(tmp_path):
    source,selected=files(tmp_path)
    unknown=selected/"c.JPG";unknown.touch()
    with pytest.raises(ValueError,match="Ambiguous or unmatched"):
        inventory_case(scan_source_images([source]),list(selected.iterdir()),case_id="wedding")
    other=tmp_path/"other";other.mkdir();(other/"a.JPG").touch()
    with pytest.raises(ValueError,match="Ambiguous or unmatched"):
        inventory_case(scan_source_images([source]),[selected/"a.JPG",other/"a.JPG"],case_id="wedding")


def test_direct_scan_does_not_recurse_and_ignores_appledouble(tmp_path):
    (tmp_path/"a.ARW").touch();(tmp_path/"._a.ARW").write_bytes(b"resource fork")
    child=tmp_path/"child";child.mkdir();(child/"b.ARW").touch()
    assert scan_source_images([tmp_path],recursive=False)==[tmp_path/"a.ARW"]
    assert set(scan_source_images([tmp_path]))=={tmp_path/"a.ARW",child/"b.ARW"}


def capture_reader(paths):
    return [{"SourceFile":str(path),"DateTimeOriginal":"2026:09:28 12:00:00","SubSecTimeOriginal":0,
             "Model":"camera","RawFileName":path.stem+".ARW"} for path in paths]


def test_capture_metadata_is_bounded_and_does_not_decode_or_change_labels(tmp_path,monkeypatch):
    from event_cull import engine
    monkeypatch.setattr(engine,"_load_preview",lambda *a,**k:pytest.fail("metadata must not decode"))
    source,selected=files(tmp_path)
    report=inventory_case(scan_source_images([source]),list(selected.iterdir()),case_id="wedding")
    before=[r["human_retained"] for r in report["records"]]
    calls=[]
    def reader(paths):
        calls.append(paths)
        return capture_reader(paths)
    checked=verify_capture_metadata(report,reader=reader,batch_size=1)
    assert all(len(paths)==1 for paths in calls) and checked["files_read"]==2
    assert checked["status_counts"]=={"consistent_fine_capture_metadata":1}
    assert checked["records"][0]["checks"]["SubSecTimeOriginal"] is True
    assert not checked["pixel_identity_proven"] and not checked["human_labels_modified"]
    assert before==[r["human_retained"] for r in report["records"]]


def test_missing_capture_metadata_is_unknown_not_success_or_bad_picture(tmp_path):
    source,selected=files(tmp_path)
    report=inventory_case(scan_source_images([source]),list(selected.iterdir()),case_id="wedding")
    checked=verify_capture_metadata(report,reader=lambda paths:[{"SourceFile":str(p)} for p in paths])
    assert checked["status_counts"]=={"unknown":1}
    assert all(v is None for v in checked["records"][0]["checks"].values())


def test_capture_contradiction_is_explicit_without_relabelling(tmp_path):
    source,selected=files(tmp_path)
    report=inventory_case(scan_source_images([source]),list(selected.iterdir()),case_id="wedding")
    def reader(paths):
        rows=capture_reader(paths)
        for row in rows:
            if Path(row["SourceFile"]).parent==selected:
                row["RawFileName"]="different.ARW"
        return rows
    checked=verify_capture_metadata(report,reader=reader)
    assert checked["status_counts"]=={"contradicted":1}
    assert checked["records"][0]["checks"]["RawFileName"] is False
    assert report["records"][0]["human_retained"] is True


@pytest.mark.parametrize("mode",["missing","duplicate","foreign","changed"])
def test_metadata_subset_or_mutated_files_are_refused(tmp_path,mode):
    source,selected=files(tmp_path)
    report=inventory_case(scan_source_images([source]),list(selected.iterdir()),case_id="wedding")
    def reader(paths):
        rows=capture_reader(paths)
        if mode=="missing":return rows[:-1]
        if mode=="duplicate":return rows+rows[:1]
        if mode=="foreign":rows[0]["SourceFile"]=str(tmp_path/"unrequested.JPG")
        if mode=="changed":(selected/"a.JPG").write_bytes(b"modified during read")
        return rows
    with pytest.raises(ValueError):
        verify_capture_metadata(report,reader=reader)

from pathlib import Path
import pytest
from scripts.benchmark_event_cull import sampled_paths,sample_logical_photos,frozen_visual_features


def test_benchmark_subsample_spans_entire_wedding_with_scaled_quantity():
    paths = [Path(f"{i}.ARW") for i in range(5049)]
    selected, floor, ceiling = sampled_paths(paths,1000,sample_mode="evenly")
    assert len(selected) == len(set(selected)) == 1000
    assert selected[0] == paths[0] and selected[-1] == paths[-1]
    assert (floor,ceiling) == (203,238)


def test_full_inventory_preserves_requested_5049_budget():
    paths = [Path(f"{i}.ARW") for i in range(5049)]
    selected, floor, ceiling = sampled_paths(paths,None)
    assert selected == paths and (floor,ceiling) == (1023,1200)


def test_budget_never_exceeds_actual_photo_count():
    selected, floor, ceiling = sampled_paths([Path("a.jpg")],1000)
    assert len(selected) == floor == ceiling == 1


def test_empty_inventory_rejected():
    with pytest.raises(ValueError):
        sampled_paths([],None)


def test_pairs_are_collapsed_before_limit_not_counted_as_two_photos(tmp_path):
    files=[]
    for index in range(20):
        for ext in ("ARW","JPG"):
            p=tmp_path/f"{index:03}.{ext}"
            p.touch()
            files.append(p)
    selected,floor,ceiling=sample_logical_photos(files,10,sample_mode="evenly",keep_min=8,keep_max=10)
    assert len(selected)==10 and all(p.suffix==".ARW" for p in selected)
    assert len({p.stem for p in selected})==10 and (floor,ceiling)==(4,5)


def test_new_wedding_explicit_budget_not_inherited_from_old_5049_case():
    selected,floor,ceiling=sampled_paths([Path(f"{i}.ARW") for i in range(2575)],1000,
                                      keep_min=870,keep_max=870)
    assert len(selected)==1000 and floor==ceiling==338


@pytest.mark.parametrize("floor,ceiling",[(0,1),(2,1),(1.5,2),(True,2)])
def test_invalid_explicit_benchmark_budgets_rejected(floor,ceiling):
    with pytest.raises(ValueError):
        sampled_paths([Path("a.ARW")],None,keep_min=floor,keep_max=ceiling)


def test_feature_archive_reuses_existing_order_and_rejects_missing_rows():
    from types import SimpleNamespace
    import numpy as np
    rows=[SimpleNamespace(path=Path(f"{i}.ARW"), visual_features=[i,1.0], visual_feature_pipeline="frozen-v1") for i in range(3)]
    frozen=frozen_visual_features(rows,"frozen-v1")
    assert frozen["features"].shape==(3,2)
    assert list(frozen["paths"])==["0.ARW","1.ARW","2.ARW"]
    rows[1].visual_features=[]
    with pytest.raises(ValueError,match="dimensions"):
        frozen_visual_features(rows,"frozen-v1")
    rows[1].visual_features=[np.nan,1]
    with pytest.raises(ValueError,match="nonfinite"):
        frozen_visual_features(rows,"frozen-v1")
    rows[1].visual_feature_pipeline="wrong"
    with pytest.raises(ValueError,match="pipeline"):
        frozen_visual_features(rows,"frozen-v1")

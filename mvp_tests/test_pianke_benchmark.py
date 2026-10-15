from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.benchmark_pianke_fast import measure, native_ranker


def test_metrics_compare_unique_capture_paths_not_basenames():
    result = measure(["/am/a.ARW", "/pm/a.ARW", "/am/a.ARW"], {"/pm/a.ARW", "/am/b.ARW"})
    assert result["retained"] == 2 and result["exact"] == 1
    assert result["precision"] == result["recall"] == .5
    assert result["human_only"] == result["extra_not_delivered"] == 1


def test_native_scoring_is_loaded_without_importing_flask_or_key_setup(tmp_path):
    functions = '\n'.join(f'def {name}(info): return 1' for name in (
        '_quality_score', '_aesthetic_score', '_face_quality_score', '_subject_sharpness', '_quality_flags'))
    # Deliberate module-level failure proves no top-level app code is run.
    (tmp_path / 'app.py').write_text('raise RuntimeError("auth must not be read")\n' + functions +
                                    '\ndef _composite_score(info, main_subject_present=False):\n'
                                    '    return _quality_score(info) + 2\n')
    rank = native_ranker(tmp_path)
    assert rank(SimpleNamespace()) == 3


def test_changed_native_contract_fails_instead_of_using_our_own_formula(tmp_path):
    (tmp_path / 'app.py').write_text('def _composite_score(info): return 0\n')
    with pytest.raises(ValueError, match='contract changed'):
        native_ranker(tmp_path)

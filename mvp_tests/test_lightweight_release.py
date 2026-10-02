"""Guard the supported fast installers against accidental GGUF rebundling."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_standard_desktop_specs_do_not_collect_optional_content_runtime():
    for filename in ("照片原图快速筛选器.spec", "MiaoXuan-Windows.spec"):
        source = (ROOT / filename).read_text(encoding="utf-8")
        assert "build_asset_tuples" not in source
        assert "vendor/content" not in source
        assert "binaries = []" in source


def test_windows_release_does_not_download_optional_content_runtime():
    workflow = (ROOT / ".github/workflows/windows-culler.yml").read_text(encoding="utf-8")
    assert "prepare_bundled_content.py" not in workflow


def test_fast_desktop_ui_cannot_offer_or_enable_unshipped_model():
    source = (ROOT / "event_cull/app.py").read_text(encoding="utf-8")
    assert "content_model_enabled=False," in source
    assert '"content_model_enabled": False,' in source
    assert "command=self.check_content_model" not in source

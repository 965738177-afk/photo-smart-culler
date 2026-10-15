from event_cull.build_info import APP_VERSION


def test_release_version_starts_at_v2_and_matches_current_build():
    assert APP_VERSION == "2.1.30"

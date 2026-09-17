from event_cull.app import EventCullApp


def test_output_star_mapping_keeps_engine_tiers_but_supports_four_and_five_stars():
    assert [EventCullApp._mapped_rating(tier, 2, 3) for tier in range(4)] == [0, 1, 2, 3]
    assert [EventCullApp._mapped_rating(tier, 4, 5) for tier in range(4)] == [0, 1, 4, 5]

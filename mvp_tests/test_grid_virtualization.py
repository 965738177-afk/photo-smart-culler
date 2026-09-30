"""Large review walls must use bounded physical widgets, not giant backing views."""
from pathlib import Path
from types import SimpleNamespace

from event_cull.app import EventCullApp


def test_large_grid_keeps_physical_frame_small_and_rows_in_local_coordinates():
    subject = object.__new__(EventCullApp)
    subject._virtual_grid_layout = (5, 218, 144, 360)
    subject._virtual_grid_entries = [(i, Path(f"IMG{i}.ARW"), [], str(i), 1, []) for i in range(2764)]
    subject._virtual_grid_range = None
    subject._virtual_grid_rows = {}
    subject._grid_thumbnail_cache = {}
    subject._photo_grid_window = "window"
    dimensions = {}
    positions = []
    subject.photo_grid_inner = SimpleNamespace(configure=lambda **kw: dimensions.update(kw))
    subject.photo_grid_canvas = SimpleNamespace(
        canvasy=lambda _y: 100000,
        winfo_height=lambda: 700,
        itemconfigure=lambda *_args, **kw: None,
        coords=lambda *args: positions.append(args),
    )
    offsets = []
    subject._append_grid_cards = lambda entries, **kw: offsets.append(kw["virtual_row_offset"]) or []
    subject._render_virtual_grid_window()
    assert dimensions["height"] < 4000
    first, last = subject._virtual_grid_range
    assert 0 <= first < last <= 553
    assert positions == [("window", 0, first * 360)]
    assert all(offset == first for offset in offsets)


def test_scroll_position_beyond_new_smaller_result_set_never_leaves_grid_empty():
    subject = object.__new__(EventCullApp)
    subject._virtual_grid_layout = (5, 218, 144, 360)
    subject._virtual_grid_entries = [(0, Path("IMG.ARW"), [], "IMG", 1, [])]
    subject._virtual_grid_range = None
    subject._virtual_grid_rows = {}
    subject._grid_thumbnail_cache = {}
    subject._photo_grid_window = "window"
    subject.photo_grid_inner = SimpleNamespace(configure=lambda **kw: None)
    subject.photo_grid_canvas = SimpleNamespace(
        canvasy=lambda _y: 900000, winfo_height=lambda: 700,
        itemconfigure=lambda *_args, **kw: None, coords=lambda *args: None,
    )
    rendered = []
    subject._append_grid_cards = lambda entries, **kw: rendered.extend(entries) or []
    subject._render_virtual_grid_window()
    assert len(rendered) == 1
    assert subject._virtual_grid_range == (0, 1)


def test_inner_resize_preserves_virtual_scroll_extent_but_updates_list_extent():
    subject = object.__new__(EventCullApp)
    updates = []
    subject.photo_grid_canvas = SimpleNamespace(
        bbox=lambda _tag: (0, 0, 620, 1800),
        configure=lambda **kw: updates.append(kw),
    )
    subject._virtual_grid_layout = (3, 174, 144, 360)
    subject._on_grid_inner_configure()
    assert updates == []
    subject._virtual_grid_layout = None
    subject._on_grid_inner_configure()
    assert updates == [{"scrollregion": (0, 0, 620, 1800)}]

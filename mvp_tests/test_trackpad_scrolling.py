"""Tk 9 precise trackpad input and legacy wheel events share one route."""
from types import SimpleNamespace

import pytest
import tkinter as tk

from event_cull.app import EventCullApp


class Canvas:
    def __init__(self):
        self.increment = 12
        self.moves = []

    def cget(self, _option):
        return self.increment

    def configure(self, **options):
        self.increment = options["yscrollincrement"]

    def yview_scroll(self, amount, unit):
        self.moves.append((amount, unit))


def packed(x, y):
    return ((x & 0xffff) << 16) | (y & 0xffff)


@pytest.mark.parametrize("x,y", [(0, 1), (0, -1), (7, -30), (-15, 6), (14, 0), (0, -32768), (0, 32767)])
def test_precise_vertical_delta_preserves_signed_pixels_and_ignores_x(x, y):
    assert EventCullApp._touchpad_vertical_delta(packed(x, y)) == y


def test_precise_input_does_not_amplify_tiny_trackpad_movements():
    app = object.__new__(EventCullApp)
    canvas = Canvas()
    for _ in range(10):
        assert app._scroll_canvas(SimpleNamespace(delta=packed(0, -1)), canvas, precise=True) == "break"
    assert canvas.moves == [(1, "units")] * 10
    assert canvas.increment == 1
    # Switching back to a wheel still moves the original twelve pixels.
    app._scroll_canvas(SimpleNamespace(delta=-120), canvas)
    assert canvas.moves[-1] == (12, "units")
    app._scroll_canvas(SimpleNamespace(delta=240), canvas)
    assert canvas.moves[-1] == (-24, "units")


@pytest.mark.parametrize("number,expected", [(4, -12), (5, 12)])
def test_linux_wheel_direction_matches_mousewheel(number, expected):
    app = object.__new__(EventCullApp)
    canvas = Canvas()
    app._scroll_canvas(SimpleNamespace(num=number, delta=0), canvas)
    assert canvas.moves == [(expected, "units")]


def test_horizontal_only_event_does_not_move_vertical_scroll():
    app = object.__new__(EventCullApp)
    canvas = Canvas()
    assert app._scroll_canvas(SimpleNamespace(delta=packed(-5, 0)), canvas, precise=True) is None
    assert not canvas.moves and canvas.increment == 12


def test_modal_settings_owns_scroll_before_background_surfaces():
    app = object.__new__(EventCullApp)
    app._picker_dialog = object()
    app._picker_form_canvas = Canvas()
    app.photo_grid_canvas = Canvas()
    app.decision_canvas = Canvas()
    app._pointer_is_over_widget = lambda *_args: True
    assert app._route_mousewheel(SimpleNamespace(delta=packed(0, -3)), precise=True) == "break"
    assert app._picker_form_canvas.moves == [(3, "units")]
    assert not app.photo_grid_canvas.moves and not app.decision_canvas.moves


def test_pointer_outside_modal_never_scrolls_background():
    app = object.__new__(EventCullApp)
    app._picker_dialog = object()
    app._picker_form_canvas = Canvas()
    app.photo_grid_canvas = Canvas()
    app._pointer_is_over_widget = lambda _event, widget: widget is app.photo_grid_canvas
    assert app._route_mousewheel(SimpleNamespace(delta=-1)) is None
    assert not app.photo_grid_canvas.moves


def test_closed_modal_returns_ownership_to_workbench():
    app = object.__new__(EventCullApp)
    app._picker_dialog = None
    app._picker_form_canvas = None
    app.photo_grid_canvas = Canvas()
    app._pointer_is_over_widget = lambda *_args: True
    app._route_mousewheel(SimpleNamespace(delta=-1))
    assert app.photo_grid_canvas.moves == [(12, "units")]


def test_old_tk_ignores_unsupported_precise_binding_but_retains_wheel():
    app = object.__new__(EventCullApp)
    installed = {}

    def bind(sequence, callback, **_kwargs):
        if sequence == "<TouchpadScroll>":
            raise tk.TclError("bad event type")
        installed[sequence] = callback

    canvas = Canvas()
    app._bind_canvas_scrolling(SimpleNamespace(bind=bind), canvas)
    assert set(installed) == {"<MouseWheel>", "<Button-4>", "<Button-5>"}
    assert installed["<MouseWheel>"](SimpleNamespace(delta=-1)) == "break"


def test_destroyed_canvas_does_not_break_event_processing():
    app = object.__new__(EventCullApp)
    canvas = Canvas()
    canvas.yview_scroll = lambda *_args: (_ for _ in ()).throw(tk.TclError("destroyed"))
    assert app._scroll_canvas(SimpleNamespace(delta=-1), canvas) is None

from types import SimpleNamespace
from pathlib import Path

from event_cull.app import EventCullApp


def test_lightroom_guard_matches_only_the_exact_main_process(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("event_cull.app.subprocess.run", fake_run)
    assert EventCullApp._lightroom_is_running() is False
    assert calls == [["pgrep", "-x", "Adobe Lightroom Classic"]]


class FakeDropWidget:
    def __init__(self, *children):
        self.children = list(children)
        self.registered = []
        self.bindings = []

    def drop_target_register(self, *types):
        self.registered.extend(types)

    def dnd_bind(self, event_name, callback):
        self.bindings.append((event_name, callback))

    def winfo_children(self):
        return self.children


def test_drop_registration_covers_the_complete_widget_tree():
    leaf = FakeDropWidget()
    branch = FakeDropWidget(leaf)
    root = FakeDropWidget(branch)
    app = object.__new__(EventCullApp)
    app.accept_drop = lambda _event: "break"

    EventCullApp._register_drop_targets(app, root)

    for widget in (root, branch, leaf):
        assert widget.registered
        assert [event for event, _callback in widget.bindings] == ["<<Drop>>"]


def test_drop_is_rejected_while_analysis_is_running(monkeypatch):
    shown = []
    monkeypatch.setattr(
        "event_cull.app.messagebox.showinfo",
        lambda title, detail: shown.append((title, detail)),
    )
    app = object.__new__(EventCullApp)
    app.analysis_running = True
    app.t = {
        "drop_busy_title": "正在分析",
        "drop_busy_detail": "请先中止当前分析。",
    }

    assert EventCullApp.accept_drop(app, SimpleNamespace(data="ignored")) == "break"
    assert shown == [("正在分析", "请先中止当前分析。")]


def test_macos_folder_picker_requests_multiple_folders_and_restores_last_location(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="/photos/one/\x1e/photos/two/\n")

    monkeypatch.setattr("event_cull.app.subprocess.run", fake_run)

    selected = EventCullApp._choose_folders_macos(Path("/photos"), "选择一个或多个原片文件夹")

    assert selected == [Path("/photos/one"), Path("/photos/two")]
    command, kwargs = calls[0]
    assert command[0] == "/usr/bin/osascript"
    assert "multiple selections allowed" in command[2]
    assert command[-1] == "/photos"
    assert kwargs["capture_output"] is True
    assert kwargs["env"]["AppleLanguages"] == "(zh_CN)"
    assert kwargs["env"]["AppleLocale"] == "zh_CN"

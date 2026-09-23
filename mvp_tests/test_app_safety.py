from types import SimpleNamespace
from pathlib import Path

from event_cull.app import COPY, REFUSE_DROP, EventCullApp


def test_metadata_editor_guard_ignores_lightroom_crash_helpers(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="/Applications/Adobe Lightroom Classic/Adobe Lightroom Classic.app/Contents/Frameworks/AdobeCrashReporter.framework/Versions/A/CrashHandler\n",
        )

    monkeypatch.setattr("event_cull.app.subprocess.run", fake_run)
    assert EventCullApp._metadata_editors_running() == ()
    assert calls == [["ps", "-axo", "comm="]]


def test_metadata_editor_guard_detects_each_supported_photo_editor(monkeypatch):
    monkeypatch.setattr(
        "event_cull.app.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="\n".join((
                "/Applications/Adobe Lightroom Classic/Adobe Lightroom Classic.app/Contents/MacOS/Adobe Lightroom Classic",
                "/Applications/Adobe Photoshop 2026/Adobe Photoshop 2026.app/Contents/MacOS/Adobe Photoshop 2026",
                "/Applications/像素蛋糕.app/Contents/MacOS/像素蛋糕",
            )),
        ),
    )

    assert EventCullApp._metadata_editors_running() == (
        "Lightroom Classic", "Photoshop", "像素蛋糕",
    )


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
        assert [event for event, _callback in widget.bindings] == [
            "<<DropEnter>>",
            "<<DropPosition>>",
            "<<DropLeave>>",
            "<<Drop>>",
        ]


def test_drop_is_rejected_while_analysis_is_running(monkeypatch):
    shown = []
    monkeypatch.setattr(
        "event_cull.app.messagebox.showinfo",
        lambda title, detail: shown.append((title, detail)),
    )
    app = object.__new__(EventCullApp)
    app.analysis_running = True
    app.calibration_running = False
    app.t = {
        "drop_busy_title": "正在分析",
        "drop_busy_detail": "请先中止当前分析。",
    }

    assert EventCullApp.accept_drop(app, SimpleNamespace(data="ignored")) == REFUSE_DROP
    assert shown == [("正在分析", "请先中止当前分析。")]


def test_drop_target_explicitly_accepts_finder_files_when_idle():
    app = object.__new__(EventCullApp)
    app.analysis_running = False
    app.calibration_running = False
    app.results = []
    app.t = {"empty_preview": "请在任意位置拖入文件或文件夹"}

    class Preview:
        def configure(self, **kwargs):
            self.kwargs = kwargs

    app.preview_label = Preview()

    assert EventCullApp._accept_drop_target(app) == COPY
    assert app.preview_label.kwargs["text"] == "请在任意位置拖入文件或文件夹"


def test_lightroom_sync_help_can_be_expanded_and_collapsed():
    class Steps:
        def __init__(self):
            self.visible = False
            self.packed = []

        def winfo_manager(self):
            return "pack" if self.visible else ""

        def pack(self, **kwargs):
            self.visible = True
            self.packed.append(kwargs)

        def pack_forget(self):
            self.visible = False

    class Toggle:
        def __init__(self):
            self.text = ""

        def configure(self, **kwargs):
            self.text = kwargs["text"]

    app = object.__new__(EventCullApp)
    app.t = {"sync_show": "查看 Lightroom 同步说明", "sync_hide": "收起 Lightroom 同步说明"}
    app.sync_steps_frame = Steps()
    app.sync_toggle_button = Toggle()

    EventCullApp.toggle_lightroom_sync_instructions(app)
    assert app.sync_steps_frame.visible is True
    assert app.sync_steps_frame.packed == [{"fill": "x", "pady": (10, 0)}]
    assert app.sync_toggle_button.text == "收起 Lightroom 同步说明"

    EventCullApp.toggle_lightroom_sync_instructions(app)
    assert app.sync_steps_frame.visible is False
    assert app.sync_toggle_button.text == "查看 Lightroom 同步说明"


def test_main_workspace_uses_native_horizontal_resizable_panes():
    source = Path(__file__).parents[1] / "event_cull" / "app.py"
    contents = source.read_text(encoding="utf-8")

    assert "tk.PanedWindow(" in contents
    assert "orient=tk.HORIZONTAL" in contents
    assert 'content.add(center, minsize=260' in contents
    assert 'content.add(left, minsize=500' in contents
    assert 'content.add(right_shell, minsize=280' in contents


def test_main_authorization_never_opens_a_nested_team_dialog_on_macos():
    source = Path(__file__).parents[1] / "event_cull" / "app.py"
    contents = source.read_text(encoding="utf-8")

    assert "from event_cull.license_ui import LicenseDialog" in contents
    assert "TeamAuthorizationDialog" not in contents


def test_results_table_keeps_identifier_columns_compact():
    source = Path(__file__).parents[1] / "event_cull" / "app.py"
    contents = source.read_text(encoding="utf-8")

    assert '("group", "组", 48, 42, False)' in contents
    assert '("star", "评级", 62, 56, False)' in contents
    assert '("file", "原片", 280, 180, True)' in contents
    assert '("reason", "分析原因", 440, 240, True)' in contents


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

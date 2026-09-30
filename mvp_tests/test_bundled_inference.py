"""Offline assets, private process ownership, cancellation and shared pipeline."""
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from event_cull import bundled_inference as runtime, engine, wedding_content as content


@pytest.fixture
def assets(tmp_path, monkeypatch):
    root = tmp_path / "content"
    (root / "models").mkdir(parents=True)
    executable = root / "runtime" / "darwin-arm64" / "llama-server"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"fixture")
    weights = {}
    for name in ("model.gguf", "mmproj-model.gguf"):
        data = name.encode()
        (root / "models" / name).write_bytes(data)
        weights[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = dict(version=1, model_id="fixture-model", weights=weights,
                    executables={"darwin-arm64": "runtime/darwin-arm64/llama-server"})
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(runtime, "platform_key", lambda: "darwin-arm64")
    return runtime.BundledAssets.load(root)


def rewrite_manifest(assets, value):
    (assets.root / "manifest.json").write_text(json.dumps(value), encoding="utf-8")


def test_relative_installation_root_becomes_absolute_before_subprocess_launch(assets, monkeypatch):
    monkeypatch.chdir(assets.root.parent)
    loaded = runtime.BundledAssets.load(Path("content"))
    assert loaded.root == assets.root.resolve() and loaded.root.is_absolute()


def test_installed_asset_check_never_starts_runtime_or_contacts_services(assets, monkeypatch):
    monkeypatch.setattr(runtime.BundledAssets, "load", lambda: assets)
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("UI must not load model"))
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *_a, **_kw: pytest.fail("network forbidden"))
    assert runtime.bundled_status()[0]


@pytest.mark.parametrize("kind", ["array", "missing-projector", "digest", "traversal", "runtime-traversal", "size"])
def test_invalid_manifest_or_paths_fail_closed(assets, kind):
    value = json.loads(json.dumps(assets.manifest))
    if kind == "array":
        value = []
    elif kind == "missing-projector":
        value["weights"]["other.gguf"] = value["weights"].pop("mmproj-model.gguf")
    elif kind == "digest":
        value["weights"]["model.gguf"]["sha256"] = "bad"
    elif kind == "traversal":
        value["weights"]["../model.gguf"] = value["weights"].pop("model.gguf")
    elif kind == "runtime-traversal":
        value["executables"]["darwin-arm64"] = "../outside"
    else:
        value["weights"]["model.gguf"]["size"] += 1
    rewrite_manifest(assets, value)
    with pytest.raises(ValueError):
        runtime.BundledAssets.load(assets.root)


def test_weight_symlink_cannot_escape_installation(assets, tmp_path):
    path = assets.root / "models" / "model.gguf"
    data = path.read_bytes()
    path.unlink()
    external = tmp_path / "outside.gguf"
    external.write_bytes(data)
    path.symlink_to(external)
    with pytest.raises(ValueError, match="路径"):
        runtime.BundledAssets.load(assets.root)


def test_sha_check_reuses_verified_files_but_detects_modified_weights(assets):
    calls = []
    assets.verify(lambda: calls.append(1))
    first = len(calls)
    assets.verify(lambda: calls.append(1))
    assert len(calls) - first == 2  # Checkpoints, no repeated file reads.
    path = assets.root / "models" / "model.gguf"
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ValueError, match="校验失败"):
        assets.verify(lambda: None)


class Process:
    def __init__(self):
        self.done = False
        self.terminated = 0

    def poll(self):
        return 0 if self.done else None

    def terminate(self):
        self.terminated += 1
        self.done = True

    def wait(self, **_kwargs):
        return 0


def test_process_is_loaded_once_with_offline_authenticated_loopback_only(assets, monkeypatch):
    spawned = []
    process = Process()

    def spawn(command, **kwargs):
        spawned.append((command, kwargs))
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", spawn)
    client = runtime.BundledInference(assets)
    monkeypatch.setattr(client, "_request", lambda *_args, **_kwargs: {"status": "ok"})
    client.start()
    client.start()
    assert len(spawned) == 1
    command, options = spawned[0]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert "--offline" in command and "--no-webui" in command
    assert command[command.index("--image-max-tokens")+1] == "1024"
    assert options["env"]["LLAMA_API_KEY"] == client._token
    assert client._token not in command and not options.get("shell")
    client.close()
    client.close()
    assert process.terminated == 1


@pytest.mark.parametrize("value", [0, 255, 4097, True, "4096", 1024.0])
def test_invalid_image_token_limit_never_starts_a_process(assets, monkeypatch, value):
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_a, **_k:pytest.fail("invalid config"))
    with pytest.raises(ValueError, match="token"):
        runtime.BundledInference(assets, image_token_limit=value)


def test_detail_diagnostic_extends_same_runtime_not_a_second_model(assets, monkeypatch):
    commands = []
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda command, **_k: commands.append(command) or Process())
    client = runtime.BundledInference(assets, image_token_limit=4096)
    monkeypatch.setattr(client, "_request", lambda *_a, **_k:{"status":"ok"})
    client.start()
    client.start()
    assert len(commands) == 1 and commands[0][commands[0].index("--image-max-tokens")+1] == "4096"
    client.close()


def test_no_download_or_external_service_discovery_in_bundled_adapter(assets, monkeypatch):
    monkeypatch.setattr(content.BundledAssets, "load", lambda: assets)
    monkeypatch.setattr(content, "_json_request", lambda *_a, **_kw: pytest.fail("Ollama forbidden"))
    chats = []

    def chat(encoded, prompt, schema):
        chats.append((encoded, prompt))
        if schema is content.ROUTE_SCHEMA:
            return {"content_type": "formal_group", "section": 5}
        return dict(checks=dict.fromkeys(content.CHECK_NAMES, "pass"), observations=[], rule_ids=["W048"])

    model = content.BundledWeddingContentModel()
    monkeypatch.setattr(model.runtime, "chat", chat)
    result = model.inspect(np.zeros((80, 100, 3), dtype=np.uint8))
    assert result["status"] == "ok" and len(chats) == 2
    assert chats[0][0] == chats[1][0]
    assert "W041" in chats[1][1] and "W240" in chats[1][1]
    model.close()


@pytest.mark.parametrize("cancel", [True, False])
def test_cancel_or_timeout_closes_only_owned_process_without_waiting_for_http(assets, monkeypatch, cancel):
    release = threading.Event()
    stop = threading.Event()
    process = Process()
    client = runtime.BundledInference(assets, stop)
    client.port = 23456
    client.process = process

    def open_request(request, **_kwargs):
        assert request.full_url == "http://127.0.0.1:23456/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer " + client._token
        release.wait(1)
        raise TimeoutError("fixture")

    monkeypatch.setattr(runtime.urllib.request, "build_opener", lambda *_a: SimpleNamespace(open=open_request))
    if cancel:
        stop.set()
    try:
        with pytest.raises(runtime.InferenceCancelled if cancel else TimeoutError):
            client._request("/v1/chat/completions", {}, timeout=.01)
        assert process.terminated == 1
        if not cancel:
            with pytest.raises(RuntimeError):
                client.checkpoint()  # Timeout must not masquerade as user cancellation.
    finally:
        release.set()


def test_redirects_to_external_addresses_are_rejected():
    with pytest.raises(ValueError):
        runtime._NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com")


@pytest.mark.parametrize("progress_error", [False, True])
def test_enabled_analysis_shares_one_bundled_model_and_always_closes_it(tmp_path, monkeypatch, progress_error):
    paths = [tmp_path / "one.jpg", tmp_path / "two.jpg"]
    for path in paths:
        path.write_bytes(b"fixture")
    models, used, closed = [], [], []

    def new_model(*_args):
        model = SimpleNamespace(close=lambda: closed.append(1))
        models.append(model)
        return model

    def analyze(path, pairs, options, cache, model):
        used.append(model)
        return engine.PhotoResult(path, score=90, sharpness=150, subject_sharpness=150), str(path), None

    monkeypatch.setattr(engine, "BundledWeddingContentModel", new_model)
    monkeypatch.setattr(engine, "LocalWeddingContentModel", lambda *_a: pytest.fail("legacy discovery forbidden"))
    monkeypatch.setattr(engine, "_load_analysis_cache", lambda: {})
    monkeypatch.setattr(engine, "_write_analysis_cache", lambda _cache: None)
    monkeypatch.setattr(engine, "_analyze_cached", analyze)

    def progress(*_args):
        if progress_error:
            raise RuntimeError("fixture callback")

    options = engine.AnalysisOptions(scene_type="婚礼", content_model_enabled=True)
    if progress_error:
        with pytest.raises(RuntimeError, match="callback"):
            engine.analyze_paths(paths, options, progress)
    else:
        results = engine.analyze_paths(paths, options, progress)
        assert all(item.suggested_rating != 3 for item in results)
    assert len(models) == 1 and used and all(model is models[0] for model in used)
    assert closed == [1]


def test_packager_excludes_other_platform_and_unneeded_runtime_tools(assets):
    irrelevant = assets.executable.parent / "llama-cli"
    irrelevant.write_bytes(b"fixture")
    from unittest.mock import patch
    with patch.object(runtime.BundledAssets, "load", return_value=assets):
        datas, binaries = runtime.build_asset_tuples()
    assert (str(assets.executable), str(runtime.RESOURCE_ROOT / "runtime/darwin-arm64")) in binaries
    assert all(source != str(irrelevant) for source, _ in datas + binaries)
    assert sum(source.endswith(".gguf") for source, _ in datas) == 2


def test_short_burst_review_reuses_the_same_private_runtime_with_a_bounded_reply(assets,monkeypatch):
    client=runtime.BundledInference(assets)
    requests=[]
    monkeypatch.setattr(client,"start",lambda:None)
    monkeypatch.setattr(client,"checkpoint",lambda:None)
    def request(path,payload):
        requests.append(payload)
        return {"choices":[{"message":{"content":'{"best":1}'}}]}
    monkeypatch.setattr(client,"_request",request)
    assert client.chat("fixture","compare",{},max_tokens=80)=={"best":1}
    assert requests[0]["max_tokens"]==80
    assert requests[0]["messages"][1]["content"][1]["image_url"]["url"]=="data:image/jpeg;base64,fixture"
    client.close()


@pytest.mark.parametrize("budget",[0,-1,801,True,1.5,"80"])
def test_invalid_reply_budget_is_rejected_before_native_start(assets,monkeypatch,budget):
    client=runtime.BundledInference(assets)
    monkeypatch.setattr(client,"start",lambda:pytest.fail("invalid budget must not start a model"))
    with pytest.raises(ValueError,match="budget|预算"):
        client.chat("fixture","compare",{},max_tokens=budget)
    client.close()

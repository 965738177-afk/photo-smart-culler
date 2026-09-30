"""Semantic crop ownership, bounded work, unknown evidence and safe provenance."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch

from src.core.image_features import semantic_regions as semantic
from src.core.image_features.human_analysis import PIPELINE_ID as PERSON_PIPELINE


def face(x=.4, y=.2, width=.1, height=.1):
    return {"bbox": [x, y, x+width, y+height]}


def test_regions_reuse_caller_preview_and_bound_face_crops():
    image = Image.new("RGB", (1000, 1000))
    regions, kinds, masks = semantic.build_regions(image, [face(x=.1+i*.05) for i in range(12)])
    assert regions[0] is image
    assert len(regions) <= 6 and kinds.count("face") == 4
    assert masks[:2] == [12, 4]
    for crop in regions[1:]:
        crop.close()
    assert image.getpixel((0, 0)) == (0, 0, 0)
    image.close()


def test_tiny_faces_are_unknown_not_negative_semantic_values():
    image = Image.new("RGB", (100, 100))
    regions, kinds, masks = semantic.build_regions(image, [face(width=.1, height=.1)])
    assert regions == [image] and kinds == ["whole"] and masks == [0, 0, False, False]
    values = np.ones((1, semantic.EMBEDDING_DIMENSION), np.float32)
    result = semantic.aggregate_regions(values, kinds, masks)
    assert result.shape == (semantic.FEATURE_DIMENSION,)
    assert np.all(result[:semantic.EMBEDDING_DIMENSION*2] == 1)
    assert np.all(result[semantic.EMBEDDING_DIMENSION*2:] == 0)
    image.close()


def test_bad_frozen_geometry_is_rejected():
    image = Image.new("RGB", (100, 100))
    with pytest.raises(ValueError, match="geometry"):
        semantic.build_regions(image, [{"bbox": [0, 0, float("nan"), 1]}])
    image.close()


def test_region_aggregation_keeps_largest_and_worst_face_information():
    values = np.vstack([np.ones(768)*i for i in (1, 2, 3, 5)]).astype(np.float32)
    features = semantic.aggregate_regions(values, ["whole", "context", "face", "face"], [2, 2, 1, 1])
    assert [features[i*768] for i in range(6)] == [1, 2, 4, 3, 5, 3]


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.ones(1))
        self.calls = []
    def forward(self, pixels):
        self.calls.append(len(pixels))
        # A test of runtime aggregation, not proof of semantic model accuracy.
        return SimpleNamespace(pooler_output=torch.ones((len(pixels), 768)))


def make_service(*, checkpoint=lambda: None):
    creations = []
    model = FakeModel()
    def processor(*, images, return_tensors):
        return {"pixels": torch.ones((len(images), 3, 4, 4))}
    def backend(snapshot, device):
        creations.append((snapshot, device))
        return processor, model
    service = semantic.SemanticRegionEncoder(resolver=lambda: "injected-local",
        backend_factory=backend, device_resolver=lambda: "cpu", checkpoint=checkpoint)
    return service, model, creations


def test_one_semantic_owner_under_concurrent_callers_without_own_decoder():
    service, model, creations = make_service()
    image = Image.new("RGB", (200, 200))
    evidence = {"status": "ok", "pipeline": PERSON_PIPELINE, "faces": [face(width=.25, height=.25)]}
    with ThreadPoolExecutor(max_workers=4) as workers:
        outputs = list(workers.map(lambda _: service.encode([image], [evidence]), range(10)))
    assert creations == [("injected-local", "cpu")]
    assert all(value.shape == (1, semantic.FEATURE_DIMENSION) for value in outputs)
    assert all(n <= semantic.BATCH_SIZE for n in model.calls)
    service.close()
    assert service._model is None and service._processor is None
    assert image.getpixel((0, 0)) == (0, 0, 0)
    image.close()


def test_region_batches_and_preview_batch_are_bounded():
    service, model, _ = make_service()
    images = [Image.new("RGB", (200, 200)) for _ in range(16)]
    evidence = {"status": "ok", "pipeline": PERSON_PIPELINE,
                "faces": [face(x=i*.15, width=.2, height=.2) for i in range(4)]}
    result = service.encode(images, [evidence]*16)
    assert result.shape == (16, semantic.FEATURE_DIMENSION)
    assert len(model.calls) >= 2 and max(model.calls) <= semantic.BATCH_SIZE
    with pytest.raises(ValueError, match="bounded"):
        service.encode(images+[images[0]], [evidence]*17)
    service.close()
    for image in images:
        image.close()


def test_cancelled_semantic_start_never_initializes_model():
    def cancelled():
        raise InterruptedError("cancelled")
    service, _, creations = make_service(checkpoint=cancelled)
    image = Image.new("RGB", (100, 100))
    with pytest.raises(InterruptedError):
        service.encode([image], [{"status": "ok", "faces": []}])
    assert not creations
    image.close()


def test_missing_person_evidence_is_not_silently_called_valid_semantics():
    service, _, creations = make_service()
    image = Image.new("RGB", (100, 100))
    with pytest.raises(ValueError, match="incomplete"):
        service.encode([image], [{"status": "error", "faces": []}])
    with pytest.raises(ValueError, match="incomplete"):
        service.encode([image], [{"status": "ok", "pipeline": "incompatible", "faces": []}])
    assert creations == []
    service.close()
    image.close()


def test_new_semantic_model_is_pinned_safe_and_not_a_required_desktop_download(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"src"))
    from core.model_provisioning import SEMANTIC_RETENTION_MODEL, SEMANTIC_RETENTION_TEXT_MODEL, MODEL_REGISTRY
    spec = SEMANTIC_RETENTION_MODEL
    assert spec.repo_id == "google/siglip2-base-patch16-224"
    assert len(spec.revision) == 40
    assert "model.safetensors" in spec.allow_patterns
    assert not any(p.endswith(".py") or p.endswith(".bin") for p in spec.allow_patterns)
    assert spec.key not in MODEL_REGISTRY
    text = SEMANTIC_RETENTION_TEXT_MODEL
    assert text.key not in MODEL_REGISTRY and text.revision == spec.revision and text.repo_id == spec.repo_id
    assert "tokenizer.json" in text.allow_patterns


def test_description_owner_loads_text_once_without_decoding_or_image_initialization():
    loads, calls = [], []
    class TextModel(torch.nn.Module):
        def forward(self, input_ids):
            assert input_ids.dtype == torch.int64 and input_ids.shape[1] == 64
            calls.append(len(input_ids))
            return SimpleNamespace(pooler_output=torch.ones((len(input_ids), 768)))
    def backend(snapshot, device):
        loads.append((snapshot, device))
        def tokenizer(texts, **kwargs):
            assert kwargs == dict(padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            return {"input_ids": torch.ones((len(texts), 64), dtype=torch.int64)}
        return tokenizer, TextModel()
    def fail(*args):
        pytest.fail("a text-only projection must not initialize the image tower")
    owner = semantic.SemanticRegionEncoder(text_resolver=lambda: "local", text_backend_factory=backend,
                                           backend_factory=fail, device_resolver=lambda: "cpu")
    values = owner.encode_texts(["A fixed description"]*42)
    assert values.shape == (42, 768) and np.allclose(np.linalg.norm(values, axis=1), 1)
    owner.encode_texts(["A second description"])
    assert loads == [("local", "cpu")] and calls == [16, 16, 10, 1]
    owner.close()
    assert owner._text_model is None and owner._tokenizer is None and owner._model is None


@pytest.mark.parametrize("texts", [[], [""], [1], ["text"]*129])
def test_invalid_text_batch_does_not_resolve_or_load_models(texts):
    owner = semantic.SemanticRegionEncoder(text_resolver=lambda: pytest.fail("should fail before resolving"))
    with pytest.raises(ValueError, match="bounded"):
        owner.encode_texts(texts)


def test_content_projection_uses_actual_region_vectors_not_coordinatewise_face_extrema():
    from src.core.image_features import semantic_concepts as concepts
    regions = np.zeros((2, semantic.FEATURE_DIMENSION), dtype=np.float32)
    regions[0, :768] = 1
    regions[0, 768:1536] = 2
    regions[0, 1536:2304] = 3
    regions[:, 2304:3840] = 9999  # Not actual worst/best people's embeddings.
    regions[0, 3840:4608] = 4
    regions[0, -4:] = [2, 2, 1, 1]
    text = np.zeros((len(concepts.CONCEPTS), 768), dtype=np.float32)
    text[:, 0] = 1
    values = concepts.project_concepts(regions, text)
    n = len(concepts.CONCEPTS)
    assert values.shape == (2, concepts.FEATURE_DIMENSION)
    assert [values[0, i*n] for i in range(4)] == [1, 2, 3, 4]
    assert np.all(values[1] == 0) and list(values[0, -4:]) == [2, 2, 1, 1]


def test_content_projection_rejects_missing_or_nonfinite_direction():
    from src.core.image_features import semantic_concepts as concepts
    values = np.zeros((2, semantic.FEATURE_DIMENSION), dtype=np.float32)
    texts = np.ones((len(concepts.CONCEPTS), 768), dtype=np.float32)
    texts[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        concepts.project_concepts(values, texts)
    texts[:] = 0
    with pytest.raises(ValueError, match="Empty"):
        concepts.project_concepts(values, texts)


def test_decode_failure_closes_successful_and_inflight_previews_once():
    from scripts.extract_retention_embeddings import owned_preview_batch
    import threading
    done = threading.Event()
    started = threading.Event()
    class Preview:
        def __init__(self):
            self.closes = 0
        def close(self):
            self.closes += 1
    previews = [Preview(), Preview()]
    def loader(index):
        if index == 1:
            assert started.wait(2)
            raise OSError("bad input")
        if index == 2:
            started.set()
            assert done.wait(2)
        return previews[0 if index == 0 else 1]
    with ThreadPoolExecutor(max_workers=3) as readers:
        with pytest.raises(OSError, match="bad input"):
            with owned_preview_batch(readers, loader, [0, 1, 2]):
                pytest.fail("bad input must not be silently skipped")
        done.set()
    assert [p.closes for p in previews] == [1, 1]


def test_successful_decode_batch_reuses_previews_and_closes_on_inference_failure():
    from scripts.extract_retention_embeddings import owned_preview_batch
    previews = [Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10))]
    calls = []
    with ThreadPoolExecutor(max_workers=2) as readers:
        with pytest.raises(InterruptedError):
            with owned_preview_batch(readers, lambda i: calls.append(i) or previews[i], [0, 1]) as batch:
                assert batch == previews
                raise InterruptedError("cancelled")
    assert sorted(calls) == [0, 1]
    for preview in previews:
        with pytest.raises(ValueError):
            preview.getpixel((0, 0))

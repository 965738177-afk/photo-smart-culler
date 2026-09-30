"""Bounded semantic whole-frame/subject-context/face evidence, not a decoder.

The caller supplies shared RGB previews and previously observed face boxes.
There is no new detector, identity recognition, file scan, label input, remote
code execution, upload, profile activation or claim of completed actions.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np

PIPELINE_ID = "siglip2-base-whole-context-faces-v1"
EMBEDDING_DIMENSION = 768
# Whole, context, face mean/min/max/largest, then explicit observation masks.
FEATURE_DIMENSION = EMBEDDING_DIMENSION * 6 + 4
MAX_FACES = 4
BATCH_SIZE = 32
MAX_IMAGE_BATCH = 16
TOKEN_PIPELINE_ID = "siglip2-base-regions-before-last-two-blocks-v1"
TOKEN_COUNT = 196
TAIL_LAYERS = 2


@dataclass(frozen=True, slots=True)
class SpatialTokenSpec:
    """Single contract for extraction, cache reading and task training."""
    pipeline: str
    token_count: int
    dimension: int
    trainable_layers: int


SIGLIP_SPATIAL_SPEC = SpatialTokenSpec(TOKEN_PIPELINE_ID, TOKEN_COUNT, EMBEDDING_DIMENSION, TAIL_LAYERS)
TINY_SPATIAL_SPEC = SpatialTokenSpec("tinyclip-8m-uncropped-patch-before-all-ten-blocks-v2", 197, 256, 10)


def validate_batch(images, observations):
    from .human_analysis import PIPELINE_ID as person_pipeline
    if len(images) != len(observations):
        raise ValueError("Semantic previews/evidence length mismatch")
    if len(images) > MAX_IMAGE_BATCH:
        raise ValueError("Semantic preview batch exceeds bounded image limit")
    if any(not isinstance(e, dict) or e.get("status") != "ok" or e.get("pipeline") != person_pipeline
           for e in observations):
        raise ValueError("Frozen person observations incomplete")


@contextmanager
def owned_regions(images, observations, checkpoint=lambda: None):
    """One crop owner for frozen embeddings and trainable-prefix extraction."""
    regions, metadata = [], []
    try:
        for image, evidence in zip(images, observations):
            checkpoint()
            crops, kinds, masks = build_regions(image, evidence.get("faces", ()))
            metadata.append((len(regions), len(crops), kinds, masks))
            regions.extend(crops)
        yield regions, metadata
    finally:
        for start, length, _kinds, _masks in metadata:
            for region in regions[start+1:start+length]:
                region.close()


def build_regions(image, faces):
    """At most six regions, using normalized face boxes without another scan.

    Context is merely an expanded known-face region, not a verified body box.
    No observed face is unknown. Side faces can supply semantic content without
    becoming confident eye/gaze measurements.
    """
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("Empty semantic preview")
    boxes = []
    for face in faces:
        bounds = np.asarray(face.get("bbox", ()), dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError("Invalid frozen face geometry")
        x0, y0, x1, y1 = np.clip(bounds, 0, 1) * (width, height, width, height)
        if x1-x0 < 24 or y1-y0 < 24:
            continue  # Tiny faces remain unknown, not negative examples.
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append((x0, y0, x1, y1))
    boxes.sort(key=lambda box: (box[2]-box[0])*(box[3]-box[1]), reverse=True)
    regions, kinds = [image], ["whole"]
    context_known = False
    if boxes:
        left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
        right, bottom = max(b[2] for b in boxes), max(b[3] for b in boxes)
        largest_w, largest_h = boxes[0][2]-boxes[0][0], boxes[0][3]-boxes[0][1]
        context = (max(0, int(left-largest_w*.8)), max(0, int(top-largest_h*.4)),
                   min(width, int(right+largest_w*.8)), min(height, int(bottom+largest_h*4)))
        # A context equal to virtually the whole image would duplicate work.
        if (context[2]-context[0])*(context[3]-context[1]) < width*height*.95:
            regions.append(image.crop(context))
            kinds.append("context")
            context_known = True
        for box in boxes[:MAX_FACES]:
            x0, y0, x1, y1 = box
            pad_x, pad_y = (x1-x0)*.2, (y1-y0)*.2
            regions.append(image.crop((max(0, int(x0-pad_x)), max(0, int(y0-pad_y)),
                                       min(width, int(x1+pad_x)), min(height, int(y1+pad_y)))))
            kinds.append("face")
    return regions, kinds, [len(boxes), min(len(boxes), MAX_FACES), context_known, bool(boxes)]


def aggregate_regions(values, kinds, masks):
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(kinds), EMBEDDING_DIMENSION) or not np.isfinite(values).all():
        raise ValueError("Invalid semantic region features")
    whole = values[0]
    context = values[kinds.index("context")] if "context" in kinds else whole
    faces = values[[i for i, kind in enumerate(kinds) if kind == "face"]]
    summary = (np.concatenate((faces.mean(0), faces.min(0), faces.max(0), faces[0]))
               if len(faces) else np.zeros(EMBEDDING_DIMENSION*4, dtype=np.float32))
    result = np.concatenate((whole, context, summary, masks)).astype(np.float32)
    if result.shape != (FEATURE_DIMENSION,) or not np.isfinite(result).all():
        raise ValueError("Invalid semantic aggregation")
    return result


def _load_backend(snapshot, device):
    import torch
    from transformers import SiglipImageProcessor, SiglipVisionModel
    processor = SiglipImageProcessor.from_pretrained(snapshot, local_files_only=True)
    dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    model, info = SiglipVisionModel.from_pretrained(
        snapshot, local_files_only=True, dtype=dtype, output_loading_info=True)
    # Never let a prefix/load mismatch leave random weights and a plausible
    # feature shape. Unused text tower keys are expected in image-only loading.
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("Semantic image weights did not load completely")
    model.to(device).eval()
    return processor, model


def _load_text_backend(snapshot, device):
    import torch
    from transformers import AutoTokenizer, SiglipTextModel
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True,
                                              trust_remote_code=False)
    dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    model, info = SiglipTextModel.from_pretrained(
        snapshot, local_files_only=True, dtype=dtype, output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("Semantic text weights did not load completely")
    model.to(device).eval()
    return tokenizer, model


def _load_tiny_backend(snapshot, device):
    """Use the installed Transformers architecture, no remote code/text tower."""
    import torch
    from transformers import CLIPImageProcessor, CLIPVisionModel
    processor = CLIPImageProcessor.from_pretrained(snapshot, local_files_only=True)
    # Float32 keeps the patch boundary identical to trainable float32 blocks;
    # no Core ML fallback/provider switch is hidden in this experimental ID.
    model, info = CLIPVisionModel.from_pretrained(
        snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float32,
        output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("Tiny image weights did not load completely")
    if (model.config.hidden_size != 256 or len(model.encoder.layers) != 10
            or model.config.patch_size != 16 or model.config.image_size != 224):
        raise ValueError("Unexpected tiny vision architecture")
    model.to(device).eval()
    return _uncropped_processor(processor), model


def _uncropped_processor(processor):
    """Preserve whole-frame actions instead of CLIP's default center crop.

    Provider-specific preprocessing, not another decoder. Prepared squares are
    owned here; caller images and the shared region-crop owner remain untouched.
    """
    def process(*, images, return_tensors):
        from PIL import Image, ImageOps
        squares = []
        try:
            for image in images:
                squares.append(ImageOps.pad(image, (224,224), method=Image.Resampling.BICUBIC,
                                             color=(0,0,0), centering=(.5,.5)))
            return processor(images=squares, return_tensors=return_tensors,
                             do_resize=False, do_center_crop=False)
        finally:
            for square in squares:
                square.close()
    return process


def _class_token_pooler():
    import torch
    class ClassTokenPooler(torch.nn.Module):
        def forward(self, values):
            return values[:, 0, :]
    return ClassTokenPooler()


def _pooled_tail(model, value, spec):
    if spec == TINY_SPATIAL_SPEC:
        return model.post_layernorm(value[:, 0, :])
    return model.head(model.post_layernorm(value))


class SemanticRegionEncoder:
    """One caller-owned local image tower for all regions and workers."""
    def __init__(self, *, checkpoint=lambda: None, resolver=None, backend_factory=_load_backend,
                 device_resolver=None, text_resolver=None, text_backend_factory=_load_text_backend,
                 tiny_vision=False):
        if type(tiny_vision) is not bool:
            raise ValueError("Explicit boolean tiny vision configuration required")
        self.spatial_spec = TINY_SPATIAL_SPEC if tiny_vision else SIGLIP_SPATIAL_SPEC
        self._checkpoint, self._resolver = checkpoint, resolver
        self._backend_factory = _load_tiny_backend if tiny_vision and backend_factory is _load_backend else backend_factory
        self._device_resolver = device_resolver
        self._processor = self._model = None
        self._text_resolver, self._text_backend_factory = text_resolver, text_backend_factory
        self._tokenizer = self._text_model = None
        self._lock = threading.RLock()
        self.device = "cpu"
        self.regions_encoded = 0
        self.last_prefix_check = None

    def load(self):
        with self._lock:
            self._checkpoint()
            if self._model is not None:
                return
            if self._resolver is None:
                from ..model_provisioning import SEMANTIC_RETENTION_MODEL, TINY_RETENTION_MODEL, resolve_snapshot
                spec = TINY_RETENTION_MODEL if self.spatial_spec == TINY_SPATIAL_SPEC else SEMANTIC_RETENTION_MODEL
                snapshot = resolve_snapshot(spec, allow_download=False)
            else:
                snapshot = self._resolver()
            if self._device_resolver is None:
                from ..app_settings import get_preferred_torch_device
                device = get_preferred_torch_device()
            else:
                device = self._device_resolver()
            processor, model = self._backend_factory(snapshot, device)
            self._checkpoint()
            self._processor, self._model, self.device = processor, model, device

    def encode(self, images, observations):
        if self.spatial_spec != SIGLIP_SPATIAL_SPEC:
            raise ValueError("Tiny task training uses its spatial contract, not SigLIP descriptors")
        images, observations = list(images), list(observations)
        self._checkpoint()
        validate_batch(images, observations)
        if not images:
            return np.empty((0, FEATURE_DIMENSION), dtype=np.float32)
        with self._lock:
            self._checkpoint()
            self.load()
            import torch
            with owned_regions(images, observations, self._checkpoint) as (regions, metadata):
                encoded = []
                for start in range(0, len(regions), BATCH_SIZE):
                    self._checkpoint()
                    batch = self._processor(images=regions[start:start+BATCH_SIZE], return_tensors="pt")
                    dtype = next(self._model.parameters()).dtype
                    inputs = {key: value.to(device=self.device, dtype=dtype) for key, value in batch.items()}
                    with torch.inference_mode():
                        output = self._model(**inputs).pooler_output
                    values = output.float().detach().cpu().numpy()
                    norm = np.linalg.norm(values, axis=1, keepdims=True)
                    if (values.ndim != 2 or values.shape[1] != EMBEDDING_DIMENSION
                            or not np.isfinite(values).all() or np.any(norm < 1e-8)):
                        raise ValueError("Invalid semantic image output")
                    encoded.append(values/norm)
                    self.regions_encoded += len(values)
                self._checkpoint()
                all_values = np.vstack(encoded)
                return np.vstack([aggregate_regions(all_values[start:start+length], kinds, masks)
                                  for start, length, kinds, masks in metadata])

    def encode_prefix(self, images, observations, *, verify_boundary=False):
        """Cache label-free spatial tokens before the configured trainable blocks.

        No human labels or trainable updates here. An original preview is shared
        across its regions. The same owner/weights/processor as encode() is used;
        there is no second detector/decoder or per-epoch backbone initialization.
        """
        images, observations = list(images), list(observations)
        self._checkpoint()
        validate_batch(images, observations)
        if not images:
            return [], []
        with self._lock:
            self._checkpoint()
            self.load()
            model = self._model
            spec = self.spatial_spec
            minimum_layers = spec.trainable_layers + (spec != TINY_SPATIAL_SPEC)
            if (not hasattr(model, "encoder") or len(model.encoder.layers) < minimum_layers
                    or not hasattr(model, "embeddings")):
                raise ValueError("Unsupported pinned semantic prefix architecture")
            import torch
            values = []
            with owned_regions(images, observations, self._checkpoint) as (regions, metadata):
                for start in range(0, len(regions), BATCH_SIZE):
                    self._checkpoint()
                    batch = self._processor(images=regions[start:start+BATCH_SIZE], return_tensors="pt")
                    dtype = next(model.parameters()).dtype
                    if set(batch) != {"pixel_values"}:
                        raise ValueError("Unsupported semantic prefix processor")
                    pixels = batch["pixel_values"].to(device=self.device, dtype=dtype)
                    with torch.inference_mode():
                        hidden = model.embeddings(pixels)
                        if spec == TINY_SPATIAL_SPEC:
                            hidden = model.pre_layrnorm(hidden)
                        for layer in model.encoder.layers[:-spec.trainable_layers]:
                            self._checkpoint()
                            hidden = layer(hidden, None)
                    tokens = hidden.detach().cpu().to(torch.float16).numpy()
                    if (tokens.shape != (len(pixels), spec.token_count, spec.dimension)
                            or not np.isfinite(tokens).all()):
                        raise ValueError("Incomplete/nonfinite pretrained prefix tokens")
                    if verify_boundary:
                        # Same preprocessed pixels, same original model, one
                        # decoder. Check half-precision disk tokens do not alter
                        # the untrained representation at the split boundary.
                        with torch.inference_mode():
                            direct = model(pixel_values=pixels).pooler_output.float().cpu().numpy()
                        direct = direct / np.maximum(np.linalg.norm(direct, axis=1, keepdims=True), 1e-8)
                        restored = self.finish_prefix(tokens)
                        error = np.abs(restored - direct)
                        passed = bool(np.isfinite(direct).all() and error.max() < .003
                                      and (restored * direct).sum(1).min() > .999)
                        self.last_prefix_check = {"regions": len(tokens), "max_absolute_error": float(error.max()),
                                                  "passed_numerical_boundary": passed}
                        if not passed:
                            raise ValueError("Pretrained spatial boundary is not numerically equivalent")
                    values.append(tokens)
                    self.regions_encoded += len(tokens)
                self._checkpoint()
                packed = np.concatenate(values)
                return [packed[start:start+length].copy() for start,length,_,_ in metadata], [
                    {"kinds":kinds, "masks":masks} for _,_,kinds,masks in metadata]

    def copy_trainable_tail(self):
        """Clone only the owned pretrained tail; never mutate the base snapshot."""
        with self._lock:
            self._checkpoint()
            self.load()
            import copy
            import torch
            model = self._model
            spec = self.spatial_spec
            minimum_layers = spec.trainable_layers + (spec != TINY_SPATIAL_SPEC)
            if (not hasattr(model, "encoder") or len(model.encoder.layers) < minimum_layers
                    or not hasattr(model, "post_layernorm")
                    or (spec != TINY_SPATIAL_SPEC and not hasattr(model, "head"))):
                raise ValueError("Unsupported pinned semantic tail architecture")
            return (torch.nn.ModuleList([copy.deepcopy(layer).cpu().float()
                                         for layer in model.encoder.layers[-spec.trainable_layers:]]),
                    copy.deepcopy(model.post_layernorm).cpu().float(),
                    _class_token_pooler() if spec == TINY_SPATIAL_SPEC else copy.deepcopy(model.head).cpu().float())

    def finish_prefix(self, tokens):
        """Numerical boundary check using the original, untrained tail weights."""
        values = np.asarray(tokens)
        spec = self.spatial_spec
        if (values.ndim != 3 or not 1 <= len(values) <= BATCH_SIZE
                or values.shape[1:] != (spec.token_count,spec.dimension)
                or not np.isfinite(values).all()):
            raise ValueError("Invalid bounded pretrained spatial prefix")
        with self._lock:
            self._checkpoint()
            self.load()
            import torch
            model = self._model
            dtype = next(model.parameters()).dtype
            with torch.inference_mode():
                value = torch.as_tensor(values.copy(),device=self.device,dtype=dtype)
                for layer in model.encoder.layers[-spec.trainable_layers:]:
                    self._checkpoint()
                    value = layer(value,None)
                vector = _pooled_tail(model, value, spec).float().cpu().numpy()
            norms = np.linalg.norm(vector,axis=1,keepdims=True)
            if not np.isfinite(vector).all() or np.any(norms < 1e-8):
                raise ValueError("Invalid pretrained tail reconstruction")
            return vector/norms

    def close(self):
        with self._lock:
            self._processor = self._model = None
            self._tokenizer = self._text_model = None

    def encode_texts(self, texts):
        """Encode small fixed descriptions, not a generative per-photo call.

        Uses the same managed snapshot and this owner's lifecycle. A text-only
        projection of frozen image features must not initialize the image tower
        or decode photos again. Similarity is not a calibrated detection score.
        """
        if self.spatial_spec != SIGLIP_SPATIAL_SPEC:
            raise ValueError("Tiny task training never loads a text tower")
        texts = list(texts)
        if not texts or len(texts) > 128 or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("A bounded nonempty text description batch is required")
        with self._lock:
            self._checkpoint()
            import torch
            if self._text_model is None:
                if self._text_resolver is None:
                    from ..model_provisioning import SEMANTIC_RETENTION_TEXT_MODEL, resolve_snapshot
                    snapshot = resolve_snapshot(SEMANTIC_RETENTION_TEXT_MODEL, allow_download=False)
                else:
                    snapshot = self._text_resolver()
                if self._model is not None:
                    device = self.device
                elif self._device_resolver is None:
                    from ..app_settings import get_preferred_torch_device
                    device = get_preferred_torch_device()
                else:
                    device = self._device_resolver()
                tokenizer, model = self._text_backend_factory(snapshot, device)
                self._checkpoint()
                self._tokenizer, self._text_model, self.device = tokenizer, model, device
            values = []
            for start in range(0, len(texts), 16):
                self._checkpoint()
                tokens = self._tokenizer(texts[start:start+16], padding="max_length",
                                         max_length=64, truncation=True, return_tensors="pt")
                # Token IDs stay integral; the float model dtype is NOT applied
                # to tokenizer output. Fixed length matches upstream training.
                inputs = {key: value.to(self.device) for key, value in tokens.items()}
                with torch.inference_mode():
                    output = self._text_model(**inputs).pooler_output
                encoded = output.float().detach().cpu().numpy()
                norm = np.linalg.norm(encoded, axis=1, keepdims=True)
                if (encoded.shape != (min(16, len(texts)-start), EMBEDDING_DIMENSION)
                        or not np.isfinite(encoded).all() or np.any(norm < 1e-8)):
                    raise ValueError("Invalid semantic text output")
                values.append(encoded/norm)
            self._checkpoint()
            return np.vstack(values)

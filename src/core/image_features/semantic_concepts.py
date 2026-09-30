"""Fixed descriptive axes projected from the shared SigLIP2 region encoder.

These are image/text similarities, NOT verified gaze, expression, action or
wedding-role detections. A learned delivery ranker can use them; a description
alone never rejects or promotes a photograph. No human labels define the axes.
"""
from __future__ import annotations

import numpy as np
from .semantic_regions import EMBEDDING_DIMENSION, FEATURE_DIMENSION as REGION_DIMENSION

PIPELINE_ID = "siglip2-whole-context-face-content-axes-v1"
CONCEPTS = (
    "a sharp well focused wedding photograph",
    "an out of focus wedding photograph with blurry people",
    "a wedding portrait with a deliberately blurred background",
    "a posed group portrait where the people look at the camera",
    "a group portrait where some people look away from the camera",
    "people smiling happily in a wedding photograph",
    "people laughing together naturally at a wedding",
    "a calm elegant portrait with a serious expression",
    "a person blinking with closed eyes in a posed portrait",
    "a person smiling with naturally narrowed eyes",
    "a person sticking their tongue out",
    "a person speaking with their mouth open",
    "an awkward unfinished pose or gesture",
    "a confident complete stylish pose",
    "a couple looking at each other affectionately",
    "a couple holding hands and walking together",
    "a couple embracing or kissing",
    "people interacting naturally without looking at the camera",
    "a person walking with a natural confident stride",
    "a portrait from behind showing a wedding dress",
    "a portrait with a person's face obstructed by another person",
    "a full length portrait showing the complete outfit",
    "a close up portrait of a person's face",
    "a wedding portrait showing people and the surrounding environment",
    "a bride getting her hair and makeup done",
    "a bride and groom posing together",
    "friends playing a wedding party game",
    "people celebrating with confetti at a wedding",
    "a wedding ceremony with an emotional exchange",
    "people exchanging rings at a wedding ceremony",
    "a couple entering a wedding venue",
    "people raising glasses and toasting at a wedding reception",
    "parents and children embracing at a wedding",
    "an emotional candid wedding photograph",
    "a close up of wedding rings flowers or clothing details",
    "a wide photograph of a decorated wedding venue",
    "an empty repetitive photograph without a clear subject",
    "a well composed wedding photograph with a clear subject",
    "a photograph with distracting clutter in front of the subject",
    "a candid photograph of wedding guests",
    "a photograph with bright overexposed faces",
    "a low light wedding photograph with clearly visible people",
)
DESCRIPTIONS = tuple(f"This is a photo of {concept}." for concept in CONCEPTS)
# Whole, subject context, mean observed face, largest observed face, masks.
FEATURE_DIMENSION = len(CONCEPTS) * 4 + 4


def project_concepts(region_features, description_vectors):
    """Project frozen region values; no photo decode or per-photo text call.

    Coordinate-wise min/max pooled vectors are NOT embeddings of actual faces,
    so they deliberately do not masquerade as worst/best person's similarity.
    Mean face scores average the actual unit-normalized face vectors. Missing
    faces stay zero with explicit coverage masks, never "closed eyes"/bad pose.
    """
    values = np.asarray(region_features, dtype=np.float32)
    texts = np.asarray(description_vectors, dtype=np.float32)
    if (values.ndim != 2 or values.shape[1] != REGION_DIMENSION
            or texts.shape != (len(CONCEPTS), EMBEDDING_DIMENSION)
            or not np.isfinite(values).all() or not np.isfinite(texts).all()):
        raise ValueError("Incomplete or nonfinite semantic content features")
    norms = np.linalg.norm(texts, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Empty semantic description direction")
    texts = texts / norms
    sections = [values[:, i*EMBEDDING_DIMENSION:(i+1)*EMBEDDING_DIMENSION] @ texts.T
                for i in (0, 1, 2, 5)]
    result = np.concatenate((*sections, values[:, -4:]), axis=1).astype(np.float32)
    if result.shape != (len(values), FEATURE_DIMENSION) or not np.isfinite(result).all():
        raise ValueError("Invalid semantic content projection")
    return result

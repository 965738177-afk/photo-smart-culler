"""Read-only content-axis experiment; no further RAW decoding or app activation."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

from src.core.image_features.semantic_regions import SemanticRegionEncoder, PIPELINE_ID as REGION_PIPELINE
from src.core.image_features.semantic_concepts import DESCRIPTIONS, PIPELINE_ID, project_concepts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("regions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    # Resolve the existing shared core package exactly as the extraction CLI.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from core.model_provisioning import SEMANTIC_RETENTION_TEXT_MODEL
    with np.load(args.regions, allow_pickle=False) as frozen:
        if str(frozen["pipeline"]) != REGION_PIPELINE:
            parser.error("Shared SigLIP2 region feature version required")
        paths = frozen["paths"].copy()
        regions = frozen["features"].copy()
    if (args.output.exists() or len(paths) != len(regions) or len(set(paths)) != len(paths)
            or any(args.output.resolve().is_relative_to(Path(p).parent.resolve()) for p in paths)):
        parser.error("New output outside photo folders and complete unique inputs required")
    started = time.monotonic()
    with closing(SemanticRegionEncoder()) as owner:
        texts = owner.encode_texts(DESCRIPTIONS)
        features = project_concepts(regions, texts)
        device = owner.device
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "features.npz", paths=paths, features=features,
                        pipeline=np.asarray(PIPELINE_ID))
    np.savez_compressed(args.output / "descriptions.npz", texts=np.asarray(DESCRIPTIONS), vectors=texts)
    with args.regions.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"images": len(paths), "dimensions": features.shape[1], "pipeline": PIPELINE_ID,
              "model": SEMANTIC_RETENTION_TEXT_MODEL.repo_id,
              "revision": SEMANTIC_RETENTION_TEXT_MODEL.revision, "device": device,
              "seconds": time.monotonic()-started, "description_count": len(DESCRIPTIONS),
              "source_regions": str(args.regions), "source_sha256": source_hash,
              "shared_preview_decodes": 0, "new_face_detector_calls": 0,
              "profile_changed": False, "similarity_is_verified_detection": False,
              "descriptions": list(DESCRIPTIONS)}
    with (args.output / "projection.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "descriptions"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

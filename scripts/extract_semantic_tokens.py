"""Cache label-free spatial tokens using the shared preview/semantic owners.

An exclusive isolated experiment directory is created. Incomplete runs have no
complete token manifest and must never be treated as successful extraction.
No application cache/profile, source file or XMP is changed; no upload occurs.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import cv2
import numpy as np
from PIL import Image

from event_cull.engine import _load_preview
from scripts.extract_retention_embeddings import owned_preview_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tiny-vision", action="store_true")
    args = parser.parse_args()
    baseline_bytes = args.baseline.read_bytes()
    payload = json.loads(baseline_bytes)
    rows = payload["photos"]
    if args.limit is not None:
        if not 1 <= args.limit <= len(rows):
            parser.error("invalid smoke-test limit")
        rows = [rows[i] for i in np.linspace(0,len(rows)-1,args.limit,dtype=int)]
    paths = [Path(row["folder"])/row["file"] for row in rows]
    if not paths or args.output.exists() or any(args.output.resolve().is_relative_to(p.parent.resolve()) for p in paths):
        parser.error("new isolated output outside photo directories required")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
    from src.core.image_features.semantic_regions import SemanticRegionEncoder
    from src.core.model_provisioning import SEMANTIC_RETENTION_MODEL,TINY_RETENTION_MODEL
    owner = SemanticRegionEncoder(tiny_vision=args.tiny_vision)
    spec = owner.spatial_spec
    managed = TINY_RETENTION_MODEL if args.tiny_vision else SEMANTIC_RETENTION_MODEL
    from src.core.image_features.human_analysis import validate_observation
    for row in rows:
        validate_observation(row.get("person_evidence"), require_details=True)
    existing_parent = args.output.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    maximum_bytes = len(paths)*6*spec.token_count*spec.dimension*2
    if shutil.disk_usage(existing_parent).free < maximum_bytes+1024**3:
        parser.error("insufficient space for the bounded spatial token cache")
    args.output.mkdir(parents=True)
    metadata = []
    started, region_count, last = time.monotonic(), 0, 0
    def preview(path):
        image = _load_preview(path, maximum_dimension=1600)
        return Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    try:
        with ExitStack() as lifetime, ThreadPoolExecutor(max_workers=6) as readers:
            lifetime.callback(owner.close)
            owner.load()
            with (args.output/"tokens.f16").open("xb") as stream:
                for start in range(0,len(paths),16):
                    with owned_preview_batch(readers,preview,paths[start:start+16]) as images:
                        tokens, entries = owner.encode_prefix(images,[row["person_evidence"] for row in rows[start:start+16]],
                                                             verify_boundary=args.tiny_vision and start == 0)
                        for path, value, entry in zip(paths[start:start+16],tokens,entries):
                            value.tofile(stream)
                            metadata.append({"path":str(path), "start":region_count, "count":len(value), **entry})
                            region_count += len(value)
                    now = time.monotonic()
                    if now-last >= 20 or start+16 >= len(paths):
                        print(f"shared prefix tokens {len(metadata)}/{len(paths)}; regions {region_count}; {now-started:.1f}s",flush=True)
                        last = now
        if len(metadata) != len(paths):
            raise ValueError("Incomplete token extraction; no silent subset")
        report = {"complete":True, "pipeline":spec.pipeline, "tail_layers":spec.trainable_layers,
                  "model":managed.repo_id, "revision":managed.revision,
                  "baseline":str(args.baseline), "baseline_sha256":hashlib.sha256(baseline_bytes).hexdigest(),
                  "images":len(paths), "regions":region_count, "token_count":spec.token_count,
                  "dimension":spec.dimension, "dtype":"float16", "device":owner.device,
                  "initial_boundary_check":owner.last_prefix_check,
                  "shared_preview_decodes":len(paths), "new_detector_calls":0,
                  "training_performed":False, "active_profile_changed":False, "smoke_subset":bool(args.limit),
                  "seconds":time.monotonic()-started, "records":metadata}
        with (args.output/"tokens.json").open("x",encoding="utf-8") as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2)
        print(json.dumps({k:v for k,v in report.items() if k!="records"},ensure_ascii=False,indent=2),flush=True)
    except BaseException as exc:
        with (args.output/"failed.json").open("x",encoding="utf-8") as stream:
            json.dump({"complete":False,"images_finished":len(metadata),"regions":region_count,
                       "error":str(exc),"seconds":time.monotonic()-started},stream,ensure_ascii=False,indent=2)
        raise


if __name__ == "__main__":
    main()

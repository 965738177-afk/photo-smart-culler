"""Isolated local semantic-feature experiment using the existing shared owner.

No profile activation, directory scan, second workflow decoder, manual labels,
cloud inference, source mutation, or arbitrary model code download.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import sys
import time
import numpy as np
import cv2
from PIL import Image

from event_cull.engine import _load_preview

PIPELINE_ID="dinov2-small-visual-patches-2x2-v1"


@contextmanager
def owned_preview_batch(executor, loader, paths):
    """Close every owned PIL result even if a different decoder fails first.

    A failed ordered map can hide already completed/later in-flight results.
    Bound futures to one batch and attach cleanup to any unfinished decode,
    instead of leaking its image or waiting for unrelated future batches.
    """
    futures = []
    def close_result(future):
        if not future.cancelled():
            try:
                image = future.result()
            except Exception:
                return  # Decode error is raised by the consumer, not cleanup.
            image.close()
    try:
        for path in paths:
            futures.append(executor.submit(loader, path))
        yield [future.result() for future in futures]
    finally:
        for future in futures:
            if not future.cancel():
                future.add_done_callback(close_result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--limit",type=int)
    parser.add_argument("--semantic-regions",action="store_true",help="local SigLIP2 whole/context/face comparison")
    args=parser.parse_args()
    payload=json.loads(args.baseline.read_text(encoding="utf-8"))
    records=payload["photos"]
    if args.limit is not None and not 1 <= args.limit <= len(records):
        parser.error("limit must be between 1 and the complete photo count")
    if args.limit:
        # Spread the smoke test over the entire wedding, not only makeup.
        records=[records[i] for i in np.linspace(0,len(records)-1,args.limit,dtype=int)]
    paths=[Path(r["folder"])/r["file"] for r in records]
    if args.output.exists() or any(args.output.resolve().is_relative_to(p.parent.resolve()) for p in paths):
        parser.error("new evaluation output outside original directories required")
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
    from core.model_provisioning import EMBEDDING_MODEL,SEMANTIC_RETENTION_MODEL
    started=time.monotonic()
    if args.semantic_regions:
        from core.image_features.semantic_regions import SemanticRegionEncoder,PIPELINE_ID as pipeline
        from core.image_features.human_analysis import PIPELINE_ID as person_pipeline
        model=SemanticRegionEncoder()
        model_spec=SEMANTIC_RETENTION_MODEL
        if any(row.get("person_evidence",{}).get("status")!="ok"
               or row.get("person_evidence",{}).get("pipeline")!=person_pipeline for row in records):
            parser.error("complete frozen person boxes required; no new face detection")
    else:
        from core.similarity_embedding_model import SimilarityEmbeddingModel
        from core.subject_grouping_models import _dense_grid_evidence
        model=SimilarityEmbeddingModel(allow_download=False)
        model_spec,pipeline=EMBEDDING_MODEL,PIPELINE_ID
    features=[]
    last=0
    def preview(path):
        image=_load_preview(path,maximum_dimension=1600 if args.semantic_regions else 448)
        return Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    with ExitStack() as lifetime, ThreadPoolExecutor(max_workers=6) as readers:
        if args.semantic_regions:
            lifetime.callback(model.close)
        model.load()
        for start in range(0,len(paths),16):
            with owned_preview_batch(readers, preview, paths[start:start+16]) as images:
                if args.semantic_regions:
                    features.extend(model.encode(images,[row["person_evidence"] for row in records[start:start+16]]))
                else:
                    globals,patches=model.encode_with_patches(images)
                    for full,patch in zip(globals,patches):
                        # Same spatial-pooling owner as the subject grouping engine;
                        # never average unrelated learned feature channels.
                        grids=_dense_grid_evidence(patch,grid_size=2)
                        features.append(np.concatenate([full,*[g.descriptor for g in grids]]))
            now=time.monotonic()
            if now-last>=20 or start+16>=len(paths):
                print(f"shared semantic features {min(start+16,len(paths))}/{len(paths)}, {now-started:.1f}s",flush=True)
                last=now
    values=np.asarray(features,dtype=np.float32)
    if len(values)!=len(paths) or not np.isfinite(values).all():
        raise ValueError("incomplete semantic features")
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/"features.npz",features=values,paths=np.asarray([str(p) for p in paths]),
                        pipeline=np.asarray(pipeline))
    summary={"images":len(paths),"seconds":time.monotonic()-started,"dimensions":values.shape[1],
             "model":model_spec.repo_id,"revision":model_spec.revision,"device":model.device,
             "pipeline":pipeline,"profile_changed":False,"smoke_subset":bool(args.limit),
             "semantic_regions_encoded":getattr(model,"regions_encoded",None),
             "shared_preview_decodes":len(paths),"new_face_detector_calls":0}
    with (args.output/"extraction.json").open("x",encoding="utf-8") as stream:
        json.dump(summary,stream,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()

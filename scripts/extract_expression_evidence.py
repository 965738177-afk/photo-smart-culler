"""Read-only expression experiment; reuse person boxes and shared RAW decoder."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np

from event_cull.engine import _load_preview
from src.core.image_features.human_analysis import bundled_model
from src.core.image_features.expression_analysis import ExpressionEvidenceService,PIPELINE_ID

MODELS={"facial_expression_recognition_mobilefacenet_2022july.onnx":
        "4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9",
        "face_detection_yunet_2023mar.onnx":
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    data=json.loads(args.baseline.read_text(encoding="utf-8"))
    rows=data["photos"]
    if args.limit:
        rows=[rows[i] for i in np.linspace(0,len(rows)-1,args.limit,dtype=int)]
    paths=[Path(r["folder"])/r["file"] for r in rows]
    if args.output.exists() or any(args.output.resolve().is_relative_to(p.parent.resolve()) for p in paths):
        parser.error("New output outside photograph directories required")
    for name,digest in MODELS.items():
        if hashlib.sha256(bundled_model(name).read_bytes()).hexdigest()!=digest:
            parser.error("Expression weights differ from pinned upstream LFS hashes")
    started,last=time.monotonic(),0.
    features=[]
    observed=unknown=0
    service=ExpressionEvidenceService()
    def preview(path):
        return cv2.cvtColor(_load_preview(path,maximum_dimension=1600),cv2.COLOR_BGR2RGB)
    try:
        with ThreadPoolExecutor(max_workers=6) as readers:
            for start in range(0,len(paths),16):
                images=list(readers.map(preview,paths[start:start+16]))
                for rgb,row in zip(images,rows[start:start+16]):
                    person=row.get("person_evidence",{})
                    if person.get("status")!="ok":
                        raise ValueError("No subset evaluation of incomplete person evidence")
                    result=service.inspect(rgb,faces=person["faces"])
                    features.append(result["features"])
                    observed+=len(result["faces"])
                    unknown+=result["unknown_faces"]
                now=time.monotonic()
                if now-last>20 or start+16>=len(paths):
                    print(f"aligned expression features {min(start+16,len(paths))}/{len(paths)}, {now-started:.1f}s",flush=True)
                    last=now
    finally:
        service.close()
    values=np.asarray(features,dtype=np.float32)
    if len(values)!=len(paths) or not np.isfinite(values).all():
        raise ValueError("Incomplete expression evidence")
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/"features.npz",features=values,paths=np.asarray([str(p) for p in paths]),
                        pipeline=np.asarray(PIPELINE_ID))
    report={"images":len(paths),"seconds":time.monotonic()-started,"observed_faces":observed,
            "unknown_faces":unknown,"pipeline":PIPELINE_ID,"model_sha256":MODELS,
            "smoke_subset":bool(args.limit),"profile_changed":False}
    with (args.output/"extraction.json").open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=="__main__":main()

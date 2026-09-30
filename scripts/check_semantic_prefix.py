"""Check cached prefix + unchanged tail against frozen full-image embeddings.

This proves the numerical split, not photographic accuracy. No decode/train.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from event_cull.semantic_tail_ranker import SpatialTokenCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tokens","frozen_features","output"):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new numerical-check output required")
    cache = SpatialTokenCache(args.tokens,allow_smoke=True)
    paths = [entry["path"] for entry in cache.records]
    if any(args.output.resolve().is_relative_to(Path(path).parent.resolve()) for path in paths):
        cache.close()
        parser.error("output must be outside photo sources")
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
    from core.image_features.semantic_regions import (SemanticRegionEncoder,aggregate_regions,
                                                      PIPELINE_ID, BATCH_SIZE)
    with np.load(args.frozen_features,allow_pickle=False) as saved:
        names = list(saved["paths"])
        if len(names) != len(set(names)) or str(saved["pipeline"]) != PIPELINE_ID or not set(paths).issubset(names):
            cache.close()
            parser.error("matching unique frozen semantic features required")
        indexes = {path:index for index,path in enumerate(names)}
        expected = saved["features"][[indexes[path] for path in paths]]
    owner = SemanticRegionEncoder()
    started = time.monotonic()
    try:
        vectors = []
        for start in range(0,len(cache.tokens),BATCH_SIZE):
            vectors.append(owner.finish_prefix(cache.tokens[start:start+BATCH_SIZE]))
        vectors = np.concatenate(vectors)
        actual = np.stack([aggregate_regions(vectors[e["start"]:e["start"]+e["count"]],e["kinds"],e["masks"])
                           for e in cache.records])
        error = np.abs(actual[:,:-4]-expected[:,:-4])
        whole_cosines = np.sum(actual[:,:768]*expected[:,:768],axis=1)
        passed = bool(np.array_equal(actual[:,-4:],expected[:,-4:]) and error.max() < .003
                      and whole_cosines.min() > .999)
        report = {"tokens":str(args.tokens),"frozen_features":str(args.frozen_features),
                  "images":len(paths),"regions":len(vectors),"max_absolute_error":float(error.max()),
                  "mean_absolute_error":float(error.mean()),"minimum_whole_cosine":float(whole_cosines.min()),
                  "observation_masks_identical":bool(np.array_equal(actual[:,-4:],expected[:,-4:])),
                  "passed_numerical_boundary":passed,"training_performed":False,"image_decodes":0,
                  "seconds":time.monotonic()-started,"release_approved":False}
        args.output.mkdir(parents=True)
        with (args.output/"check.json").open("x",encoding="utf-8") as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        if not passed:
            raise ValueError("Prefix/tail split is not numerically equivalent; no valid fine-tuning experiment")
    finally:
        owner.close()
        cache.close()


if __name__ == "__main__":
    main()

"""Isolated task-trained vision evaluation; no active model/profile mutation.

Every held-out frame and adjacent continuous blocks are absent from training.
Legacy fixed epochs or training-side validation, never held-out epoch selection. --fold is explicitly
a partial pilot, not full-batch acceptance. Cached tokens are label-free.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

from event_cull import calibration, engine
from event_cull.semantic_tail_ranker import (SemanticTailRanker, SpatialTokenCache, PIPELINE_ID,TRAINING_EPOCHS,
                                            SCALED_PIPELINE_ID,MAX_TRAINING_EPOCHS,EMBEDDING_DIMENSION,
                                            TINY_TRAINED_PIPELINE_ID)
from event_cull.selection_compare import compare_selection
from scripts.evaluate_person_retention import frozen_group_ids
from scripts.evaluate_retention_calibration import metrics,purged_folds,result_from_row,train_side_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","manual","tokens","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--fold",type=int,choices=range(5))
    parser.add_argument("--nested-scaled",action="store_true",help="train-side checkpoint selection and label-free modality scaling")
    parser.add_argument("--tiny-vision",action="store_true",help="train all ten small vision blocks; patch embeddings stay frozen")
    args = parser.parse_args()
    if args.tiny_vision and not args.nested_scaled:
        parser.error("Tiny vision requires training-side checkpoint selection")
    data = args.baseline.read_bytes()
    payload = json.loads(data)
    compared = compare_selection(payload,args.manual,confirmed_complete=True)
    if (compared["ambiguous_stems"] or compared["unmatched_human_stems"]
            or any(row["ai_rating"] is None for row in compared["photos"])):
        parser.error("unique complete original delivery and pure frozen AI baseline required")
    if args.output.exists() or args.output.resolve().is_relative_to(args.manual.resolve()) or any(
            args.output.resolve().is_relative_to(Path(row["folder"]).resolve()) for row in payload["photos"]):
        parser.error("new isolated output outside photo directories required")
    results = [result_from_row(row) for row in payload["photos"]]
    paths = [r.path for r in results]
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
    from src.core.image_features.semantic_regions import SemanticRegionEncoder
    from src.core.model_provisioning import SEMANTIC_RETENTION_MODEL,TINY_RETENTION_MODEL
    base = SemanticRegionEncoder(tiny_vision=args.tiny_vision)
    spec = base.spatial_spec
    managed = TINY_RETENTION_MODEL if args.tiny_vision else SEMANTIC_RETENTION_MODEL
    cache = SpatialTokenCache(args.tokens,expected_paths=paths,
                              expected_model=(managed.repo_id,managed.revision),spatial_spec=spec)
    if cache.report["baseline_sha256"] != hashlib.sha256(data).hexdigest():
        cache.close()
        parser.error("token frozen-baseline hash differs")
    features = np.asarray([row["person_evidence"]["features"] for row in payload["photos"]],np.float32)
    labels = np.asarray([row["human_retained"] for row in compared["photos"]],bool)
    groups = frozen_group_ids(results)
    blocks = calibration.blocked_validation_groups(groups)
    options = engine.AnalysisOptions(**payload["settings"])
    args.output.mkdir(parents=True)
    scores,predictions,evaluated = np.zeros(len(results)),np.zeros(len(results),bool),np.zeros(len(results),bool)
    folds = []
    started = time.monotonic()
    try:
        base.load()
        for number,(training,held) in enumerate(purged_folds(blocks)):
            if args.fold is not None and number != args.fold:
                continue
            progress = lambda epoch,total,loss,seconds: print(
                f"task-trained visual fold {number} epoch {epoch}/{total}; loss {loss:.4f}; {seconds:.1f}s",flush=True)
            batch_progress = lambda epoch,current,total,seconds: print(
                f"visual fold {number}, epoch {epoch}: trained {current}/{total}; {seconds:.1f}s",flush=True)
            selection = None
            epochs,semantic_scale = TRAINING_EPOCHS,1.
            if args.nested_scaled:
                inner_training,inner_validation = train_side_validation(blocks,training)
                development = SemanticTailRanker(base.copy_trainable_tail,device=base.device,
                    progress=progress,batch_progress=batch_progress,mixed_precision=base.device=="mps",
                    semantic_scale=spec.dimension**.5,epochs=MAX_TRAINING_EPOCHS,spatial_spec=spec,
                    validation_progress=lambda row:print("training-side validation: "+json.dumps(row),flush=True))
                try:
                    development.fit(cache,features,labels,inner_training,groups,validation=inner_validation,patience=3)
                    epochs,semantic_scale = development.selected_epoch,spec.dimension**.5
                    selection = {"fit_examples":int(inner_training.sum()),"validation_examples":int(inner_validation.sum()),
                        "both_masks_inside_outer_training":bool(not ((inner_training|inner_validation)&held).any()),
                        "selected_epoch":epochs,"maximum_epochs":MAX_TRAINING_EPOCHS,
                        "patience":3,"criterion":"training_side_average_precision", "history":development.validation_history}
                finally:
                    development.close()
                with (args.output/f"development-{number}.json").open("x",encoding="utf-8") as stream:
                    json.dump(selection,stream,ensure_ascii=False,indent=2)
                print(f"Outer fold {number}: refit all {training.sum()} training frames for {epochs} epochs",flush=True)
            owner = SemanticTailRanker(base.copy_trainable_tail,device=base.device,
                                       progress=progress,batch_progress=batch_progress,mixed_precision=base.device=="mps",
                                       epochs=epochs,semantic_scale=semantic_scale,spatial_spec=spec)
            try:
                owner.fit(cache,features,labels,training,groups)
                density = owner.positive_fraction
                held_indexes = np.flatnonzero(held)
                infer_started = time.monotonic()
                scores[held] = owner.predict(cache,features,held_indexes)
                inference_seconds = time.monotonic()-infer_started
            finally:
                owner.close()
            items = [results[i] for i in held_indexes]
            for item,score in zip(items,scores[held]):
                item.calibration_confidence = float(score)
            floor = round((options.target_keep_count or 1023)*held.mean())
            ceiling = round((options.target_keep_max_count or 1200)*held.mean())
            take = max(floor,min(ceiling,round(len(items)*density)))
            grouped = {}
            for item in items:
                grouped.setdefault(item.group_id,[]).append(item)
            engine._assign_tiered_ratings(list(grouped.values()),replace(
                options,target_keep_count=floor,target_keep_max_count=take))
            predictions[held] = [item.suggested_rating > 0 for item in items]
            evaluated |= held
            summary = metrics(labels[held],predictions[held])
            summary.update(fold=number,training_examples=int(training.sum()),held_out_examples=int(held.sum()),
                           auc=calibration._roc_auc(labels[held],scores[held]),
                           cached_tail_inference_seconds=inference_seconds,
                           training_epochs=epochs,semantic_scale=semantic_scale,development=selection)
            folds.append(summary)
            with (args.output/f"fold-{number}.json").open("x",encoding="utf-8") as stream:
                json.dump(summary,stream,ensure_ascii=False,indent=2)
            print(json.dumps(summary,ensure_ascii=False),flush=True)
        outcome = metrics(labels[evaluated],predictions[evaluated])
        outcome["auc"] = calibration._roc_auc(labels[evaluated],scores[evaluated])
        complete = bool(evaluated.all())
        outcome["passed_acceptance"] = complete and outcome["precision"] >= .75 and outcome["recall"] >= .75
        report = {"baseline":str(args.baseline),"manual":str(args.manual),"tokens":str(args.tokens),
                  "pipeline":TINY_TRAINED_PIPELINE_ID if args.tiny_vision else SCALED_PIPELINE_ID if args.nested_scaled else PIPELINE_ID,
                  "visual_layers_trained":spec.trainable_layers,"patch_embeddings_trained":False,
                  "model":managed.repo_id,"revision":managed.revision,
                  "baseline_sha256":hashlib.sha256(data).hexdigest(),
                  "training_epochs":None if args.nested_scaled else TRAINING_EPOCHS,
                  "nested_validation":args.nested_scaled,"semantic_scale":semantic_scale,
                  "training_precision":"float32_weights_bfloat16_autocast" if base.device=="mps" else "float32",
                  "optimizer":"AdamW_fused" if base.device=="mps" else "AdamW",
                  "scope":"same_wedding_purged_5_fold_continuous_100_frame_blocks",
                  "complete_full_batch":complete,"evaluated_count":int(evaluated.sum()),
                  "out_of_fold":outcome,"folds":folds,"evaluation_seconds":time.monotonic()-started,
                  "scores_are_calibrated_probabilities":False,"retouch_accuracy":None,
                  "exploratory_validation":True,"release_approved":False,"active_profile_changed":False,
                  "additional_image_decodes":0,"photos":[{"file":r.path.name,"evaluated":bool(evaluated[i]),
                      "rating":r.suggested_rating if evaluated[i] else None,
                      "score":float(scores[i]) if evaluated[i] else None,"human_retained":bool(labels[i])}
                     for i,r in enumerate(results)]}
        with (args.output/"evaluation.json").open("x",encoding="utf-8") as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2)
        print(json.dumps({k:v for k,v in report.items() if k!="photos"},ensure_ascii=False,indent=2),flush=True)
    except BaseException as exc:
        with (args.output/"failed.json").open("x",encoding="utf-8") as stream:
            json.dump({"complete":False,"error":str(exc),"finished_folds":folds},stream,ensure_ascii=False,indent=2)
        raise
    finally:
        base.close()
        cache.close()


if __name__ == "__main__":
    main()

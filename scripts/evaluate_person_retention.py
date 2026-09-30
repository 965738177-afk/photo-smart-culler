"""Evaluate experimental person/relative ranking; never change active profiles.

Five purged continuous-block folds, final shared technical/dedup/star gates,
train-only density, exact-file precision AND recall. Same wedding validation
is not a new-wedding guarantee. No photos, identifiers or label files uploaded.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np

from event_cull import calibration,engine
from event_cull.retention_ranker import (RetentionRanker,PairwiseRetentionRanker,observed_features,
                                        combine_pairwise_scores)
from event_cull.selection_compare import compare_selection
from scripts.evaluate_retention_calibration import metrics,purged_folds,result_from_row


def frozen_group_ids(results):
    """Snapshot burst identities before shared rating assignment renumbers them.

    These identifiers only delimit comparisons. They are not classifier inputs.
    Keeping live PhotoResult IDs across folds would merge unrelated training
    bursts after a previously held-out fold is re-rated.
    """
    groups=np.asarray([str(r.group_id) for r in results])
    groups.setflags(write=False)
    return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline",type=Path)
    parser.add_argument("manual",type=Path)
    parser.add_argument("visual_features",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--pairwise",action="store_true",help="experimental bounded within-burst comparisons")
    parser.add_argument("--classifier",choices=("boosted","linear","rbf","neural","person-set","person-set-body"),default="boosted")
    parser.add_argument("--expression-features",type=Path,help="matching immutable aligned expression evidence")
    args=parser.parse_args()
    set_strategy=args.classifier in {"person-set","person-set-body"}
    with_bodies=args.classifier=="person-set-body"
    if set_strategy and args.pairwise:
        parser.error("person-set already has training-only burst supervision; do not apply a second pair classifier")
    payload=json.loads(args.baseline.read_text(encoding="utf-8"))
    comparison=compare_selection(payload,args.manual,confirmed_complete=True)
    if (comparison["ambiguous_stems"] or comparison["unmatched_human_stems"]
            or any(p["ai_rating"] is None for p in comparison["photos"])):
        parser.error("unique complete manual delivery and pure frozen AI baseline required")
    source_dirs={Path(p["folder"]).resolve() for p in payload["photos"]}|{args.manual.resolve()}
    if any(args.output.resolve()==p or args.output.resolve().is_relative_to(p) for p in source_dirs):
        parser.error("evaluation must be outside photo directories")
    if args.output.exists():
        parser.error("a new non-overwriting output directory is required")
    results=[result_from_row(row) for row in payload["photos"]]
    for result,row in zip(results,payload["photos"]):
        result.person_evidence=row.get("person_evidence",{})
    with np.load(args.visual_features,allow_pickle=False) as frozen:
        if list(frozen["paths"])!=[str(r.path) for r in results]:
            parser.error("visual feature identities/order differ from frozen baseline")
        visual_pipeline=str(frozen["pipeline"])
        from src.core.image_features.semantic_regions import PIPELINE_ID as semantic_pipeline
        from src.core.image_features.semantic_concepts import PIPELINE_ID as concept_pipeline
        if visual_pipeline not in {calibration.DEEP_PIPELINE_ID,calibration.CPU_DEEP_PIPELINE_ID,
                                   "dinov2-small-visual-patches-2x2-v1",semantic_pipeline,concept_pipeline}:
            parser.error("incompatible visual feature pipeline")
        visual=frozen["features"].copy()
    expression_pipeline=None
    if args.expression_features:
        from src.core.image_features.expression_analysis import PIPELINE_ID,FEATURE_DIMENSION
        with np.load(args.expression_features,allow_pickle=False) as extra:
            if (list(extra["paths"])!=[str(r.path) for r in results]
                    or str(extra["pipeline"])!=PIPELINE_ID
                    or extra["features"].shape!=(len(results),FEATURE_DIMENSION)
                    or not np.isfinite(extra["features"]).all()):
                parser.error("Incomplete or mismatched expression evidence; no silent subset")
            expression_pipeline=PIPELINE_ID
            visual=np.concatenate((visual,extra["features"]),axis=1)
    started=time.monotonic()
    features=observed_features(results,visual)
    labels=np.asarray([row["human_retained"] for row in comparison["photos"]],dtype=bool)
    blocks=calibration.blocked_validation_groups([str(r.group_id) for r in results])
    options=engine.AnalysisOptions(**payload["settings"])
    predictions=np.zeros(len(results),dtype=bool)
    scores=np.zeros(len(results))
    source_groups=frozen_group_ids(results)
    sets=mask=bodies=body_mask=None
    if set_strategy:
        from event_cull.person_set_ranker import (PersonSetRanker,person_sets,body_sets,
                                                 PERSON_SET_PIPELINE,JOINT_SET_PIPELINE)
        sets,mask=person_sets(results)
        if with_bodies:
            bodies,body_mask=body_sets(results)
    folds=[]
    for number,(training,held) in enumerate(purged_folds(blocks)):
        if set_strategy:
            model=PersonSetRanker(with_bodies=with_bodies,progress=lambda epoch,total: print(
                f"person-set fold {number}: epoch {epoch}/{total}",flush=True))
            try:
                body_training={} if not with_bodies else {"bodies":bodies[training],"body_mask":body_mask[training]}
                body_held={} if not with_bodies else {"bodies":bodies[held],"body_mask":body_mask[held]}
                model.fit(features[training],labels[training],sets[training],mask[training],
                          groups=source_groups[training],**body_training)
                scores[held]=model.predict(features[held],sets[held],mask[held],**body_held)
            finally:
                model.close()
        else:
            model=RetentionRanker(strategy=args.classifier).fit(features[training],labels[training])
            scores[held]=model.predict(features[held])
        if args.pairwise:
            pair=PairwiseRetentionRanker(strategy=args.classifier).fit_pairs(features[training],labels[training],source_groups[training])
            # Preserve the absolute scene/delivery score; an arbitrary singleton
            # or a best-of-bad burst must not win just by relative comparison.
            relative=pair.compare(features[held],source_groups[held])
            scores[held]=combine_pairwise_scores(scores[held], relative)
            pair.close()
        items=[results[i] for i in np.flatnonzero(held)]
        for item,score in zip(items,scores[held]):
            item.calibration_confidence=float(score)
        floor=round((options.target_keep_count or 1023)*held.mean())
        ceiling=round((options.target_keep_max_count or 1200)*held.mean())
        take=max(floor,min(ceiling,round(len(items)*model.positive_fraction)))
        groups={}
        for item in items:
            groups.setdefault(item.group_id,[]).append(item)
        engine._assign_tiered_ratings(list(groups.values()),replace(
            options,target_keep_count=floor,target_keep_max_count=take))
        predictions[held]=[r.suggested_rating>0 for r in items]
        score=metrics(labels[held],predictions[held])
        score.update(fold=number,training_examples=int(training.sum()),held_out_examples=int(held.sum()))
        folds.append(score)
        print(json.dumps(score,ensure_ascii=False),flush=True)
    outcome=metrics(labels,predictions)
    outcome["auc"]=calibration._roc_auc(labels,scores)
    outcome["passed_acceptance"]=(outcome["precision"]>=calibration.MIN_RETENTION_AGREEMENT
                                   and outcome["recall"]>=calibration.MIN_RETENTION_AGREEMENT)
    report={"baseline":str(args.baseline),"manual":str(args.manual),"visual":str(args.visual_features),
            "feature_dimensions":features.shape[1],"visual_pipeline":visual_pipeline,
            "evaluation_seconds":time.monotonic()-started,
            "scope":"same_wedding_purged_5_fold_continuous_100_frame_blocks",
            "pairwise_comparisons":args.pairwise,
            "classifier":args.classifier,"scores_are_calibrated_probabilities":False,
            "expression_pipeline":expression_pipeline,
            "person_set_pipeline":(JOINT_SET_PIPELINE if with_bodies else PERSON_SET_PIPELINE) if sets is not None else None,
            "person_set_maximum_faces":sets.shape[1] if sets is not None else None,
            "person_set_maximum_poses":bodies.shape[1] if bodies is not None else None,
            "person_set_training_epochs":60 if sets is not None else None,
            "group_assignments_frozen_before_rating":True,
            "exploratory_validation":True,"release_approved":False,
            "required_precision_and_recall":calibration.MIN_RETENTION_AGREEMENT,
            "out_of_fold":outcome,"folds":folds,"active_profile_changed":False,"retouch_accuracy":None,
            "photos":[{"file":r.path.name,"rating":r.suggested_rating,"score":float(scores[i]),
                       "human_retained":bool(labels[i])} for i,r in enumerate(results)]}
    args.output.mkdir(parents=True)
    with (args.output/"evaluation.json").open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!="photos"},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()

"""Isolated burst attention validation; no photos decoded or profile activated.

Cases must be complete unfiltered original-retention references. A preculled
positive control is explicitly refused. Model selection uses training-side
continuous blocks only; an entire different wedding may be tested once after
the trained model and epoch count are frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass,replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from event_cull import calibration,engine
from event_cull.burst_set_ranker import BurstSetRanker,PIPELINE_ID,MAX_EPOCHS
from event_cull.person_set_ranker import person_sets,body_sets
from event_cull.retention_ranker import observed_features,PairedRetentionRanker
from event_cull.selection_compare import compare_selection
from scripts.evaluate_retention_calibration import (metrics,purged_folds,result_from_row,
                                                   train_side_validation)


@dataclass
class CaseData:
    case: dict
    baseline: Path
    feature_file: Path
    baseline_sha256: str
    feature_sha256: str
    results: list
    groups: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    faces: np.ndarray
    face_mask: np.ndarray
    bodies: np.ndarray
    body_mask: np.ndarray
    options: engine.AnalysisOptions


def validate_recovery_measurements(rows, settings):
    """Do not silently replace requested recovered observations with old ones."""
    recovery = settings.get("human_face_recovery_enabled", False)
    if type(recovery) is not bool:
        raise ValueError("Invalid recovery measurement configuration")
    if recovery:
        from src.core.image_features.face_analysis import FACE_REGION_PIPELINE
        if not settings.get("human_analysis_enabled") or any(
                row.get("person_evidence", {}).get("region_pipeline") != FACE_REGION_PIPELINE for row in rows):
            raise ValueError("Requested recovered face measurements incomplete")


def require_comparable_observations(training_options, test_options):
    """Same feature dimensions do not make different detectors interchangeable."""
    if training_options.human_face_recovery_enabled != test_options.human_face_recovery_enabled:
        raise ValueError("Training/test face measurement pipelines differ")
    if training_options.calibration_feature_pipeline != test_options.calibration_feature_pipeline:
        raise ValueError("Training/test visual measurement pipelines differ")


def load_case(case,baseline,feature_file):
    if (case.get("source_preculled") or not case.get("retention_labels_complete")
            or "positive_control" in case.get("role","")):
        raise ValueError("Preculled/partial reference cannot be full-selection supervised validation")
    raw=Path(baseline).read_bytes()
    payload=json.loads(raw)
    rows=payload.get("photos",[])
    source=Path(case["originals"]).resolve()
    if (len(rows)!=case["original_count"] or not rows
            or any(Path(row["folder"]).resolve()!=source for row in rows)):
        raise ValueError("Case source identity/count differs from frozen baseline")
    validate_recovery_measurements(rows, payload.get("settings", {}))
    compared=compare_selection(payload,Path(case["selected_originals"]),confirmed_complete=True)
    if (compared["metrics"] is None or compared["human_unique_stems"]!=case["human_retained_count"]
            or any(row["ai_rating"] is None for row in compared["photos"])):
        raise ValueError("Unique complete pure AI advice and original delivery required")
    results=[result_from_row(row) for row in rows]
    for item,row in zip(results,rows):
        item.person_evidence=row.get("person_evidence",{})
    with np.load(feature_file,allow_pickle=False) as saved:
        if (str(saved["pipeline"]) not in {calibration.DEEP_PIPELINE_ID,calibration.CPU_DEEP_PIPELINE_ID}
                or (payload.get("settings",{}).get("calibration_feature_pipeline")
                    and str(saved["pipeline"])!=payload["settings"]["calibration_feature_pipeline"])
                or list(saved["paths"])!=[str(r.path) for r in results]
                or saved["features"].shape!=(len(results),832)
                or not np.isfinite(saved["features"]).all()):
            raise ValueError("Incomplete/mismatched shared visual descriptors")
        features=observed_features(results,saved["features"].copy())
    faces,mask=person_sets(results)
    bodies,body_mask=body_sets(results)
    groups=np.asarray([f"{case['case_id']}:{r.group_id}" for r in results])
    groups.setflags(write=False)
    return CaseData(case,Path(baseline),Path(feature_file),hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(Path(feature_file).read_bytes()).hexdigest(),results,groups,features,
        np.asarray([row["human_retained"] for row in compared["photos"]],bool),faces,mask,bodies,body_mask,
        engine.AnalysisOptions(**payload["settings"]))


def fit_selected(data,eligible,output,name):
    """Select epochs inside training only, then refit all eligible examples."""
    blocks=calibration.blocked_validation_groups(data.groups)
    train,val=train_side_validation(blocks,eligible)
    def progress(row,seconds):
        if row["epoch"]==1 or row["epoch"]%5==0:
            print(f"{name}: "+json.dumps(row)+f"; {seconds:.1f}s",flush=True)
    development=BurstSetRanker(progress=progress)
    try:
        development.fit(data.features,data.labels,data.groups,data.faces,data.face_mask,
                        bodies=data.bodies,body_mask=data.body_mask,training=train,validation=val)
        epochs=development.selected_epoch
        report={"training_examples":int(train.sum()),"validation_examples":int(val.sum()),
                "outside_eligible_examples":int(((train|val)&~eligible).sum()),"selected_epoch":epochs,
                "criterion":"training_side_average_precision","maximum_epochs":MAX_EPOCHS,
                "history":development.validation_history}
    finally:
        development.close()
    with (output/f"{name}-development.json").open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    owner=BurstSetRanker(epochs=epochs,progress=progress)
    try:
        owner.fit(data.features,data.labels,data.groups,data.faces,data.face_mask,
                  bodies=data.bodies,body_mask=data.body_mask,training=eligible)
    except BaseException:
        owner.close()
        raise
    return owner,report


def fit_fixed_pairwise(data, eligible):
    """Fixed prior baseline, with no epoch/parameter choice on test labels."""
    owner = PairedRetentionRanker()
    try:
        owner.fit(data.features[eligible], data.labels[eligible], data.groups[eligible])
    except BaseException:
        owner.close()
        raise
    return owner, {"training_examples": int(eligible.sum()), "validation_examples": 0,
                   "criterion": "preexisting_fixed_boosted_and_pairwise_parameters",
                   "test_labels_used_for_training": False}


def score_shared(data,indexes,owner,*,with_sets=True):
    started=time.monotonic()
    scores=(owner.predict(data.features[indexes],data.groups[indexes],data.faces[indexes],data.face_mask[indexes],
                          bodies=data.bodies[indexes],body_mask=data.body_mask[indexes]) if with_sets
            else owner.predict(data.features[indexes],data.groups[indexes]))
    seconds=time.monotonic()-started
    fraction=len(indexes)/len(data.results)
    floor=round(data.options.target_keep_count*fraction)
    ceiling=round(data.options.target_keep_max_count*fraction)
    take=max(floor,min(ceiling,round(len(indexes)*owner.positive_fraction)))
    grouped={}
    for index,score in zip(indexes,scores):
        item=data.results[index]
        item.calibration_confidence=float(score)
        # Use the frozen grouping, never IDs re-numbered in a previous fold.
        grouped.setdefault(data.groups[index],[]).append(item)
    engine._assign_tiered_ratings(list(grouped.values()),replace(data.options,
                                  target_keep_count=floor,target_keep_max_count=take))
    kept=np.asarray([data.results[i].suggested_rating>0 for i in indexes])
    outcome=metrics(data.labels[indexes],kept)
    outcome.update(auc=calibration._roc_auc(data.labels[indexes],scores),cached_ranking_seconds=seconds,
                   target_keep_min=floor,target_keep_max=take)
    # Diagnostic only: distinguish observation/ranking weakness from shared
    # technical/duplicate gates. This is not the delivered software result.
    raw=np.zeros(len(indexes),bool)
    raw[np.argsort(-scores,kind="stable")[:int(kept.sum())]]=True
    outcome["ranking_only_without_shared_gates"]=metrics(data.labels[indexes],raw)
    return scores,kept,outcome


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","baseline","features","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--case-id",required=True)
    parser.add_argument("--fold",type=int,choices=range(5),help="explicit partial pilot, never full acceptance")
    parser.add_argument("--cross-case-id")
    parser.add_argument("--cross-baseline",type=Path)
    parser.add_argument("--cross-features",type=Path)
    parser.add_argument("--classifier",choices=("burst-set","boosted-pairwise"),default="burst-set",
                        help="reuse the fixed existing pair classifier for a controlled whole-case comparison")
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding="utf-8"))
    cases=manifest["cases"]
    if len({c["case_id"] for c in cases})!=len(cases):
        parser.error("unique case identities required")
    cases={c["case_id"]:c for c in cases}
    cross=(args.cross_case_id,args.cross_baseline,args.cross_features)
    if ((any(cross) and (not all(cross) or args.fold is not None))
            or args.case_id not in cases or (args.cross_case_id and args.cross_case_id not in cases)
            or args.cross_case_id==args.case_id):
        parser.error("one disjoint cross-wedding case with complete arguments, or purged folds required")
    for case_id in (args.case_id,args.cross_case_id):
        if case_id is None:
            continue
        for folder in ("originals","selected_originals"):
            if args.output.resolve().is_relative_to(Path(cases[case_id][folder]).resolve()):
                parser.error("output must be outside photo folders")
    if args.output.exists():
        parser.error("new non-overwriting experiment output required")
    data=load_case(cases[args.case_id],args.baseline,args.features)
    test=None if not all(cross) else load_case(cases[args.cross_case_id],args.cross_baseline,args.cross_features)
    if test is not None:
        require_comparable_observations(data.options, test.options)
    args.output.mkdir(parents=True)
    started=time.monotonic()
    summaries=[]
    scored_data=test or data
    scores=np.zeros(len(scored_data.results),np.float32)
    kept=np.zeros(len(scores),bool)
    evaluated=np.zeros(len(scores),bool)
    try:
        splits=[(np.ones(len(data.results),bool),np.ones(len(test.results),bool))] if test else list(
            purged_folds(calibration.blocked_validation_groups(data.groups)))
        for number,(eligible,held) in enumerate(splits):
            if args.fold is not None and number!=args.fold:
                continue
            owner,development=(fit_selected(data,eligible,args.output,f"split-{number}") if args.classifier=="burst-set"
                               else fit_fixed_pairwise(data,eligible))
            try:
                indexes=np.flatnonzero(held)
                scores[held],kept[held],outcome=score_shared(scored_data,indexes,owner,
                                                          with_sets=args.classifier=="burst-set")
            finally:
                owner.close()
            evaluated|=held
            outcome.update(split=number,training_examples=int(eligible.sum()),test_examples=int(held.sum()),development=development)
            summaries.append(outcome)
            with (args.output/f"split-{number}.json").open("x",encoding="utf-8") as stream:
                json.dump(outcome,stream,ensure_ascii=False,indent=2)
            print(json.dumps(outcome,ensure_ascii=False),flush=True)
        outcome=metrics(scored_data.labels[evaluated],kept[evaluated])
        outcome["auc"]=calibration._roc_auc(scored_data.labels[evaluated],scores[evaluated])
        complete=bool(evaluated.all())
        outcome["passed_quality_on_evaluated_case"]=complete and outcome["precision"]>=.75 and outcome["recall"]>=.75
        report={"pipeline":PIPELINE_ID if args.classifier=="burst-set" else "fixed-boosted-pairwise-retention-v1",
            "classifier":args.classifier,
            "scope":"whole_untrained_wedding_development_validation" if test else "same_wedding_purged_5_fold",
            "training_case":data.case["case_id"],"evaluated_case":scored_data.case["case_id"],
            "training_baseline_sha256":data.baseline_sha256,"evaluated_baseline_sha256":scored_data.baseline_sha256,
            "training_features_sha256":data.feature_sha256,"evaluated_features_sha256":scored_data.feature_sha256,
            "manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "complete_evaluated_case":complete,"evaluation_seconds":time.monotonic()-started,
            "outcome":outcome,"splits":summaries,"release_approved":False,"active_profile_changed":False,
            "additional_image_decodes":0,"additional_detectors":0,"retouch_accuracy":None,
            "feature_dimensions":data.features.shape[1],"scores_are_calibrated_probabilities":False,
            "photos":[{"file":r.path.name,"evaluated":bool(evaluated[i]),"human_retained":bool(scored_data.labels[i]),
                "ai_rating":r.suggested_rating if evaluated[i] else None,"score":float(scores[i]) if evaluated[i] else None}
                for i,r in enumerate(scored_data.results)]}
        with (args.output/"evaluation.json").open("x",encoding="utf-8") as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2)
        print(json.dumps({k:v for k,v in report.items() if k!="photos"},ensure_ascii=False,indent=2),flush=True)
    except BaseException as exc:
        with (args.output/"failed.json").open("x",encoding="utf-8") as stream:
            json.dump({"complete":False,"error":str(exc),"completed_splits":summaries},stream,ensure_ascii=False,indent=2)
        raise


if __name__=="__main__":
    main()

"""Counterbalanced paired visual probe, isolated from production ratings.

No reference names/labels in inference. Each pair tested in both orders; source
RAW decoding and model initialization occur once. Feedback scoring occurs only
after all model replies, never fed back into generation or prompt selection.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import time

from event_cull.mlx_content_runtime import MlxContentRuntime
from event_cull.pair_observation import inspect_pair, PROMPT, VERSION, order_consistent, gated_hint


def chosen_source(decision,left,right):
    return left if decision=='keep_left' else right if decision=='keep_right' else decision


def score_pair(pair,feedback):
    responses=pair['responses']
    if any(r.get('status')!='ok' for r in responses):
        return {'evaluation':'model_error','order_stable':False,'feedback_agreement':None}
    choices=[chosen_source(r['decision'],r['left_path'],r['right_path']) for r in responses]
    stable=order_consistent(*responses)
    interpretation=feedback['interpretation']
    if interpretation=='equivalent_pair':
        agrees=stable and choices[0]=='equivalent'
    elif interpretation in {'prefer_brighter_usable_subject','prefer_less_occlusion','prefer_complete_people'}:
        agrees=stable and choices[0]==feedback['human_path']
    elif interpretation=='distinct_action_also_usable':
        # Pair alone lacks the already-delivered third frame. Both/review may
        # be useful but is not an exact correct winner against this feedback.
        return {'evaluation':'distinct_action_context_incomplete','order_stable':stable,
                'feedback_agreement':None,'both_requested':stable and choices[0]=='keep_both'}
    else:
        agrees=None
    return {'evaluation':'development_feedback_diagnostic','order_stable':stable,
            'feedback_agreement':agrees,'mapped_choices':choices}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','manifest','review','inventory','feedback','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--input-mode',choices=('separate','contact_sheet'),default='separate')
    parser.add_argument('--controls-only',action='store_true')
    args=parser.parse_args()
    packet_raw=(args.review/'review.json').read_bytes()
    packet=json.loads(packet_raw)
    samples=packet['samples']['same_group_different_frame']
    if args.controls_only:
        missed=packet['samples']['human_group_missed']
        samples=[{'ai_path':missed[2]['human_path'],'human_path':missed[0]['human_path'],'group':'control-pet-vs-makeup'},
                 {'ai_path':missed[2]['human_path'],'human_path':missed[1]['human_path'],'group':'control-pet-vs-hands'}]
    paths=list(dict.fromkeys(s[k] for s in samples for k in ('ai_path','human_path')))
    inventory=json.loads(args.inventory.read_bytes())
    manifest_raw=args.manifest.read_bytes();manifest=json.loads(manifest_raw)
    feedback_raw=args.feedback.read_bytes()  # Kept separate; not parsed until after inference.
    def verify():
        for p in paths:
            path=Path(p)
            if p not in inventory or not path.is_file() or [path.stat().st_size,path.stat().st_mtime_ns]!=inventory[p]:
                raise ValueError('Changed or unavailable pair original')
    verify()
    protected={Path(p).parent.resolve() for p in inventory}|{args.review.resolve(),args.model.resolve(),args.feedback.parent.resolve()}
    if args.output.exists() or any(args.output.resolve().is_relative_to(p) for p in protected):
        raise ValueError('Independent non-overwriting output required')
    args.output.mkdir(parents=True)
    from event_cull.engine import _load_preview
    started=time.monotonic()
    runtime=MlxContentRuntime(args.model,manifest)
    previews={}
    records=[]
    rng=random.Random(42)
    try:
        with (args.output/'progress.jsonl').open('x',encoding='utf-8') as progress:
            for number,sample in enumerate(samples,1):
                original=[sample['ai_path'],sample['human_path']]
                first=original[::-1] if rng.getrandbits(1) else original
                pair={'sample_number':number,'group':sample['group'],'responses':[]}
                for left,right in (first,first[::-1]):
                    for path in (left,right):
                        if path not in previews:
                            previews[path]=_load_preview(Path(path))
                    began=time.monotonic()
                    try:
                        response=inspect_pair(previews[left],previews[right],runtime.chat,input_mode=args.input_mode)
                    except Exception as exc:
                        response={'status':'error','error':f'{type(exc).__name__}: {exc}',
                                  'automatic_selection_allowed':False}
                    row={**response,'left_path':left,'right_path':right,'seconds':time.monotonic()-began,
                         'raw_reply':runtime.last_reply}
                    pair['responses'].append(row)
                    progress.write(json.dumps({'sample_number':number,**row},ensure_ascii=False)+'\n');progress.flush()
                    runtime.clear_cache()
                records.append(pair)
                print(f'Pair {number}/{len(samples)}: {[r.get("decision","error") for r in pair["responses"]]}',flush=True)
    finally:
        runtime.close()
    # Only now read photographer interpretations for diagnostic scoring.
    feedback=json.loads(feedback_raw)
    if feedback['baseline_sha256']!=packet['baseline_sha256']:
        raise ValueError('Feedback identity baseline mismatch')
    by_number={r['number']:r for r in feedback['records']}
    for pair,sample in zip(records,samples):
        if args.controls_only:
            pair['diagnostic']={'evaluation':'visually_different_control','feedback_agreement':None,
                                'order_stable':order_consistent(*pair['responses']),
                                'wrong_equivalence':any(r.get('relation')=='equivalent' for r in pair['responses'])}
            continue
        note=by_number[pair['sample_number']]
        if (note['ai_path'],note['human_path'])!=(sample['ai_path'],sample['human_path']):
            raise ValueError('Feedback sample mismatch')
        pair['diagnostic']=score_pair(pair,note)
        pair['safe_hint']=gated_hint(*pair['responses'])
    verify()
    if args.feedback.read_bytes()!=feedback_raw or (args.review/'review.json').read_bytes()!=packet_raw or args.manifest.read_bytes()!=manifest_raw:
        raise ValueError('Frozen input changed')
    summary={'at':datetime.now(timezone.utc).isoformat(),'pipeline':VERSION,'prompt':PROMPT,
             'repo':manifest['repo'],'revision':manifest['revision'],'seed':42,'runs_per_pair':2,
             'input_mode':args.input_mode,'controls_only':args.controls_only,
             'unique_decodes':len(previews),'model_load_seconds':runtime.load_seconds,
             'elapsed_seconds':time.monotonic()-started,'pairs':records,
             'order_stable_pairs':sum(p['diagnostic']['order_stable'] for p in records),
             'strict_feedback_agreements':sum(p['diagnostic']['feedback_agreement'] is True for p in records),
             'strict_feedback_scored_pairs':sum(p['diagnostic']['feedback_agreement'] is not None for p in records),
             'review_sha256':hashlib.sha256(packet_raw).hexdigest(),'feedback_sha256':hashlib.sha256(feedback_raw).hexdigest(),
             'activated':False,'labels_in_prompt':False,'source_modified':False,
             'scope':'8 reviewed development pairs not independent wedding or full-culling accuracy'}
    with (args.output/'summary.json').open('x',encoding='utf-8') as stream:
        json.dump(summary,stream,ensure_ascii=False,indent=2)


if __name__=='__main__':
    main()

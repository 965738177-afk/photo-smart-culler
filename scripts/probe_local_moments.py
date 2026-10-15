"""Pinned local MLX Qwen visual observations on 32 development RAWs.

No answer labels in prompts, no HTTP photo endpoint, no remote Python loading,
no rating writes. Uses the shared preview/encoding service and one model owner.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

from event_cull.moment_observation import MomentObservationService, VERSION, PROMPT, SCHEMA
from event_cull.mlx_content_runtime import MlxContentRuntime


def parse_reply(text):
    text=text.strip()
    if text.startswith('```json') and text.endswith('```'):
        text=text[7:-3].strip()
    return json.loads(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','manifest','review','inventory','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--limit',type=int,default=32)
    parser.add_argument('--multiscale',action='store_true')
    parser.add_argument('--sample-numbers',type=int,nargs='+',help='Explicit development diagnostic groups, not a random/holdout sample')
    args=parser.parse_args()
    if not 1<=args.limit<=32:
        parser.error('Probe scope must be between 1 and 32 examples')
    raw=(args.review/'review.json').read_bytes()
    packet=json.loads(raw)
    samples=[s for ss in packet['samples'].values() for s in ss]
    if args.sample_numbers:
        if any(n<1 or n>len(samples) for n in args.sample_numbers):
            parser.error('Invalid review sample number')
        samples=[samples[n-1] for n in args.sample_numbers]
    paths=list(dict.fromkeys(s[k] for s in samples for k in ('ai_path','human_path') if s[k]))[:args.limit]
    inventory=json.loads(args.inventory.read_bytes())
    manifest_raw=args.manifest.read_bytes()
    manifest=json.loads(manifest_raw)
    def verify():
        for p in paths:
            f=Path(p)
            if p not in inventory or not f.is_file() or [f.stat().st_size,f.stat().st_mtime_ns]!=inventory[p]:
                raise ValueError('Reviewed source changed or missing')
    verify()
    protected={Path(p).parent.resolve() for p in inventory}|{args.review.resolve(),args.model.resolve(),args.manifest.parent.resolve()}
    if args.output.exists() or any(args.output.resolve().is_relative_to(p) for p in protected):
        raise ValueError('Independent non-overwriting output required')
    args.output.mkdir(parents=True)
    from event_cull.engine import _load_preview
    import importlib.metadata
    started=time.monotonic()
    runtime=MlxContentRuntime(args.model,manifest)
    load_seconds=runtime.load_seconds
    responses=[]
    def chat(encoded,prompt):
        try:
            return runtime.chat([encoded],prompt,SCHEMA,multiscale=args.multiscale)
        finally:
            if runtime.last_reply is not None:
                responses.append(runtime.last_reply)
    service=MomentObservationService(chat,maximum_side=1792 if args.multiscale else 896)
    observations=[]
    with (args.output/'progress.jsonl').open('x',encoding='utf-8') as progress:
        for index,p in enumerate(paths,1):
            began=time.monotonic()
            before=len(responses)
            try:
                preview=_load_preview(Path(p))
                result=service.inspect(preview)
            except Exception as exc:
                result={'status':'error','error':f'{type(exc).__name__}: {exc}','automatic_selection_allowed':False}
            row={'path':p,**result,'seconds':time.monotonic()-began,
                 'raw_reply':responses[-1] if len(responses)>before else None}
            observations.append(row)
            progress.write(json.dumps(row,ensure_ascii=False)+'\n');progress.flush()
            print(f"Local moment {index}/{len(paths)} {result.get('status')} {result.get('action','unknown')}",flush=True)
            runtime.clear_cache()
    runtime.close()
    verify()
    if (args.review/'review.json').read_bytes()!=raw or args.manifest.read_bytes()!=manifest_raw:
        raise ValueError('Frozen review or model manifest changed')
    result={'at':datetime.now(timezone.utc).isoformat(),'pipeline':VERSION,'prompt':PROMPT,
            'repo':manifest['repo'],'revision':manifest['revision'],'model_manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),
            'review_sha256':hashlib.sha256(raw).hexdigest(),'model_load_seconds':load_seconds,
            'elapsed_seconds':time.monotonic()-started,'observations':observations,
            'scope':'reviewed_development_samples_not_full_wedding_or_holdout',
            'multiscale':args.multiscale,'selected_development_sample_numbers':args.sample_numbers,
            'versions':{n:importlib.metadata.version(n) for n in ('mlx','mlx-vlm','transformers')},
            'activated':False,'photos_uploaded':False,'labels_in_model_prompt':False,
            'caution':'Model statements are unvalidated observations, not reliable identities, actions or quality acceptance.'}
    with (args.output/'observations.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)


if __name__=='__main__':
    main()

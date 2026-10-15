"""32-reviewed-photo local person probe using shared previews and person worker.

InsightFace embeddings here are local research only. No production integration,
photo uploads, family-role guesses, or full-wedding accuracy claims.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from event_cull.person_observation_cues import pose_cues, observed_face_set_match


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('review','inventory','face-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    raw=(args.review/'review.json').read_bytes()
    packet=json.loads(raw)
    samples=[s for ss in packet['samples'].values() for s in ss]
    paths=list(dict.fromkeys(s[k] for s in samples for k in ('ai_path','human_path') if s[k]))
    inventory=json.loads(args.inventory.read_bytes())
    def verify():
        for p in paths:
            f=Path(p)
            if p not in inventory or not f.is_file() or [f.stat().st_size,f.stat().st_mtime_ns]!=inventory[p]:
                raise ValueError('Reviewed originals unavailable or changed')
    verify()
    model_dir=args.face_root/'models/buffalo_l'
    for name in ('det_10g.onnx','w600k_r50.onnx'):
        if not (model_dir/name).is_file():
            raise ValueError('Existing local face weights required; do not auto-download')
    roots={Path(p).parent.resolve() for p in inventory}|{args.review.resolve(),args.face_root.resolve()}
    if args.output.exists() or any(args.output.resolve().is_relative_to(p) for p in roots):
        raise ValueError('Use a new independent output directory')
    args.output.mkdir(parents=True)
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
    from event_cull.engine import _load_preview
    from src.core.image_features.human_process import HumanProcessService
    # Exclude age/gender/other optional modules: only the scoped face tasks run.
    face=FaceAnalysis(name='buffalo_l',root=str(args.face_root),allowed_modules=['detection','recognition'],
                      providers=['CPUExecutionProvider'])
    face.prepare(ctx_id=-1,det_size=(640,640))
    people=HumanProcessService(recover_faces=True,timeout=30)
    observations={};errors=[];started=time.monotonic()
    try:
        for index,p in enumerate(paths,1):
            image=_load_preview(Path(p))
            native=people.inspect(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            h,w=image.shape[:2]
            records=[]
            for found in face.get(image):
                x0,y0,x1,y1=map(float,found.bbox)
                # Conservative quality eligibility; not a calibrated false-match guarantee.
                if float(found.det_score)<.8 or min(x1-x0,y1-y0)<40:
                    continue
                embedding=np.asarray(found.embedding,float)
                records.append({'bbox':[x0/w,y0/h,x1/w,y1/h],
                                'detection_score':float(found.det_score),'embedding':embedding.tolist()})
            if native.get('status')=='ok':
                cues=pose_cues(native,image_size=(w,h))
            else:
                cues={'visibility_verdict':'unknown','action_verdict':'unknown'}
                errors.append({'path':p,'person_error':native.get('error','unknown')})
            observations[p]={'native_person':native,'eligible_faces':records,'cues':cues}
            print(f'Local face/pose evidence {index}/{len(paths)}',flush=True)
    finally:
        people.close()
    comparisons=[{'group':s['group'],'left':s['ai_path'],'right':s['human_path'],
                  'observed_face_match':observed_face_set_match(observations[s['ai_path']]['eligible_faces'],
                                                               observations[s['human_path']]['eligible_faces'])}
                 for s in samples if s['ai_path'] and s['human_path']]
    verify()
    if (args.review/'review.json').read_bytes()!=raw:
        raise ValueError('Review changed')
    result={'at':datetime.now(timezone.utc).isoformat(),'review_sha256':hashlib.sha256(raw).hexdigest(),
            'scope':'32 reviewed development RAWs not whole-wedding validation',
            'unique_decodes':len(paths),'errors':errors,'observations':observations,'comparisons':comparisons,
            'elapsed_seconds':time.monotonic()-started,'activated':False,
            'face_weights_license':'non-commercial research only, not included in paid app',
            'automatic_story_policy_observations':0,
            'caution':'Geometric near-face cues do not prove occlusion/drinking/kissing. Observed matching faces do not prove the complete people combination.'}
    with (args.output/'observations.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)


if __name__=='__main__':
    main()

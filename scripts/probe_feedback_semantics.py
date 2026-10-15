"""Local CLIP category probe on reviewed examples, not a culling benchmark.

Uses previously installed RN50 weights and shared RAW preview loader. Category
scores are NOT calibrated confidence or evidence of identity/completed actions.
No new model download, photo upload, permanent cache or production changes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

PROMPTS = {
    'makeup': ['a wedding photo of a bride having makeup applied in front of a mirror',
               'a photo of a bride getting her hair and makeup done'],
    'detail': ['a close up wedding photo of hands and small objects',
               'a photo of wedding jewelry decorations or gifts without faces'],
    'pet': ['a photo of a pet dog in a house', 'a wedding photo of a dog'],
    'tea': ['a Chinese wedding photo of people offering tea to seated relatives',
            'a Chinese wedding tea ceremony with a person drinking from a cup'],
    'gift': ['a Chinese wedding photo of people handing over gifts or red envelopes',
             'a wedding photo of someone giving jewelry to the bride'],
    'kiss': ['a wedding photo of the bride and groom kissing', 'a photo of a couple kissing each other'],
    'portrait': ['a posed portrait photo of a bride in a red wedding dress',
                 'a formal wedding group portrait of people posing for the camera'],
    'interaction': ['a candid wedding photo of people smiling and interacting',
                    'a wedding photo of guests walking or carrying things'],
    'other': ['a wedding photo of a room or venue', 'a photo of another wedding scene'],
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    packet_raw=(args.review/'review.json').read_bytes()
    packet=json.loads(packet_raw)
    paths=list(dict.fromkeys(s[k] for rows in packet['samples'].values() for s in rows
                            for k in ('ai_path','human_path') if s[k]))
    inventory=json.loads(args.inventory.read_bytes())
    def verify():
        for p in paths:
            path=Path(p)
            if p not in inventory or not path.is_file() or [path.stat().st_size,path.stat().st_mtime_ns]!=inventory[p]:
                raise ValueError('Unavailable/changed reviewed original: '+p)
    verify()
    if not args.weights.is_file():
        raise ValueError('Existing CLIP weights required; no automatic downloads')
    protected={Path(p).parent.resolve() for p in inventory} | {args.review.resolve(),args.weights.parent.resolve(),args.inventory.parent.resolve()}
    if args.output.exists() or any(args.output.resolve().is_relative_to(p) for p in protected):
        raise ValueError('Use a new output directory outside originals')
    args.output.mkdir(parents=True)
    import torch
    import clip
    from PIL import Image
    import cv2
    from event_cull.engine import _load_preview
    torch.set_num_threads(2)
    model, preprocess=clip.load(str(args.weights),device='cpu',jit=False)
    model.eval()
    names=list(PROMPTS)
    started=time.monotonic()
    with torch.inference_mode():
        text=model.encode_text(clip.tokenize([p for prompts in PROMPTS.values() for p in prompts])).float()
        text=text/text.norm(dim=-1,keepdim=True)
        text=text.reshape(len(names),2,-1).mean(dim=1)
        text=text/text.norm(dim=-1,keepdim=True)
        observations=[]
        for index,p in enumerate(paths,1):
            rgb=Image.fromarray(cv2.cvtColor(_load_preview(Path(p)),cv2.COLOR_BGR2RGB))
            feature=model.encode_image(preprocess(rgb).unsqueeze(0)).float()
            feature=feature/feature.norm(dim=-1,keepdim=True)
            scores=(feature@text.T)[0]
            values=scores.cpu().tolist()
            if not torch.isfinite(scores).all():
                raise ValueError('Nonfinite semantic model scores')
            order=sorted(range(len(names)),key=lambda i:values[i],reverse=True)
            observations.append({'path':p,'proposed_category':names[order[0]],
                                 'cosine_scores':dict(zip(names,values)),
                                 'top_two_margin':values[order[0]]-values[order[1]],
                                 'scene_segment':None,'participants':None,'action_verified':False,
                                 'eligible_for_candidate_automatic_policy':False})
            print(f'CLIP local probe {index}/{len(paths)}',flush=True)
    verify()
    if (args.review/'review.json').read_bytes()!=packet_raw:
        raise ValueError('Review identities changed')
    result={'at':datetime.now(timezone.utc).isoformat(),
            'review_sha256':hashlib.sha256(packet_raw).hexdigest(),
            'weights_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            'scope':'32 reviewed development examples not a blind/full-wedding culling test',
            'prompts':PROMPTS,'observations':observations,'unique_decodes':len(paths),
            'seconds_after_model_load':time.monotonic()-started,'activated':False,
            'caution':'CLIP categories do not establish story segments, identities, visibility or subjective expression. Scores are not probabilities.'}
    with (args.output/'observations.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)


if __name__=='__main__':
    main()

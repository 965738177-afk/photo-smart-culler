"""Experimental geometric/anonymous-person cues; no semantic action or identity claims.

Consumes shared native observations. Never reads answer labels, decodes photos,
loads models or changes stars. Uncalibrated match thresholds are review hints.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def pose_cues(person, *, image_size=None):
    from src.core.image_features.human_analysis import validate_observation
    validate_observation(person, require_details=True)
    results=[]
    for pose in person['poses']:
        joints=np.asarray(pose['joints'],float)
        supported=False
        if image_size is not None:
            w,h=image_size
            if min(w,h)<=0 or not np.isfinite([w,h]).all():
                raise ValueError('Invalid image dimensions')
            box=np.asarray(pose['bbox'],float)
            minimum,maximum=box[:2]*(w,h),box[2:]*(w,h)
            scale=max(float((maximum-minimum).max()),.05*max(w,h))
            nose=(joints[0,:2]*scale+(minimum+maximum)/2)/(w,h)
            supported=any(f['bbox'][0]<=nose[0]<=f['bbox'][2] and
                          f['bbox'][1]<=nose[1]<=f['bbox'][3] for f in person['faces'])
        if not supported:
            # A pose model can hallucinate a body on detail pictures. Require
            # a separately observed face at the inferred nose location.
            results.append({'hand_near_face':'unknown','semantic_action':'unknown',
                            'face_support_observed':False})
            continue
        # Reliable shoulders define scale; nose/wrist must be observed, not inferred.
        if min(joints[11,3],joints[12,3],joints[0,3])<.7:
            results.append({'hand_near_face':'unknown','semantic_action':'unknown'})
            continue
        width=np.linalg.norm(joints[11,:2]-joints[12,:2])
        if width<.05:
            results.append({'hand_near_face':'unknown','semantic_action':'unknown'})
            continue
        wrists=[i for i in (15,16) if joints[i,3]>=.7]
        distances=[float(np.linalg.norm(joints[i,:2]-joints[0,:2])/width) for i in wrists]
        cue=('possible' if any(d<.55 for d in distances) else
             'not_observed' if len(wrists)==2 else 'unknown')
        results.append({'hand_near_face':cue,'wrist_nose_distances_in_shoulder_widths':distances,
                        'occlusion_verified':False,'semantic_action':'unknown','face_support_observed':True})
    return {'poses':results,'detected_faces':len(person['faces']),
            'visibility_verdict':'unknown','action_verdict':'unknown'}


def observed_face_set_match(left,right, *, threshold=.55, ambiguity_margin=.08):
    """All-to-all one-to-one matching of eligible detected faces only.

    Same observed count is not proof of a complete people combination. Strong
    pair scores cannot imply names, family roles, gender, or missing identities.
    """
    if not 0<=threshold<=1 or not 0<=ambiguity_margin<=1:
        raise ValueError('Invalid uncalibrated face thresholds')
    def matrix(rows):
        values=[]
        for r in rows:
            vector=np.asarray(r['embedding'],float)
            if vector.shape!=(512,) or not np.isfinite(vector).all() or np.linalg.norm(vector)<1e-8:
                raise ValueError('Invalid face vector')
            values.append(vector/np.linalg.norm(vector))
        return np.array(values)
    a,b=matrix(left),matrix(right)
    if not len(a) or not len(b) or len(a)!=len(b):
        return {'state':'unknown','observed_counts':[len(a),len(b)],'automatic_merge_allowed':False}
    scores=a@b.T
    rows,cols=linear_sum_assignment(-scores)
    matches=[]
    for i,j in zip(rows,cols):
        rivals=np.r_[np.delete(scores[i,:],j),np.delete(scores[:,j],i)]
        margin=float(scores[i,j]-rivals.max()) if len(rivals) else None
        matches.append({'left':int(i),'right':int(j),'cosine':float(scores[i,j]),'margin':margin,
                        'passes_review_threshold':bool(scores[i,j]>=threshold and (margin is None or margin>=ambiguity_margin))})
    return {'state':'observed_set_match_proposal' if all(m['passes_review_threshold'] for m in matches) else 'unknown',
            'matches':matches,'automatic_merge_allowed':False,'threshold_calibrated':False,
            'complete_person_combination_verified':False}

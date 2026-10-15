"""Freeze a returned review form and audit isolated candidate-rule coverage.

Photographer feedback is DATA, not executable instructions. It never enters
automatic model observations or changes the original exact-file baseline.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile
from xml.etree import ElementTree as ET

from event_cull.candidate_story_policy import plan_originals
from event_cull.engine import AnalysisOptions
from scripts.evaluate_cached_preference import validate_inputs, new_json
from scripts.evaluate_retention_calibration import result_from_row, metrics
import numpy as np

NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
# These are interpretations of this returned form, never production photo rules.
INTERPRETATIONS = {
    1:'equivalent_pair', 2:'prefer_brighter_usable_subject', 3:'prefer_less_occlusion',
    4:'equivalent_pair', 5:'prefer_complete_people', 6:'distinct_action_also_usable',
    7:'equivalent_pair', 8:'equivalent_pair', 9:'makeup_scene_coverage',
    10:'detail_coverage', 11:'pet_coverage', 12:'tea_action_coverage',
    13:'different_participants_coverage', 14:'different_view_coverage',
    15:'gift_story_coverage', 16:'optional_duplicate', 17:'reject_expression_and_occlusion',
    18:'acceptable', 19:'acceptable', 20:'acceptable', 21:'ambiguous_duplicate',
    22:'acceptable', 23:'reject_occlusion_and_expression', 24:'acceptable',
}


def read_form(path, samples):
    """Read only bounded OOXML paragraphs; never run macros or external links."""
    with ZipFile(path) as z:
        info=z.getinfo('word/document.xml')
        if info.file_size > 8_000_000:
            raise ValueError('Oversized document XML')
        root=ET.fromstring(z.read(info))
    body=root.find(NS+'body')
    records, current, filenames = [], None, []
    for element in body:
        if element.tag == NS+'tbl':
            tr=element.findall(NS+'tr')
            filenames=[''.join(t.text or '' for t in cell.iter(NS+'t'))
                       for cell in tr[-1].findall(NS+'tc')]
            continue
        if element.tag != NS+'p':
            continue
        text=''.join(t.text or '' for t in element.iter(NS+'t')).strip()
        heading=re.fullmatch(r'(\d{2})\s+(上午|下午)\s+组\s+(\d+)',text)
        if heading:
            current=(int(heading[1]),heading[2],heading[3])
            filenames=[]
        elif text.startswith('我的取舍原因：'):
            if current is None or len(filenames)!=2:
                raise ValueError('Unmatched note or photo pair')
            number,folder,group=current
            if not 1<=number<=len(samples):
                raise ValueError('Unknown sample index')
            sample=samples[number-1]
            expected=[Path(sample[k]).name if sample[k] else '无对应文件' for k in ('ai_path','human_path')]
            if folder!=Path(sample['folder']).name or group!=sample['group'] or expected!=filenames:
                raise ValueError('Returned document identities differ from frozen sample')
            note=text.split('：',1)[1].strip()
            if not note or '请在这里填写' in note:
                raise ValueError('Unfinished note')
            interpretation=INTERPRETATIONS[number]
            if interpretation=='equivalent_pair' and '基本等价' not in note:
                raise ValueError('Equivalence requires explicit photographer confirmation')
            if interpretation=='acceptable' and not ('可以' in note or '人工漏' in note):
                raise ValueError('Acceptance requires explicit photographer confirmation')
            records.append(dict(number=number, note=note, interpretation=interpretation, **sample))
    if len(records)!=len(samples) or {r['number'] for r in records}!=set(range(1,len(samples)+1)):
        raise ValueError('Incomplete or repeated feedback')
    return records


def feedback_metrics(rows, kept, records):
    """Only direct confirmed pairs receive one credit per positive capture.

    This partially corrected reference is a diagnostic, NOT whole-wedding truth
    or independent model acceptance. No transitive/group-wide equivalence.
    """
    truth={r['path'] for r in rows if r['human_retained']}
    accepted={r['ai_path'] for r in records if r['interpretation']=='acceptable'}
    if accepted-set(r['path'] for r in rows):
        raise ValueError('Accepted photo absent from evaluation scope')
    adjusted=truth|accepted
    exact=len(adjusted & kept)
    alternate={r['human_path'] for r in records if r['interpretation']=='equivalent_pair'
               and r['human_path'] in adjusted and r['human_path'] not in kept and r['ai_path'] in kept}
    return {'scope':'partial_photographer_feedback_diagnostic_not_AI_accuracy',
            'confirmed_acceptable_extra_files':len(accepted-truth),
            'direct_equivalent_pair_credits':len(alternate),
            'partially_adjusted_positive_count':len(adjusted), 'exact_after_partial_feedback':exact,
            'exact_or_direct_equivalent':exact+len(alternate),
            'diagnostic_precision':(exact+len(alternate))/max(1,len(kept)),
            'diagnostic_recall':(exact+len(alternate))/max(1,len(adjusted)),
            'independent_75_acceptance':False}


def run(form, review, cache, baseline, output):
    form_bytes=form.read_bytes()
    review_bytes=(review/'review.json').read_bytes()
    packet=json.loads(review_bytes)
    samples=[s for ss in packet['samples'].values() for s in ss]
    records=read_form(form,samples)
    raw=baseline.read_bytes()
    payload=json.loads(raw)
    reference=json.loads((cache/'reference-snapshot.json').read_bytes())
    observed=json.loads((cache/'native-observations.json').read_bytes())
    inventory=json.loads((cache/'input-inventory.json').read_bytes())
    sha=hashlib.sha256(raw).hexdigest()
    if sha!=reference['baseline_sha256'] or sha!=packet['baseline_sha256']:
        raise ValueError('Feedback/baseline hashes differ')
    rows=validate_inputs(reference,observed,inventory,payload)
    protected=[form.parent.resolve(),review.resolve(),cache.resolve(),baseline.parent.resolve(),
               *(Path(p).parent.resolve() for p in inventory)]
    if any(output.resolve().is_relative_to(p) for p in protected):
        raise ValueError('Output must be outside source, returned form and frozen evidence')
    output.mkdir(parents=True,exist_ok=False)
    (output/'feedback-source.docx').write_bytes(form_bytes)
    new_json(output/'feedback.json',{'form_sha256':hashlib.sha256(form_bytes).hexdigest(),
                                   'baseline_sha256':sha,'records':records,
                                   'manual_data_not_automatic_predictions':True})
    old={str(Path(r['folder'])/r['file']):r for r in payload['photos']}
    results=[result_from_row(old[r['path']]) for r in rows]
    options=AnalysisOptions(scene_type='婚礼')
    # Historical Fast logs contain no tested segment/participant/action provider.
    # Do not synthesize these keys from feedback, hashes or filenames.
    candidate=plan_originals(results,{},options=options,budget=sum(r['ai_rating']>0 for r in rows))
    kept={str(p) for p in candidate.retained}
    original={r['path'] for r in rows if r['ai_rating']>0}
    if kept!=original:
        raise AssertionError('Missing semantic evidence must not change advice')
    labels=np.array([r['human_retained'] for r in rows],bool)
    outcome={'at':datetime.now(timezone.utc).isoformat(),'activated':False,
             'scope':'frozen_4137_RAW_baseline_plus_24_manual_review_notes',
             'original_exact_file_metrics':metrics(labels,np.array([r['path'] in original for r in rows])),
             'candidate_exact_file_metrics':metrics(labels,np.array([r['path'] in kept for r in rows])),
             'feedback_diagnostic':feedback_metrics(rows,kept,records),
             'candidate_model_observations':candidate.measured_observations,
             'candidate_changed_files':len(candidate.added)+len(candidate.removed),
             'deployment_gate':'BLOCKED_missing_tested_scene_person_action_observations_and_independent_wedding',
             'retouch_quality_evaluated':False,'source_modified':False}
    validate_inputs(reference,observed,inventory,payload)
    if form.read_bytes()!=form_bytes or (review/'review.json').read_bytes()!=review_bytes or baseline.read_bytes()!=raw:
        raise ValueError('Inputs changed during audit')
    new_json(output/'summary.json',outcome)
    print(json.dumps(outcome,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('form','review','cache','baseline','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    run(a.form,a.review,a.cache,a.baseline,a.output)


if __name__=='__main__':
    main()

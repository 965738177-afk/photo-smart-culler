import json
import hashlib
import numpy as np
import pytest

from event_cull.pair_observation import inspect_pair,validate,compose_pair_preview,gated_hint
from event_cull.mlx_content_runtime import validate_model_files
from scripts.probe_pair_preferences import score_pair


def value(decision='equivalent',relation='equivalent'):
    return {'left_content':'人物站立','right_content':'人物站立','decision':decision,'relation':relation,'visibility':'similar',
            'exposure':'similar','observations':['两张画面基本相同。']}


def test_pair_encodes_both_once_and_never_applies_a_manual_decision():
    calls=[]
    def chat(images,prompt,schema):
        calls.append(images)
        assert '未标注人工答案' in prompt
        return value()
    result=inspect_pair(np.zeros((20,20,3),np.uint8),np.zeros((20,20,3),np.uint8),chat)
    assert len(calls)==1 and len(calls[0])==2
    assert not result['automatic_selection_allowed']


def test_contact_sheet_retains_both_images_in_one_model_input():
    left=np.zeros((50,25,3),np.uint8);left[:,:,2]=255
    right=np.zeros((25,50,3),np.uint8);right[:,:,0]=255
    sheet=compose_pair_preview(left,right)
    assert sheet.shape==(768,1536,3)
    assert sheet[384,384].tolist()==[0,0,255]
    assert sheet[384,1152].tolist()==[255,0,0]
    def chat(images,prompt,schema):
        assert len(images)==1 and '左右拼图' in prompt
        return value()
    assert inspect_pair(left,right,chat,input_mode='contact_sheet')['status']=='ok'


def test_pair_input_mode_must_be_explicit():
    with pytest.raises(ValueError,match='input mode'):
        inspect_pair(np.zeros((10,10,3),np.uint8),np.zeros((10,10,3),np.uint8),None,input_mode='unknown')


def test_either_requires_equivalence_and_extra_fields_rejected():
    with pytest.raises(ValueError,match='Equivalence'):
        validate(value(relation='unknown'))
    with pytest.raises(ValueError,match='Incomplete'):
        validate(dict(value(),rating=2))


def pair(a='keep_right',b='keep_left'):
    return {'responses':[{'status':'ok',**value(a,'not_equivalent'),'left_path':'ai','right_path':'human'},
                         {'status':'ok',**value(b,'not_equivalent'),'left_path':'human','right_path':'ai'}]}


def test_preference_scoring_maps_swapped_positions_back_to_source():
    feedback={'interpretation':'prefer_less_occlusion','human_path':'human'}
    result=score_pair(pair(),feedback)
    assert result['order_stable'] and result['feedback_agreement']
    result=score_pair(pair('keep_right','keep_right'),feedback)
    assert not result['order_stable'] and not result['feedback_agreement']


def test_same_winner_with_contradictory_quality_claims_is_unstable():
    p=pair();p['responses'][0]['visibility']='right_better'
    p['responses'][1]['visibility']='right_better'
    result=score_pair(p,{'interpretation':'prefer_less_occlusion','human_path':'human'})
    assert not result['order_stable'] and not result['feedback_agreement']


def test_safe_hint_cannot_mark_unvalidated_outputs_as_manual_or_automatic_choices():
    p=pair()
    hint=gated_hint(*p['responses'])
    assert hint['action']=='keep_right' and not hint['automatic_selection_allowed']
    p['responses'][1]['exposure']='right_better'
    assert gated_hint(*p['responses'])['action']=='manual_review'
    p['responses'][0]['decision']='keep_both'
    p['responses'][0]['relation']='equivalent'
    assert gated_hint(*p['responses'])['action']=='manual_review'


def test_equivalent_is_not_a_strict_winner_and_missing_context_not_scored():
    p=pair();p['responses'][0].update(value());p['responses'][1].update(value())
    assert score_pair(p,{'interpretation':'equivalent_pair'})['feedback_agreement']
    assert score_pair(p,{'interpretation':'distinct_action_also_usable'})['feedback_agreement'] is None


def test_error_not_counted_as_agreement():
    p=pair();p['responses'][0]={'status':'error'}
    assert score_pair(p,{'interpretation':'equivalent_pair'})['feedback_agreement'] is None


def test_pinned_model_checksum_and_path_guard_before_loading(tmp_path):
    path=tmp_path/'config.json';path.write_text('{}')
    record={'file':'config.json','bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    validate_model_files(tmp_path,{'files':[record]})
    path.write_text('[]')
    with pytest.raises(ValueError,match='checksum'):
        validate_model_files(tmp_path,{'files':[record]})
    with pytest.raises(ValueError,match='inventory'):
        validate_model_files(tmp_path,{'files':[dict(record,file='../config.json')]})

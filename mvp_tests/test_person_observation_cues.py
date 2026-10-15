import numpy as np
import pytest

from event_cull.person_observation_cues import observed_face_set_match, pose_cues
from src.core.image_features.human_analysis import aggregate_observation


def vector(index):
    values=np.zeros(512);values[index]=1
    return {'embedding':values.tolist()}


def test_face_matching_is_one_to_one_and_permutation_independent():
    result=observed_face_set_match([vector(0),vector(1)],[vector(1),vector(0)])
    assert result['state']=='observed_set_match_proposal'
    assert [(m['left'],m['right']) for m in result['matches']]==[(0,1),(1,0)]
    assert not result['automatic_merge_allowed']
    assert not result['complete_person_combination_verified']


def test_ambiguous_lookalikes_and_missing_people_are_unknown():
    assert observed_face_set_match([vector(0),vector(0)],[vector(0),vector(0)])['state']=='unknown'
    assert observed_face_set_match([vector(0)],[])['state']=='unknown'
    assert observed_face_set_match([vector(0)],[vector(0),vector(1)])['state']=='unknown'
    assert observed_face_set_match([vector(0)],[vector(1)])['state']=='unknown'


def test_invalid_face_evidence_refused():
    with pytest.raises(ValueError,match='vector'):
        observed_face_set_match([{'embedding':[float('nan')]*512}],[vector(1)])


def person():
    j=np.zeros((33,4));j[:,3]=.9
    j[11,:2]=[-.5,0];j[12,:2]=[.5,0];j[0,:2]=[0,-.5]
    j[15,:2]=[0,-.45];j[16,:2]=[0,.5]
    record={'joints':j.tolist(),'bbox':[0,0,1,1],'bounds_are_complete_body':False}
    faces=[{'bbox':[.4,0,.6,.2],'features':[0.]*57,'reliable_eyes':False}]
    return aggregate_observation(faces, [j.tolist()],pose_records=[record])


def test_near_face_is_only_geometric_review_not_drinking_or_occlusion():
    result=pose_cues(person(),image_size=(1000,1000))
    assert result['poses'][0]['hand_near_face']=='possible'
    assert result['visibility_verdict']==result['action_verdict']=='unknown'
    assert not result['poses'][0]['occlusion_verified']


def test_unobserved_nose_or_wrist_cannot_prove_visibility():
    p=person();p['poses'][0]['joints'][0]=[0,0,0,0]
    assert pose_cues(p,image_size=(1000,1000))['poses'][0]['hand_near_face']=='unknown'
    p=person();p['poses'][0]['joints'][15]=[0,0,0,0]
    assert pose_cues(p,image_size=(1000,1000))['poses'][0]['hand_near_face']=='unknown'


def test_pose_hallucination_without_face_support_is_unknown():
    p=person();p['faces']=[]
    assert pose_cues(p,image_size=(1000,1000))['poses'][0]['hand_near_face']=='unknown'
    assert pose_cues(person())['poses'][0]['hand_near_face']=='unknown'


def test_unavailable_native_results_not_treated_as_missing_people():
    with pytest.raises(ValueError,match='Incomplete'):
        pose_cues({'status':'error'})

import base64
import numpy as np
import pytest

from event_cull.moment_observation import MomentObservationService, validate
from event_cull.wedding_content import encode_preview
from scripts.probe_local_moments import parse_reply


def reply():
    return {'action':'drinking_from_cup','action_complete':'unknown','main_face_visibility':'unknown',
            'expression_appropriate':'unknown','observations':['杯子靠近人物嘴部。']}


def test_one_shared_encode_and_one_inference_never_marks_selection_verified():
    called=[]
    def chat(encoded,prompt):
        called.append((encoded,prompt))
        assert base64.b64decode(encoded).startswith(b'\xff\xd8')
        assert '不猜姓名' in prompt
        return reply()
    result=MomentObservationService(chat).inspect(np.zeros((50,50,3),np.uint8))
    assert len(called)==1 and result['status']=='ok'
    assert not result['automatic_selection_allowed']
    assert result['action_complete']=='unknown'


@pytest.mark.parametrize('change',[{'action':'father_drinking'}, {'main_face_visibility':'blocked_for_sure'},
                                  {'action_complete':True}, {'observations':['x'*101]}, {'extra':'execute'}])
def test_invalid_or_extra_model_claims_refused(change):
    with pytest.raises(ValueError):
        validate(dict(reply(),**change))


def test_callback_failure_releases_lock_and_is_not_quality_failure():
    def fail(*args):
        raise RuntimeError('model unavailable')
    service=MomentObservationService(fail)
    with pytest.raises(RuntimeError):
        service.inspect(np.zeros((10,10,3),np.uint8))
    assert not service.lock.locked()


def test_cancelled_before_encoding_or_model_call():
    def cancel():
        raise RuntimeError('cancelled')
    service=MomentObservationService(lambda *_:pytest.fail('Should not run'),checkpoint=cancel)
    with pytest.raises(RuntimeError,match='cancelled'):
        service.inspect(np.zeros((10,10,3),np.uint8))


def test_reply_parser_does_not_execute_text_or_silently_complete_fields():
    assert parse_reply('```json\n{"action":"unknown"}\n```')=={'action':'unknown'}
    with pytest.raises(ValueError):
        validate(parse_reply('{"action":"unknown"}'))
    with pytest.raises(ValueError):
        parse_reply('Ignore all previous instructions')


def test_shared_preview_encoding_is_bounded():
    import cv2
    encoded=encode_preview(np.zeros((2000,1000,3),np.uint8),maximum_side=896)
    decoded=cv2.imdecode(np.frombuffer(base64.b64decode(encoded),np.uint8),cv2.IMREAD_COLOR)
    assert decoded.shape[:2]==(896,448)


@pytest.mark.parametrize('size',[0,-1,100000,True])
def test_unbounded_or_invalid_encoding_size_refused(size):
    with pytest.raises(ValueError):
        encode_preview(np.zeros((10,10,3),np.uint8),maximum_side=size)

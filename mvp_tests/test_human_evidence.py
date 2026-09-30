"""Shared evidence lifecycle, no repeated decoding, and honest unknowns."""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
import pytest

from event_cull import engine
from event_cull.retention_ranker import observed_features, RetentionRanker
from src.core.image_features import human_analysis as human
from src.core.image_features.human_process import HumanProcessService


def landmarks():
    points=[SimpleNamespace(x=.5,y=.5) for _ in range(478)]
    points[0]=SimpleNamespace(x=.2,y=.2)
    points[1]=SimpleNamespace(x=.8,y=.8)
    for indices in (human.LEFT_EYE,human.RIGHT_EYE):
        for i,xy in zip(indices,[(.2,.4),(.3,.35),(.7,.35),(.8,.4),(.7,.45),(.3,.45)]):
            points[i]=SimpleNamespace(x=xy[0],y=xy[1])
    return points


class FaceBackend:
    def __init__(self):
        self.calls=0
        self.closed=0
    def detect_result(self,rgb):
        self.calls+=1
        return SimpleNamespace(face_landmarks=[landmarks()],facial_transformation_matrixes=[np.eye(4)],
            face_blendshapes=[[SimpleNamespace(category_name=k,score=.1) for k in human.BLENDSHAPES]])
    def close(self):
        self.closed+=1


class BodyBackend:
    def __init__(self):
        self.calls=0
        self.closed=0
    def detect_result(self,rgb):
        self.calls+=1
        return SimpleNamespace(pose_landmarks=[])
    def close(self):
        self.closed+=1


def test_one_shared_native_model_per_run_even_with_concurrent_workers():
    face,body=FaceBackend(),BodyBackend()
    created=[]
    service=human.HumanAnalysisService(
        face_factory=lambda p:created.append("face") or face,
        pose_factory=lambda p:created.append("body") or body,resolver=lambda n:Path(n))
    image=np.random.default_rng(0).integers(0,255,(120,160,3),dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=6) as workers:
        observations=list(workers.map(lambda _:service.inspect(image),range(20)))
    service.close()
    assert created==["face","body"]
    assert face.calls==body.calls==20 and face.closed==body.closed==1
    assert all(len(x["features"])==human.FEATURE_DIMENSION for x in observations)


def test_no_face_is_unknown_not_closed_eyes_or_bad_action():
    result=human.aggregate_observation([],[])
    assert result["faces"]==[] and result["unknown_eyes"]==0
    assert result["pose_count"]==0 and not any(result["features"])
    assert "action_complete" not in result and "closed_eyes" not in result


def test_tiny_or_soft_face_keeps_eye_state_unknown():
    points=landmarks()
    shapes=[SimpleNamespace(category_name=k,score=.99) for k in human.BLENDSHAPES]
    record=human._face_record(points,shapes,np.eye(4),np.zeros((20,20,3),np.uint8),(0,0,20,20))
    assert not record["reliable_eyes"]
    assert human.aggregate_observation([record],[])["unknown_eyes"]==1


def test_eye_measurement_uses_image_aspect_ratio():
    points=landmarks()
    assert human._ear(points,human.LEFT_EYE,200,100)==pytest.approx(1/12)
    assert human._ear(points,human.LEFT_EYE,100,200)==pytest.approx(1/3)


def test_overlapping_face_crop_does_not_count_one_person_twice():
    face,body=FaceBackend(),BodyBackend()
    service=human.HumanAnalysisService(face_factory=lambda _:face,pose_factory=lambda _:body,resolver=Path)
    result=service.inspect(np.zeros((120,160,3),np.uint8),face_regions=[(32,24,96,72)])
    assert len(result["faces"])==1 and face.calls==1
    service.close()


def test_checkpoint_cancels_before_native_initialization():
    def stop():
        raise engine.AnalysisCancelled()
    service=human.HumanAnalysisService(face_factory=lambda _:pytest.fail("not initialized"),checkpoint=stop)
    with pytest.raises(engine.AnalysisCancelled):
        service.inspect(np.zeros((32,32,3),np.uint8))
    service.close()


def test_analysis_reuses_decoded_preview_and_original_haar_regions(monkeypatch,tmp_path):
    source=tmp_path/"test.jpg"
    image=np.zeros((32,32,3),np.uint8)
    decoded,inspected=[],[]
    monkeypatch.setattr(engine,"_load_preview",lambda p:decoded.append(p) or image)
    regions=[(1,2,10,10)]
    monkeypatch.setattr(engine,"_face_metrics",lambda *a:([],None,0,0,0,None,0,None,engine.EyePolicy("unknown"),regions))
    class Service:
        def inspect(self,rgb,*,face_regions):
            inspected.append((rgb,face_regions))
            return human.aggregate_observation([],[])
    result=engine._analyze(source,[source],engine.AnalysisOptions(),human_model=Service())
    assert decoded==[source] and len(inspected)==1 and inspected[0][1] is regions
    assert result.person_evidence["status"]=="ok"


def test_missing_person_cache_retries_and_errors_do_not_poison_cache(monkeypatch,tmp_path):
    source=tmp_path/"test.jpg"
    source.write_bytes(b"signature fixture")
    monkeypatch.setattr(engine,"_human_model_signature",lambda:"fixture")
    options=engine.AnalysisOptions(human_analysis_enabled=True)
    signature=engine._cache_signature(source,options)
    result=engine.PhotoResult(source,score=90,sharpness=100,highlight_clip=0,shadow_clip=0,perceptual_hash=12,
                              person_evidence=human.aggregate_observation([],[]))
    record=engine._cache_record(result,signature)
    assert engine._result_from_cache(source,[source],record,signature).person_evidence==result.person_evidence
    record["person_evidence"]["features"][0]=float("nan")
    assert engine._result_from_cache(source,[source],record,signature) is None
    monkeypatch.setattr(engine,"_analyze",lambda *a,**kw:engine.PhotoResult(source,person_evidence={"status":"error"}))
    _,_,new=engine._analyze_cached(source,[source],options,{},human_model=object())
    assert new is None


def test_relative_inference_features_do_not_use_names_or_delivery_labels():
    records=[engine.PhotoResult(Path(f"/photos/{i}.jpg"),group_id=1,
        person_evidence=human.aggregate_observation([],[]),sharpness=100+i) for i in range(4)]
    visual=np.arange(12).reshape(4,3)
    before=observed_features(records,visual)
    for record in records:
        record.path=Path("/renamed/arbitrary-name.jpg")
        record.suggested_rating=3
    assert np.array_equal(before,observed_features(records,visual))


def test_bad_or_incomplete_person_results_cannot_be_silently_omitted():
    records=[engine.PhotoResult(Path("test.jpg"))]
    with pytest.raises(ValueError,match="incomplete"):
        observed_features(records,np.zeros((1,3)))


def test_process_cancel_and_timeout_release_child(monkeypatch):
    import queue
    class Sink:
        def write(self,data):return len(data)
        def flush(self):pass
        def close(self):pass
    class Child:
        stdin=Sink()
        stdout=Sink()
        killed=False
        def poll(self):return 0 if self.killed else None
        def terminate(self):self.killed=True
        def wait(self,timeout):return 0
    service=HumanProcessService(timeout=.02)
    child=Child()
    service.process=child
    service._responses=queue.Queue()
    with pytest.raises(TimeoutError):
        service.inspect(np.zeros((32,32,3),np.uint8))
    assert child.killed and service.process is None
    service.close()


def test_native_abort_is_an_error_not_desktop_exit():
    import queue
    from io import BytesIO
    class Child:
        stdin=BytesIO()
        stdout=BytesIO()
        def poll(self):return -6
    service=HumanProcessService()
    service.process=Child()
    service._responses=queue.Queue()
    service._responses.put(None)
    with pytest.raises(RuntimeError,match="desktop protected"):
        service.inspect(np.zeros((32,32,3),np.uint8))
    assert service.process is None
    service.close()


def test_repeated_worker_start_failures_do_not_spawn_per_photo():
    calls=[]
    def fail(*a,**kw):
        calls.append(True)
        raise OSError("missing worker")
    service=HumanProcessService(process_factory=fail)
    for _ in range(20):
        with pytest.raises((OSError,RuntimeError)):
            service.inspect(np.zeros((32,32,3),np.uint8))
    assert len(calls)==2
    service.close()


def test_nonlinear_ranker_can_compare_interacting_features_without_file_identifiers():
    rng=np.random.default_rng(0)
    values=rng.uniform(-1,1,(400,2))
    labels=(values[:,0]*values[:,1]>0).astype(np.int8)
    model=RetentionRanker().fit(values[:300],labels[:300])
    assert np.mean((model.predict(values[300:])>.5)==labels[300:])>.9


def test_ranker_training_cancelled_before_publishing_a_model():
    def stop():
        raise engine.AnalysisCancelled()
    model=RetentionRanker(checkpoint=stop)
    with pytest.raises(engine.AnalysisCancelled):
        model.fit(np.tile([0.,1.],(100,1)),np.tile([0,1],50))
    assert model._model is None


def test_chinese_log_reasons_remain_separate_for_shared_technical_gates():
    from scripts.evaluate_retention_calibration import result_from_row
    result=result_from_row({"folder":"/photos","file":"test.jpg","reasons":"糊片；人物失焦;其他"})
    assert result.reasons==["糊片","人物失焦","其他"]


def test_aligned_metal_inputs_are_compatible_without_changing_normalized_landmarks(monkeypatch):
    from src.core.image_features.landmarker_runtime import TasksLandmarker
    captured=[]
    class Backend:
        def detect(self,image):
            captured.append(image.data.shape)
            return SimpleNamespace(face_landmarks=[[SimpleNamespace(x=.25,y=.5)]])
        def close(self):pass
    class Base(SimpleNamespace):
        Delegate=SimpleNamespace(CPU="cpu",GPU="gpu")
    mp=SimpleNamespace(tasks=SimpleNamespace(BaseOptions=Base,vision=SimpleNamespace(
        RunningMode=SimpleNamespace(IMAGE="image"),FaceLandmarkerOptions=lambda **kw:kw,
        FaceLandmarker=SimpleNamespace(create_from_options=lambda _:Backend()))),
        Image=lambda **kw:SimpleNamespace(**kw),ImageFormat=SimpleNamespace(SRGBA="rgba",SRGB="rgb"))
    backend=TasksLandmarker(Path("fake"),module_loader=lambda _:mp,platform="darwin")
    points=backend.detect_landmarks(np.zeros((37,41,3),np.uint8))
    assert captured==[(40,44,4)]
    assert (points[0][0].x,points[0][0].y)==(.25,.5)


def test_pairwise_work_is_bounded_and_stays_within_bursts():
    from event_cull.retention_ranker import same_group_pairs
    groups=np.repeat(np.arange(10),100)
    left,right=same_group_pairs(groups)
    assert len(left)<=8*len(groups)
    assert np.array_equal(groups[left],groups[right])
    assert (left!=right).all()


def test_pairwise_comparison_never_treats_a_singleton_as_a_quality_winner():
    from event_cull.retention_ranker import PairwiseRetentionRanker
    model=PairwiseRetentionRanker()
    assert np.array_equal(model.compare(np.zeros((3,4)),[1,2,3]),np.full(3,.5))


@pytest.mark.parametrize("strategy",["linear","rbf","neural"])
def test_margin_models_fit_scaling_on_training_only(strategy):
    rng=np.random.default_rng(42)
    features=rng.normal(size=(200,3))
    labels=(features[:,0]>.1).astype(int)
    model=RetentionRanker(strategy=strategy).fit(features,labels)
    centre=model._model.steps[0][1].mean_.copy()
    predicted=model.predict(np.vstack((features[:10],[[10000,10000,10000]])))
    assert np.array_equal(model._model.steps[0][1].mean_,centre)
    assert np.isfinite(predicted).all() and np.mean((model.predict(features)>.5)==labels)>.9


def test_healthy_worker_recycles_without_counting_a_model_failure(monkeypatch):
    import json,queue
    from io import BytesIO
    children=[]
    class Child:
        def __init__(self):
            self.stdin,self.stdout=BytesIO(),BytesIO()
            self.killed=False
        def poll(self):return 0 if self.killed else None
        def terminate(self):self.killed=True
        def wait(self,timeout):return 0
    service=HumanProcessService(recycle_after=2)
    def start():
        if service.process is None:
            service.process=Child()
            children.append(service.process)
            service._responses=queue.Queue()
            service._calls=0
        service._responses.put(json.dumps(human.aggregate_observation([],[])).encode())
    monkeypatch.setattr(service,"_start",start)
    for _ in range(5):service.inspect(np.zeros((32,32,3),np.uint8))
    assert len(children)==3 and all(child.killed for child in children[:2])
    assert service._restarts==0
    service.close()
    assert all(child.killed for child in children)


def test_rating_each_fold_cannot_merge_later_pairwise_training_bursts():
    from scripts.evaluate_person_retention import frozen_group_ids
    from event_cull.retention_ranker import same_group_pairs
    records=[engine.PhotoResult(Path(f"{i}.jpg"),group_id=g) for i,g in enumerate([4,4,8,8,15,15])]
    original=frozen_group_ids(records)
    # Shared _assign_tiered_ratings legitimately renumbers a held-out subset.
    for r in records[:2]:r.group_id=8
    assert original.tolist()==["4","4","8","8","15","15"]
    assert not original.flags.writeable
    left,right=same_group_pairs(original)
    assert np.array_equal(original[left],original[right])
    assert len(left)==6


def test_user_pause_is_not_charged_to_native_worker_timeout(monkeypatch):
    import json,queue
    from io import BytesIO
    from src.core.image_features import human_process
    clock=[0.]
    calls=[]
    def checkpoint():
        calls.append(True)
        if len(calls)==2:clock[0]+=100.
    class Child:
        def __init__(self):
            self.stdin,self.stdout=BytesIO(),BytesIO()
            self.killed=False
        def poll(self):return 0 if self.killed else None
        def terminate(self):self.killed=True
        def wait(self,timeout):return 0
    monkeypatch.setattr(human_process.time,"monotonic",lambda:clock[0])
    service=HumanProcessService(checkpoint=checkpoint,timeout=.02)
    service.process=Child()
    service._responses=queue.Queue()
    service._responses.put(json.dumps(human.aggregate_observation([],[])).encode())
    assert service.inspect(np.zeros((32,32,3),np.uint8))["status"]=="ok"
    assert service._restarts==0
    service.close()

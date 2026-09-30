"""Visible-expression evidence, alignment and bounded shared-preview ownership."""
from pathlib import Path
import numpy as np
import pytest

from src.core.image_features.expression_analysis import (
    ExpressionEvidenceService,similarity_transform,aggregate_expressions,TARGET,FEATURE_DIMENSION)


def test_similarity_alignment_recovers_scale_and_translation():
    source=TARGET*2+np.asarray([18,37])
    transform=similarity_transform(source)
    mapped=np.column_stack((source,np.ones(5)))@transform.T
    assert np.allclose(mapped,TARGET,atol=1e-4)
    with pytest.raises(ValueError,match="Degenerate"):
        similarity_transform(np.ones((5,2)))


def test_unknown_faces_are_not_neutral_or_unusable_photos():
    result=aggregate_expressions([],unknown=3)
    assert len(result["features"])==FEATURE_DIMENSION
    assert not any(result["features"][:-3])
    assert result["features"][-3:]==[0.,3.,0.]
    assert "reject" not in result and "mood" not in result


def test_one_lazy_model_reuses_input_preview_and_returns_finite_probabilities():
    created,crops,tensors=[],[],[]
    class Detector:
        def setInputSize(self,size):self.size=size
        def detect(self,crop):
            crops.append(crop.shape)
            return None,np.asarray([[0,0,crop.shape[1],crop.shape[0],*TARGET.ravel(),.99]])
    class Session:
        def run(self,names,inputs):
            tensors.append(inputs["data"])
            return [np.asarray([[0,0,0,3,1,0,0.]])]
    service=ExpressionEvidenceService(resolver=Path,
        detector_factory=lambda p:created.append("align") or Detector(),
        session_factory=lambda p:created.append("expression") or Session())
    image=np.zeros((200,200,3),dtype=np.uint8)
    face={"bbox":[.25,.25,.75,.75],"yaw_degrees":0}
    for _ in range(4):
        observed=service.inspect(image,faces=[face])
        assert len(observed["faces"])==1 and observed["unknown_faces"]==0
        assert np.sum(observed["faces"][0]["probabilities"])==pytest.approx(1)
    assert created==["align","expression"] and len(crops)==4
    assert all(t.shape==(1,3,112,112) and np.isfinite(t).all() for t in tensors)
    assert not image.any()
    service.close()
    with pytest.raises(RuntimeError,match="closed"):
        service.inspect(image,faces=[])


def test_tiny_or_turned_faces_do_not_initialize_or_guess_a_model():
    def fail(_):pytest.fail("No inference on tiny or turned faces")
    service=ExpressionEvidenceService(detector_factory=fail)
    result=service.inspect(np.zeros((200,200,3),np.uint8),faces=[
        {"bbox":[.4,.4,.45,.45],"yaw_degrees":0},
        {"bbox":[.1,.1,.9,.9],"yaw_degrees":60}])
    assert not result["faces"] and result["unknown_faces"]==2
    service.close()


def test_missing_alignment_cannot_become_a_neutral_expression():
    class Detector:
        def setInputSize(self,size):pass
        def detect(self,crop):return None,None
    service=ExpressionEvidenceService(resolver=Path,detector_factory=lambda _:Detector(),
                                     session_factory=lambda _:object())
    observed=service.inspect(np.zeros((200,200,3),np.uint8),faces=[
        {"bbox":[.1,.1,.9,.9],"yaw_degrees":0}])
    assert observed["unknown_faces"]==1 and observed["faces"]==[]
    service.close()

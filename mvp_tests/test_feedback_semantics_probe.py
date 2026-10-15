"""Probe guards must run before importing/loading an expensive model."""
import json
import sys
import pytest

from scripts.probe_feedback_semantics import main


@pytest.mark.parametrize('destination',['source','review','weights'])
def test_probe_refuses_protected_output_before_model_load(tmp_path,monkeypatch,destination):
    source=tmp_path/'source';source.mkdir()
    photo=source/'a.ARW';photo.write_bytes(b'fixture')
    review=tmp_path/'review';review.mkdir()
    (review/'review.json').write_text(json.dumps({'samples':{'test':[{'ai_path':str(photo),'human_path':None}]}}))
    weights_dir=tmp_path/'weights';weights_dir.mkdir()
    weights=weights_dir/'RN50.pt';weights.write_bytes(b'not-a-model')
    inventory=tmp_path/'inventory.json'
    inventory.write_text(json.dumps({str(photo):[photo.stat().st_size,photo.stat().st_mtime_ns]}))
    parent={'source':source,'review':review,'weights':weights_dir}[destination]
    output=parent/'output'
    monkeypatch.setattr(sys,'argv',['probe','--review',str(review),'--weights',str(weights),
                                   '--inventory',str(inventory),'--output',str(output)])
    with pytest.raises(ValueError,match='output directory'):
        main()
    assert not output.exists()

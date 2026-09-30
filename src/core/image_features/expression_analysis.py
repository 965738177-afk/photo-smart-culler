"""Local, bounded visible-expression evidence, not mood or wedding intent.

Reuse frozen person boxes and the caller's decoded RGB preview. YuNet is used
only inside each person box to obtain five alignment landmarks, not to rescan
the full image. Unknown, tiny or turned faces never become rejected photos.
"""
from __future__ import annotations

import threading
import cv2
import numpy as np

from .human_analysis import bundled_model
from .face_analysis import FaceRegionDetector

PIPELINE_ID = "opencv-progressive-teacher-aligned-v1"
CLASSES = ("angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised")
FEATURE_DIMENSION = 4 * len(CLASSES) + 3
TARGET = np.asarray([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                     [41.5493,92.3655],[70.7299,92.2041]], dtype=np.float64)


def similarity_transform(points):
    """Deterministic least-squares similarity alignment; no random RANSAC."""
    points=np.asarray(points,dtype=np.float64)
    if points.shape != (5,2) or not np.isfinite(points).all():
        raise ValueError("Invalid expression alignment landmarks")
    x,y=points.T
    design=np.vstack((np.column_stack((x,-y,np.ones(5),np.zeros(5))),
                      np.column_stack((y,x,np.zeros(5),np.ones(5)))))
    if np.linalg.matrix_rank(design)<4:
        raise ValueError("Degenerate expression alignment landmarks")
    a,b,tx,ty=np.linalg.lstsq(design,np.concatenate((TARGET[:,0],TARGET[:,1])),rcond=None)[0]
    return np.asarray([[a,-b,tx],[b,a,ty]],dtype=np.float32)


def aggregate_expressions(records, *, unknown):
    if records:
        values=np.asarray([row["probabilities"] for row in records])
        largest=max(records,key=lambda row:row["area"])
        vector=np.concatenate((values.mean(0),values.min(0),values.max(0),largest["probabilities"]))
    else:
        vector=np.zeros(4*len(CLASSES))
    features=np.concatenate((vector,[len(records),unknown,bool(records)]))
    if features.shape != (FEATURE_DIMENSION,) or not np.isfinite(features).all():
        raise ValueError("Invalid expression evidence")
    return {"pipeline":PIPELINE_ID,"status":"ok","features":features.tolist(),
            "faces":records,"unknown_faces":unknown}


class ExpressionEvidenceService:
    """One lazy alignment detector and ONNX session per caller-owned run."""
    def __init__(self, *, resolver=bundled_model, checkpoint=lambda:None,
                 detector_factory=None, session_factory=None):
        self._resolver,self._checkpoint=resolver,checkpoint
        self._detector_factory,self._session_factory=detector_factory,session_factory
        self._detector=self._session=None
        self._lock=threading.RLock()
        self._closed=False

    def _load(self):
        if self._closed:
            raise RuntimeError("Expression service is closed")
        if self._detector is None:
            self._detector=FaceRegionDetector(
                model_path_resolver=lambda:self._resolver("face_detection_yunet_2023mar.onnx"),
                backend_factory=self._detector_factory, checkpoint=self._checkpoint)
            self._detector.load()
        if self._session is None:
            path=str(self._resolver("facial_expression_recognition_mobilefacenet_2022july.onnx"))
            if self._session_factory:
                self._session=self._session_factory(path)
            else:
                import onnxruntime as ort
                options=ort.SessionOptions()
                options.intra_op_num_threads=2
                options.inter_op_num_threads=1
                options.log_severity_level=3
                self._session=ort.InferenceSession(path,sess_options=options,providers=["CPUExecutionProvider"])

    def inspect(self, rgb, *, faces):
        with self._lock:
            self._checkpoint()
            if self._closed:
                raise RuntimeError("Expression service is closed")
            if (rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8
                    or min(rgb.shape[:2])<1 or max(rgb.shape[:2])>1600):
                raise ValueError("Invalid shared expression preview")
            h,w=rgb.shape[:2]
            records=[]
            unknown=0
            for face in faces[:12]:
                self._checkpoint()
                bbox=np.asarray(face.get("bbox",()),dtype=float)
                if bbox.shape!=(4,) or not np.isfinite(bbox).all():
                    raise ValueError("Invalid frozen person box")
                x0,y0,x1,y1=bbox*np.asarray([w,h,w,h])
                if x1-x0<32 or y1-y0<32 or abs(face.get("yaw_degrees",90))>40:
                    unknown+=1
                    continue
                self._load()
                pad=.2*max(x1-x0,y1-y0)
                x0,y0,x1,y1=max(0,int(x0-pad)),max(0,int(y0-pad)),min(w,int(x1+pad)),min(h,int(y1+pad))
                if x1<=x0 or y1<=y0:
                    unknown+=1
                    continue
                crop=cv2.cvtColor(rgb[y0:y1,x0:x1],cv2.COLOR_RGB2BGR)
                # Alignment has bounded work even on large RAW previews.
                scale=min(1.,320/max(crop.shape[:2]))
                if scale<1:
                    crop=cv2.resize(crop,None,fx=scale,fy=scale)
                detected=self._detector.detect(crop, maximum_dimension=320)
                if not len(detected):
                    unknown+=1
                    continue
                # Select the central face in this known box, not an adjacent
                # person's expression. Do not infer missing faces as neutral.
                cx,cy=crop.shape[1]/2,crop.shape[0]/2
                row=min(detected,key=lambda d:((d[0]+d[2]/2-cx)**2+(d[1]+d[3]/2-cy)**2))
                points=np.asarray(row[4:14]).reshape(5,2)
                transform=similarity_transform(points)
                aligned=cv2.warpAffine(crop,transform,(112,112))
                tensor=cv2.cvtColor(aligned,cv2.COLOR_BGR2RGB).astype(np.float32)/127.5-1.
                tensor=np.ascontiguousarray(tensor.transpose(2,0,1)[None])
                logits=np.asarray(self._session.run(["label"],{"data":tensor})[0]).reshape(-1)
                if logits.shape!=(len(CLASSES),) or not np.isfinite(logits).all():
                    raise ValueError("Invalid expression model output")
                probabilities=np.exp(logits-logits.max())
                probabilities/=probabilities.sum()
                records.append({"probabilities":probabilities.tolist(),"box":bbox.tolist(),
                                "area":float((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))})
            self._checkpoint()
            return aggregate_expressions(records,unknown=unknown+max(0,len(faces)-12))

    def close(self):
        with self._lock:
            self._closed=True
            if self._detector is not None:
                self._detector.close()
            self._session=self._detector=None

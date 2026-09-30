"""Bounded local person evidence, shared independently of GUI or image decoder.

Blendshapes, head orientation and visible body joints are observations only.
Missing evidence is recorded explicitly, never interpreted as closed eyes,
bad poses, hidden hands, completed actions, identity, or wedding role.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import math
import threading
from functools import lru_cache
import cv2
import numpy as np

from .face_analysis import FaceAnalysisService, FaceRegionDetector, FACE_REGION_PIPELINE
from .landmarker_runtime import TasksLandmarker
from ..runtime_paths import iter_bundle_roots

PIPELINE_ID = "mediapipe-person-evidence-v1"
DETAIL_PIPELINE_ID = "mediapipe-observed-person-joint-sets-v1"
MAX_POSES = 6
BLENDSHAPES = (
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
)
LEFT_EYE, RIGHT_EYE = (33, 160, 158, 133, 153, 144), (362, 385, 387, 263, 373, 380)
# Per face: 51 blendshapes + 2 aspect ratios + yaw/roll + face width + sharpness.
FACE_DIMENSION = len(BLENDSHAPES) + 6
# 3 summaries of the faces, 1 largest-face descriptor, face/pose coverage masks;
# mean body joint coordinates/visibility (33*4), plus worst/mean visible fraction.
FEATURE_DIMENSION = FACE_DIMENSION * 4 + 4 + 33 * 4 + 2


def bundled_model(name):
    candidates = [Path(root) / "models" / name for root in iter_bundle_roots()]
    candidates.append(Path(__file__).resolve().parents[3] / "assets" / "models" / name)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing bundled person model: {name}")


def model_signature(*, recover_faces=False):
    names = ("face_landmarker.task", "pose_landmarker_lite.task")
    if recover_faces:
        names += ("face_detection_yunet_2023mar.onnx",)
    paths = tuple(bundled_model(name) for name in names)
    keys = tuple((str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns) for p in paths)
    return _model_signature(keys, FACE_REGION_PIPELINE if recover_faces else "")


@lru_cache(maxsize=4)
def _model_signature(keys, region_pipeline=""):
    digest = hashlib.sha256((PIPELINE_ID+DETAIL_PIPELINE_ID+region_pipeline).encode())
    for path, _size, _mtime in keys:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _box(points):
    return [float(min(p.x for p in points)), float(min(p.y for p in points)),
            float(max(p.x for p in points)), float(max(p.y for p in points))]


def _overlap(a, b):
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    intersection = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection / max(min(area_a, area_b), 1e-8)


def _ear(points, indices, width, height):
    # Normalized x and y have different units for portrait images. Measure in
    # pixels rather than incorrectly calling tall portrait eyes "more open".
    p = np.asarray([(points[i].x*width, points[i].y*height) for i in indices])
    return float((np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4])) /
                 max(2*np.linalg.norm(p[0]-p[3]), 1e-6))


def _face_record(points, shapes, matrix, rgb, region):
    h, w = rgb.shape[:2]
    x, y, rw, rh = region
    local = _box(points)
    bbox = [(x+local[0]*rw)/w, (y+local[1]*rh)/h,
            (x+local[2]*rw)/w, (y+local[3]*rh)/h]
    x0,y0,x1,y1 = (max(0,min(w,round(bbox[0]*w))),max(0,min(h,round(bbox[1]*h))),
                   max(0,min(w,round(bbox[2]*w))),max(0,min(h,round(bbox[3]*h))))
    crop = rgb[y0:y1,x0:x1]
    sharpness = 0.
    if crop.size:
        gray = cv2.cvtColor(cv2.resize(crop,(96,96)),cv2.COLOR_RGB2GRAY)
        sharpness = math.log1p(float(cv2.Laplacian(gray,cv2.CV_64F).var()))
    blend = {s.category_name:float(s.score) for s in shapes}
    yaw = roll = 0.
    rotation_known = matrix is not None and np.asarray(matrix).shape == (4,4)
    if rotation_known:
        m = np.asarray(matrix)
        yaw = math.atan2(-m[2,0],math.hypot(m[0,0],m[1,0]))
        roll = math.atan2(m[1,0],m[0,0])
    ratios = [_ear(points, LEFT_EYE, rw, rh),_ear(points, RIGHT_EYE, rw, rh)]
    # Tiny/turned/soft faces have unknown eye state even if the network emits
    # confident-looking coefficients. They still contribute geometric evidence.
    reliable = (x1-x0 >= 32 and y1-y0 >= 32 and abs(yaw) < math.radians(40)
                and sharpness >= math.log1p(35) and len(blend) >= 40 and rotation_known)
    values = [blend.get(name,0.) for name in BLENDSHAPES] + ratios + [yaw,roll,(x1-x0)/w,sharpness]
    if not np.isfinite(values + bbox).all():
        raise ValueError("Nonfinite person model result")
    return {"bbox":bbox,"features":values,"reliable_eyes":bool(reliable),
            "blink":max(blend.get("eyeBlinkLeft",0.),blend.get("eyeBlinkRight",0.)),
            "smile":min(blend.get("mouthSmileLeft",0.),blend.get("mouthSmileRight",0.)),
            "yaw_degrees":math.degrees(yaw)}


def aggregate_observation(faces, poses, *, pose_records=None):
    values = np.asarray([face["features"] for face in faces],dtype=np.float32)
    if len(values):
        largest = max(faces,key=lambda f:(f["bbox"][2]-f["bbox"][0])*(f["bbox"][3]-f["bbox"][1]))
        face_vector = np.concatenate((values.mean(0),values.min(0),values.max(0),largest["features"]))
    else:
        face_vector = np.zeros(FACE_DIMENSION*4)
    pose_values = np.asarray(poses,dtype=np.float32)
    if len(pose_values):
        # Body shape is translation/scale-normalized, never person identity.
        visible_fraction = (pose_values[:,:,3] >= .5).mean(axis=1)
        body = pose_values.mean(axis=0).ravel()
        visibility = [float(visible_fraction.min()),float(visible_fraction.mean())]
    else:
        body,visibility = np.zeros(33*4),[0.,0.]
    masks = [len(faces),sum(face["reliable_eyes"] for face in faces),len(poses),bool(faces)]
    features = np.concatenate((face_vector,masks,body,visibility)).astype(np.float32)
    if features.shape != (FEATURE_DIMENSION,) or not np.isfinite(features).all():
        raise ValueError("Invalid person feature dimensions")
    details_available = pose_records is not None or not poses
    return {"pipeline":PIPELINE_ID,"status":"ok","faces":faces,"pose_count":len(poses),
            "features":features.tolist(),"unknown_eyes":len(faces)-int(masks[1]),
            "detail_pipeline":DETAIL_PIPELINE_ID if details_available else None,
            "poses":pose_records if pose_records is not None else []}


def pose_record(points, width, height):
    """Retain an observed skeleton, normalized using only visible image joints.

    Bounds describe observed joints, NOT a completed body or a semantic action.
    Out-of-frame/inferred missing joints stay unknown even if the model emits a
    high presence coefficient. No face identity or role association is guessed.
    """
    if len(points) != 33:
        return None
    positions = np.asarray([(p.x, p.y, p.z) for p in points], np.float64)
    visibility = np.asarray([min(p.visibility, p.presence) for p in points], np.float64)
    if not np.isfinite(positions).all() or not np.isfinite(visibility).all():
        raise ValueError("Nonfinite observed pose")
    inside = ((positions[:, :2] >= 0) & (positions[:, :2] <= 1)).all(axis=1)
    visibility = np.where(inside, np.clip(visibility, 0, 1), 0.)
    known = visibility >= .5
    if known.sum() < 5:
        return None
    xy = positions[:, :2] * (width, height)
    minimum, maximum = xy[known].min(axis=0), xy[known].max(axis=0)
    centre = (minimum+maximum)/2
    scale = max(float((maximum-minimum).max()), .05*max(width, height))
    joints = np.column_stack(((xy-centre)/scale, positions[:, 2]*width/scale, visibility))
    joints[~known, :3] = 0.
    bounds = np.concatenate((minimum/(width, height), maximum/(width, height)))
    return {"joints":joints.tolist(),"bbox":bounds.tolist(),
            "visible_fraction":float(known.mean()),"bounds_are_complete_body":False}


def validate_observation(value, *, require_details=False):
    """Shared worker/cache contract; no decoding or model initialization."""
    if (not isinstance(value, dict) or value.get("pipeline") != PIPELINE_ID or value.get("status") != "ok"
            or not isinstance(value.get("features"), list) or len(value["features"]) != FEATURE_DIMENSION
            or not np.isfinite(value["features"]).all()):
        raise ValueError("Incomplete person evidence")
    if require_details:
        poses = value.get("poses")
        if (value.get("detail_pipeline") != DETAIL_PIPELINE_ID or not isinstance(poses, list)
                or len(poses) != value.get("pose_count") or len(poses) > MAX_POSES):
            raise ValueError("Incomplete per-person joints; not recoverable from averaged pose")
        for pose in poses:
            if not isinstance(pose, dict):
                raise ValueError("Invalid observed joint record")
            joints = np.asarray(pose.get("joints", ()), np.float64)
            box = np.asarray(pose.get("bbox", ()), np.float64)
            if (joints.shape != (33, 4) or box.shape != (4,) or not np.isfinite(joints).all()
                    or not np.isfinite(box).all() or not ((box >= 0) & (box <= 1)).all()
                    or (box[2:] < box[:2]).any() or not ((joints[:, 3] >= 0) & (joints[:, 3] <= 1)).all()
                    or np.any(joints[joints[:, 3] < .5, :3] != 0)
                    or (joints[:, 3] >= .5).sum() < 5
                    or pose.get("bounds_are_complete_body") is not False):
                raise ValueError("Invalid observed joint geometry/unknown state")


class HumanAnalysisService:
    """One lazy face/pose runtime, serialized inference, caller-owned lifecycle."""
    def __init__(self, face_factory=None, pose_factory=None, resolver=bundled_model, checkpoint=lambda:None,
                 *, recover_faces=False, region_factory=None):
        if type(recover_faces) is not bool:
            raise ValueError("Face recovery mode must be boolean")
        self._lock = threading.RLock()
        self._faces = FaceAnalysisService(
            backend_factory=face_factory or (lambda p:TasksLandmarker(p,rich=True)),
            model_path_resolver=lambda:resolver("face_landmarker.task"))
        self._pose_factory = pose_factory or (lambda p:TasksLandmarker(p,kind="pose"))
        self._resolver,self._pose = resolver,None
        self._checkpoint = checkpoint
        self._regions = (FaceRegionDetector(
            model_path_resolver=lambda:resolver("face_detection_yunet_2023mar.onnx"),
            backend_factory=region_factory, checkpoint=checkpoint) if recover_faces else None)

    def inspect(self, rgb, *, face_regions=(), checkpoint=None):
        checkpoint = checkpoint or self._checkpoint
        with self._lock:
            checkpoint()
            h,w = rgb.shape[:2]
            faces = []
            def detect(region):
                checkpoint()
                x,y,rw,rh = region
                result = self._faces.get_backend().detect_result(rgb[y:y+rh,x:x+rw])
                matrices = result.facial_transformation_matrixes
                for i,points in enumerate(result.face_landmarks):
                    record = _face_record(points,result.face_blendshapes[i],
                                          matrices[i] if i<len(matrices) else None,rgb,region)
                    if not any(_overlap(record["bbox"],f["bbox"]) > .6 for f in faces):
                        faces.append(record)
            detect((0,0,w,h))
            proposals = list(face_regions)
            if self._regions is not None:
                # The same detector owner is used by alignment. Recover native
                # observations inside proposals; a proposal alone is not a face
                # expression or an assertion that all people have been found.
                detected = self._regions.detect(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR), checkpoint=checkpoint)
                proposals = [tuple(round(float(v)) for v in row[:4]) for row in detected]+proposals
            # Reuse the existing detector's regions, not another Haar scan. At
            # most 12 crops; no giant full-resolution frame or unbounded retries.
            for x,y,rw,rh in proposals[:12]:
                bbox = [x/w,y/h,(x+rw)/w,(y+rh)/h]
                if any(_overlap(bbox,f["bbox"])>.5 for f in faces):
                    continue
                pad = round(max(rw,rh)*.5)
                x0,y0,x1,y1 = max(0,x-pad),max(0,y-pad),min(w,x+rw+pad),min(h,y+rh+pad)
                if x1-x0>=24 and y1-y0>=24:
                    detect((x0,y0,x1-x0,y1-y0))
            checkpoint()
            if self._pose is None:
                self._pose = self._pose_factory(self._resolver("pose_landmarker_lite.task"))
            # Lite body inference needs no larger than a 640-pixel preview.
            scale = min(1.,640/max(h,w))
            small = cv2.resize(rgb,None,fx=scale,fy=scale) if scale<1 else rgb
            raw = self._pose.detect_result(small)
            poses,records=[],[]
            for points in list(raw.pose_landmarks)[:MAX_POSES]:
                if len(points)!=33:
                    continue
                xy=np.asarray([(p.x*w,p.y*h) for p in points])
                visible=np.asarray([min(p.visibility,p.presence) for p in points])
                if sum(visible>=.5)<5:
                    continue
                record=pose_record(points,w,h)
                if record is None:
                    continue
                centre=(xy[23]+xy[24])/2
                size=max(float(np.linalg.norm(xy[11]-xy[12])),.05*w)
                # Keep the old hip/shoulder-normalized descriptor for baseline
                # comparison. Its native coefficients are not complete-body
                # evidence; the separate records above explicitly mask joints
                # outside the frame and are the joint-set owner's input.
                poses.append([[float((p.x*w-centre[0])/size) if v>=.5 else 0.,
                               float((p.y*h-centre[1])/size) if v>=.5 else 0.,
                               float(p.z*w/size) if v>=.5 else 0.,float(v)]
                              for p,v in zip(points,visible)])
                records.append(record)
            checkpoint()
            observed = aggregate_observation(faces,poses,pose_records=records)
            if self._regions is not None:
                observed.update(region_pipeline=FACE_REGION_PIPELINE,
                                region_proposal_count=len(proposals), region_crop_cap=12)
            return observed

    def close(self):
        with self._lock:
            self._faces.close()
            if self._regions is not None:
                self._regions.close()
            if self._pose is not None:
                self._pose.close()
                self._pose=None

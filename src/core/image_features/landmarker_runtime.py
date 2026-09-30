"""UI-independent MediaPipe runtime shared by selection and subject workflows.

Inputs are already decoded RGB arrays. No downloads, file decoding, Qt imports,
or application state live here. macOS uses the tested Metal/SRGBA path; its CPU
graph is unsafe with the pinned MediaPipe release.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np


class TasksLandmarker:
    def __init__(self, model_path: Path, *, kind="face", rich=False,
                 module_loader=__import__, platform=None):
        mp = module_loader("mediapipe")
        self._metal = (platform or sys.platform) == "darwin"
        tasks, vision = mp.tasks, mp.tasks.vision
        base = tasks.BaseOptions(model_asset_path=str(model_path), delegate=(
            tasks.BaseOptions.Delegate.GPU if self._metal else tasks.BaseOptions.Delegate.CPU))
        if kind == "face":
            options = vision.FaceLandmarkerOptions(
                base_options=base, running_mode=vision.RunningMode.IMAGE, num_faces=10,
                output_face_blendshapes=rich, output_facial_transformation_matrixes=rich)
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        elif kind == "pose":
            # Body landmarks are geometry, NOT an action-completion classifier.
            options = vision.PoseLandmarkerOptions(
                base_options=base, running_mode=vision.RunningMode.IMAGE,
                num_poses=6, output_segmentation_masks=False)
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        else:
            raise ValueError("Unsupported landmarker kind")
        self._image_type = mp.Image
        self._image_format = mp.ImageFormat.SRGBA if self._metal else mp.ImageFormat.SRGB

    def detect_result(self, rgb_image):
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3 or rgb_image.dtype != np.uint8:
            raise ValueError("Landmarker requires an uint8 RGB image")
        if self._metal:
            # CVPixelBuffer-backed GPU images require compatible row alignment.
            # Resize (not pad) so returned normalized coordinates still refer to
            # the entire original image. Tiny rounding changes no crop origin.
            h,w = rgb_image.shape[:2]
            if h % 4 or w % 4:
                import cv2
                rgb_image = cv2.resize(rgb_image,((w+3)//4*4,(h+3)//4*4))
            pixels = np.empty((*rgb_image.shape[:2], 4), dtype=np.uint8)
            pixels[..., :3], pixels[..., 3] = rgb_image, 255
        else:
            pixels = np.ascontiguousarray(rgb_image)
        return self._landmarker.detect(self._image_type(image_format=self._image_format, data=pixels))

    def detect_landmarks(self, rgb_image):
        return self.detect_result(rgb_image).face_landmarks

    def close(self):
        self._landmarker.close()

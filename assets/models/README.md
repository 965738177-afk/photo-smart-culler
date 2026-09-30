# MediaPipe Face Landmarker model

`face_landmarker.task` is Google's MediaPipe Face Landmarker float16 model
bundle. PhotoSort uses it locally for face landmarks and closed-eye scoring.

- Source: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
- Retrieved: 2026-07-15
- SHA-256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
- Model card: https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf

The model is distributed with MediaPipe under the terms documented by the
upstream MediaPipe project. Keep the checksum and source synchronized whenever
the asset is replaced.

## Pose Landmarker Lite (experimental person evidence)

- Official, pinned model: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
- Retrieved: 2026-09-29; size: 5,777,746 bytes
- SHA-256: `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`
- Model card and license (Apache 2.0): https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf
- Outputs body geometry/visibility, not action completeness, wedding roles,
  expression quality or identity. Wedding crowds/distant/occluded subjects are
  outside the original single-person fitness evaluation; no quality claim.

## Visible expression experiment (not enabled in the desktop)

Both files are unmodified OpenCV Zoo assets pinned to commit
`47534e27c9851bb1128ccc0102f1145e27f23f98`. The extractor verifies these upstream
Git LFS hashes before using them. They are not included by the current release
specs, and the ordinary GUI does not load them.

| Asset | Size in bytes | SHA-256 |
| --- | ---: | --- |
| `facial_expression_recognition_mobilefacenet_2022july.onnx` | 4791892 | `4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9` |
| `face_detection_yunet_2023mar.onnx` | 232589 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |

- Expression source and scope: https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/facial_expression_recognition
- YuNet source: https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet
- Expression classification: seven visible-expression categories, not internal
  emotions, action completeness, or whether a wedding photograph should be kept.
- Progressive Teacher contributed by Jing Jiang; MobileFaceNet ONNX conversion
  contributed by Chengrui Wang. Upstream demo copyright (C) 2022, Shenzhen
  Institute of Artificial Intelligence and Robotics for Society.
- Expression assets: upstream declares Apache 2.0; see `licenses/OPENCV-APACHE-2.0.txt`.
- YuNet: MIT, copyright (c) 2020 Shiqi Yu; see `licenses/YUNET-MIT.txt`.
- Upstream RAF-DB classification numbers are not wedding-retention accuracy.
  Tiny/turned/missing/un-aligned faces are unknown, not neutral or rejected.

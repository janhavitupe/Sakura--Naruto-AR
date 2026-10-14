"""Face + Expression Module.

Input : RGB inference image, full frame shape, timestamp (s).
Output: Face with 478 pixel landmarks (incl. iris), MediaPipe blendshape
        scores, and derived geometry used by the gesture/AR layers.

Expressions come from MediaPipe's 52 ARKit-style blendshapes, which are far
more robust than hand-crafted landmark ratios (e.g. browDown, eyeSquint).
"""
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import settings as S
from modules.temporal import OneEuroFilter

# Face-mesh indices
FOREHEAD_CENTER = 151
FOREHEAD_TOP = 10
BROW_MID = 9
NOSE_BRIDGE = 168
NOSE_TIP = 1
CHIN = 152
LEFT_FACE, RIGHT_FACE = 234, 454
# Eye contours (subject's right eye = image left after mirroring)
RIGHT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
LEFT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
RIGHT_IRIS = [468, 469, 470, 471, 472]   # centre first
LEFT_IRIS = [473, 474, 475, 476, 477]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
             379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]


class Face:
    def __init__(self, pts, blend):
        self.pts = pts                                   # (478, 3) pixels
        self.blend = blend                               # name -> score
        self.width = float(np.linalg.norm(pts[LEFT_FACE, :2] - pts[RIGHT_FACE, :2]))
        self.anchor = pts[NOSE_BRIDGE, :2].copy()
        self.forehead = pts[FOREHEAD_CENTER, :2].copy()
        eye_vec = pts[263, :2] - pts[33, :2]
        self.roll = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))
        self.up = pts[FOREHEAD_TOP, :2] - pts[CHIN, :2]
        self.up /= max(np.linalg.norm(self.up), 1e-6)
        self.expression = {
            "brow_down": 0.5 * (blend.get("browDownLeft", 0) + blend.get("browDownRight", 0)),
            "eye_squint": 0.5 * (blend.get("eyeSquintLeft", 0) + blend.get("eyeSquintRight", 0)),
            "eye_wide": 0.5 * (blend.get("eyeWideLeft", 0) + blend.get("eyeWideRight", 0)),
            "eye_blink": 0.5 * (blend.get("eyeBlinkLeft", 0) + blend.get("eyeBlinkRight", 0)),
            "smile": 0.5 * (blend.get("mouthSmileLeft", 0) + blend.get("mouthSmileRight", 0)),
        }

    def eyes(self):
        """Yields (iris_centre, iris_radius, eye_contour_polygon) per eye."""
        for iris, contour in ((RIGHT_IRIS, RIGHT_EYE), (LEFT_IRIS, LEFT_EYE)):
            c = self.pts[iris[0], :2]
            r = float(np.mean(np.linalg.norm(self.pts[iris[1:], :2] - c, axis=1)))
            yield c, r, self.pts[contour, :2].astype(np.int32)


class FaceTracker:
    def __init__(self, model_path=S.FACE_MODEL_PATH):
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=S.FACE_DETECTION_CONF,
            min_face_presence_confidence=S.FACE_PRESENCE_CONF,
            min_tracking_confidence=S.FACE_TRACKING_CONF,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.filter = OneEuroFilter()

    def process(self, rgb_small, frame_shape, t):
        h, w = frame_shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
        result = self.detector.detect_for_video(image, int(t * 1000))
        if not result.face_landmarks:
            self.filter.reset()
            return None
        lms = result.face_landmarks[0]
        pts = np.array([(p.x * w, p.y * h, p.z * w) for p in lms], dtype=np.float32)
        pts = self.filter(pts, t)
        blend = {}
        if result.face_blendshapes:
            blend = {c.category_name: c.score for c in result.face_blendshapes[0]}
        return Face(pts, blend)

    def close(self):
        self.detector.close()


def draw_face(frame, face):
    for i in FACE_OVAL + RIGHT_EYE + LEFT_EYE:
        cv2.circle(frame, tuple(face.pts[i, :2].astype(int)), 1, (255, 200, 0), -1)
    cv2.circle(frame, tuple(face.forehead.astype(int)), 4, (0, 255, 255), 1)

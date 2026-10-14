"""Hand Tracking Module.

Input : BGR frame (already mirrored), timestamp in seconds.
Output: list[Hand] with 21 landmarks in *pixel* space (x, y, z*width),
        handedness label and score, One-Euro filtered per hand.
"""
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import settings as S
from modules.temporal import OneEuroFilter


@dataclass
class Hand:
    landmarks: np.ndarray   # (21, 3) float32, pixels
    label: str              # "Left" / "Right" (user's own hand, input is mirrored)
    score: float


class HandTracker:
    def __init__(self, model_path=S.HAND_MODEL_PATH):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=S.HAND_DETECTION_CONF,
            min_hand_presence_confidence=S.HAND_PRESENCE_CONF,
            min_tracking_confidence=S.HAND_TRACKING_CONF,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.filters = {}

    def process(self, rgb_small, frame_shape, t):
        """`rgb_small` is the (possibly downscaled) RGB inference image,
        `frame_shape` the full-resolution frame the landmarks map onto."""
        h, w = frame_shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
        result = self.detector.detect_for_video(image, int(t * 1000))

        hands, seen = [], set()
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            label = handed[0].category_name
            # Two hands with the same label happens briefly when tracking
            # flips; disambiguate so filters don't fight each other.
            if label in seen:
                label = "Left" if label == "Right" else "Right"
            seen.add(label)

            pts = np.array([(p.x * w, p.y * h, p.z * w) for p in lms], dtype=np.float32)
            filt = self.filters.setdefault(label, OneEuroFilter())
            pts = filt(pts, t)
            hands.append(Hand(pts, label, handed[0].score))

        for label in list(self.filters):
            if label not in seen:
                self.filters[label].reset()
        return hands

    def close(self):
        self.detector.close()


def draw_hands(frame, hands, notes=None):
    """Debug skeleton overlay. `notes` maps label -> extra text."""
    conns = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
             (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
             (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
    for hand in hands:
        p = hand.landmarks[:, :2].astype(int)
        for a, b in conns:
            cv2.line(frame, tuple(p[a]), tuple(p[b]), (0, 255, 0), 1, cv2.LINE_AA)
        for q in p:
            cv2.circle(frame, tuple(q), 2, (0, 0, 255), -1)
        text = hand.label + (" " + notes[hand.label] if notes and hand.label in notes else "")
        cv2.putText(frame, text, tuple(p[0] + [0, 18]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

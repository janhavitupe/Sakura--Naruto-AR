"""Gesture Classification (rule-based, per frame).

Input : list[HandFeatures], Face | None
Output: dict of raw (unsmoothed) gesture booleans for this frame, e.g.
        {"rasengan_cup": True, "claw:Right": False, "forehead_touch": False}

Conflict handling happens here at the spatial level: gestures are evaluated
in GESTURE_PRIORITY order and a hand that satisfies a higher-priority
gesture is *consumed*, so it can't also produce a lower one. Example: two
cupped hands look like two claws; once they match `rasengan_cup` neither hand
is checked for `claw`.

Rules are geometric and transparent on purpose: they need no training data,
and each threshold in settings.py maps to one physical property.
"""
import numpy as np

from config import settings as S
from modules.feature_extractor import FINGERS, distance


def is_open_palm(f):
    straight = all(f.extended(n, S.FINGER_EXTENDED_MAX_BEND) for n in FINGERS)
    thumb_out = f.thumb_out > S.PALM_MIN_THUMB_OUT
    return straight and thumb_out and f.spread > S.PALM_MIN_SPREAD and f.palm_facing


def is_claw(f):
    clawed = sum(
        1 for n in FINGERS
        if S.CLAW_MIN_BEND <= f.bend[n] <= S.CLAW_MAX_BEND
        and f.tip_dist[n] >= S.CLAW_MIN_TIP_DIST
    )
    return clawed >= S.CLAW_MIN_FINGERS


def is_cup(f1, f2):
    palm = 0.5 * (f1.palm_size + f2.palm_size)
    d = distance(f1.center, f2.center) / palm
    if not (S.CUP_MIN_CENTER_DIST <= d <= S.CUP_MAX_CENTER_DIST):
        return False
    return all(S.CUP_MIN_BEND <= f.mean_curl() <= S.CUP_MAX_BEND for f in (f1, f2))


def is_prayer(f1, f2):
    palm = 0.5 * (f1.palm_size + f2.palm_size)
    if not (f1.extended("index", S.FINGER_EXTENDED_MAX_BEND)
            and f2.extended("index", S.FINGER_EXTENDED_MAX_BEND)):
        return False
    tips = distance(f1.index_tip, f2.index_tip) / palm
    wrists = distance(f1.wrist, f2.wrist) / palm
    aligned = float(np.dot(f1.index_dir, f2.index_dir))
    return (tips <= S.PRAYER_MAX_INDEX_TIP_DIST and wrists <= S.PRAYER_MAX_WRIST_DIST
            and aligned >= S.PRAYER_MIN_INDEX_COS)


def touching_forehead(f, face):
    r = S.FOREHEAD_TOUCH_RADIUS * face.width
    return min(distance(f.index_tip, face.forehead),
               distance(f.middle_tip, face.forehead)) <= r


class GestureClassifier:
    def classify(self, feats, face):
        raw = {}
        free = list(feats)

        # 1. Forehead touch: exactly one hand at the forehead. If both index
        #    tips are there it's a prayer held in front of the face instead.
        if face is not None:
            touching = [f for f in free if touching_forehead(f, face)]
            if len(touching) == 1:
                raw["forehead_touch"] = True
                free.remove(touching[0])

        # 2./3. Two-hand gestures
        if len(free) == 2:
            f1, f2 = free
            if is_prayer(f1, f2):
                raw["prayer"] = True
                free = []
            elif is_cup(f1, f2):
                raw["rasengan_cup"] = True
                free = []

        # 4./5. Single-hand gestures; claw outranks palm per hand.
        for f in free:
            if is_claw(f):
                raw[f"claw:{f.label}"] = True
            elif is_open_palm(f):
                raw[f"open_palm:{f.label}"] = True
        return raw

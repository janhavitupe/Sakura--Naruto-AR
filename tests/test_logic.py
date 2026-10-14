"""Logic tests with synthetic landmarks and timelines (no camera needed).

Run: python -m pytest tests -q
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings as S
from engine.jutsu_engine import JutsuEngine, classify_dojutsu
from modules.feature_extractor import HandFeatures
from modules.gesture_classifier import GestureClassifier, is_claw, is_open_palm
from modules.hand_tracker import Hand
from modules.state_manager import StateManager
from modules.temporal import GestureTracker, StabilityTracker, TapDetector, TemporalState


# ------------------------------------------------------------ synthetic hands
def make_hand(label="Right", bend=0.0, mcp_bend=0.0, center=(320, 300), scale=60.0,
              facing=True, spread_deg=12.0, up=(0.0, -1.0)):
    """2D hand: wrist at bottom, fingers pointing `up`. `bend` is the total
    PIP+DIP bend per finger in degrees (split evenly between the joints)."""
    up = np.array(up, float) / np.linalg.norm(up)
    right = np.array([-up[1], up[0]])
    # matches MediaPipe's convention: a "Right"-labelled palm facing the
    # camera has its index finger on the +x side
    side = 1.0 if (label == "Right") != facing else -1.0
    pts = np.zeros((21, 3), np.float32)
    wrist = np.array(center, float) - up * scale * 0.5
    pts[0, :2] = wrist
    fingers = [(5, -0.45), (9, -0.15), (13, 0.15), (17, 0.42)]
    for i, (mcp, off) in enumerate(fingers):
        base = wrist + up * scale + right * side * off * scale
        pts[mcp, :2] = base
        ang = math.radians((i - 1.5) * spread_deg * side)
        d = np.array([up[0] * math.cos(ang) - up[1] * math.sin(ang),
                      up[0] * math.sin(ang) + up[1] * math.cos(ang)])
        curl_dir = -side  # fingers curl toward the palm side (any fixed side works in 2D)
        # MCP bend: rotate the first segment relative to the wrist->MCP line
        d = _rot(d, mcp_bend * curl_dir)
        p = base
        for j, seg in enumerate((0.45, 0.3, 0.25)):
            p = p + d * seg * scale
            pts[mcp + 1 + j, :2] = p
            d = _rot(d, bend / 2 * curl_dir)
    # thumb out to the index side
    t_dir = -right * side
    for j in range(4):
        pts[1 + j, :2] = wrist + up * scale * (0.25 + 0.15 * j) + t_dir * scale * (0.25 + 0.2 * j)
    return Hand(pts, label, 0.99)


def _rot(v, deg):
    r = math.radians(deg)
    return np.array([v[0] * math.cos(r) - v[1] * math.sin(r),
                     v[0] * math.sin(r) + v[1] * math.cos(r)])


class FakeFace:
    def __init__(self, forehead=(320, 120), width=150.0):
        self.forehead = np.array(forehead, float)
        self.width = width
        self.anchor = np.array([forehead[0], forehead[1] + 40.0])
        self.expression = {"brow_down": 0.0, "eye_squint": 0.0, "eye_wide": 0.0}


def ts(**kw):
    base = dict(held=set(), just_held=set(), just_released=set(), progress={}, conf={},
                forehead_tap=False, stable=False, head_speed=0.0, brow_down=0.0,
                eye_squint=0.0, eye_wide=0.0, smile=0.0)
    base.update(kw)
    return TemporalState(**base)


# ------------------------------------------------------------ features / rules
def test_open_palm_requires_facing_camera():
    assert is_open_palm(HandFeatures(make_hand("Right", facing=True)))
    assert is_open_palm(HandFeatures(make_hand("Left", facing=True)))
    assert not is_open_palm(HandFeatures(make_hand("Right", facing=False)))


def test_open_palm_rotation_invariant():
    assert is_open_palm(HandFeatures(make_hand("Right", up=(1.0, -1.0))))


def test_claw_vs_open_vs_fist():
    assert is_claw(HandFeatures(make_hand(bend=100)))
    assert not is_claw(HandFeatures(make_hand(bend=0)))
    assert not is_open_palm(HandFeatures(make_hand(bend=100)))


def test_cup_consumes_hands_so_no_claw():
    gc = GestureClassifier()
    h1 = HandFeatures(make_hand("Right", bend=90, mcp_bend=30, center=(300, 300)))
    h2 = HandFeatures(make_hand("Left", bend=90, mcp_bend=30, center=(380, 300)))
    raw = gc.classify([h1, h2], None)
    assert raw.get("rasengan_cup")
    assert not any(k.startswith("claw") for k in raw)


def test_hands_far_apart_are_two_claws_not_cup():
    gc = GestureClassifier()
    h1 = HandFeatures(make_hand("Right", bend=100, center=(100, 300)))
    h2 = HandFeatures(make_hand("Left", bend=100, center=(560, 300)))
    raw = gc.classify([h1, h2], None)
    assert "rasengan_cup" not in raw
    assert raw.get("claw:Right") and raw.get("claw:Left")


def test_prayer_index_aligned():
    gc = GestureClassifier()
    # palms pressed together: each hand's index side faces the other hand
    h1 = HandFeatures(make_hand("Right", center=(300, 300), spread_deg=0, facing=True))
    h2 = HandFeatures(make_hand("Left", center=(350, 300), spread_deg=0, facing=True))
    assert gc.classify([h1, h2], None).get("prayer")
    # index fingers pointing different ways -> not aligned
    h2 = HandFeatures(make_hand("Left", center=(350, 300), spread_deg=0, facing=True,
                                up=(0.7, -1.0)))
    assert not gc.classify([h1, h2], None).get("prayer")


def test_forehead_touch_single_hand_only():
    gc = GestureClassifier()
    face = FakeFace()
    hand = make_hand("Right", center=(320, 120 + 60 * 1.0))  # index tip near forehead
    f = HandFeatures(hand)
    face.forehead = f.index_tip.copy()
    assert gc.classify([f], face).get("forehead_touch")
    # both index tips at the forehead = prayer in front of face, not a tap
    f2 = HandFeatures(make_hand("Left", center=(320, 180)))
    face.forehead = 0.5 * (f.index_tip + f2.index_tip)
    assert not gc.classify([f, f2], face).get("forehead_touch")


# ------------------------------------------------------------ temporal
@pytest.mark.parametrize("fps", [15, 30, 60])
def test_hold_time_is_fps_independent(fps):
    tr, dt, t = GestureTracker("claw", hold_s=0.6), 1.0 / fps, 0.0
    fired_at = None
    while t < 2.0:
        tr.update(True, t, dt)
        if tr.just_held:
            fired_at = t
        t += dt
    assert fired_at is not None and 0.6 <= fired_at <= 0.6 + 0.35


def test_flicker_does_not_trigger_and_dropout_does_not_reset():
    tr, dt = GestureTracker("claw", 0.6), 1 / 30
    for i in range(60):                      # 50% flicker for 2s
        tr.update(i % 2 == 0, i * dt, dt)
    assert not tr.held

    tr = GestureTracker("claw", 0.6)
    t = 0.0
    for i in range(30):
        tr.update(True, t, dt); t += dt
    assert tr.held
    for i in range(3):                       # 0.1s dropout (missed detections)
        tr.update(False, t, dt); t += dt
    assert tr.held


def test_tap_detector():
    tap = TapDetector()
    events = [tap.update(touch, t * 0.033) for t, touch in
              enumerate([False] * 3 + [True] * 6 + [False] * 3)]
    assert events.count(True) == 1
    tap = TapDetector()                      # a 1-frame graze is not a tap
    assert not any(tap.update(touch, t * 0.033) for t, touch in
                   enumerate([True, False, False]))


def test_stability():
    st = StabilityTracker(window_s=1.0)
    for i in range(40):
        st.update((100 + 0.2 * i, 100), 150, i / 30)
    assert st.is_stable()
    st = StabilityTracker(window_s=1.0)
    for i in range(40):
        st.update((100 + 10 * i, 100), 150, i / 30)
    assert not st.is_stable()


# ------------------------------------------------------------ state + engine
def run(engine, frames, t0=0.0, dt=1 / 30, feats=None, face=None):
    t = t0
    for f in frames:
        engine.update(f, feats or {}, face, t)
        t += dt
    return t


def test_chidori_cost_and_cooldown():
    st = StateManager()
    eng = JutsuEngine(st)
    t = run(eng, [ts(just_held={"claw:Right"}, held={"claw:Right"})])
    assert "chidori" in eng.active
    assert st.chakra == pytest.approx(S.CHAKRA_MAX - S.COST["chidori"], abs=0.1)
    t = run(eng, [ts()] * int(S.CHIDORI_DURATION_S * 30 + 5), t0=t)
    assert "chidori" not in eng.active
    assert st.on_cooldown("chidori", t)
    run(eng, [ts(just_held={"claw:Right"}, held={"claw:Right"})], t0=t)
    assert "chidori" not in eng.active             # blocked by cooldown
    assert abs(st.cooldown_left("chidori", t) - S.STANDARD_COOLDOWN_S) < S.CHIDORI_DURATION_S + 1


def test_rasengan_drains_and_ends_on_release():
    st = StateManager()
    eng = JutsuEngine(st)
    feats = {"Left": None, "Right": None}
    t = run(eng, [ts(just_held={"rasengan_cup"}, held={"rasengan_cup"})], feats=feats)
    t = run(eng, [ts(held={"rasengan_cup"})] * 60, t0=t, feats=feats)    # ~2s
    assert "rasengan" in eng.active
    expected = S.CHAKRA_MAX - S.COST["rasengan_start"] - 2 * S.COST["rasengan_per_s"]
    assert st.chakra == pytest.approx(expected, abs=1.5)
    run(eng, [ts()], t0=t, feats=feats)
    assert "rasengan" not in eng.active and st.on_cooldown("rasengan")


def test_zero_chakra_blocks_everything():
    st = StateManager()
    st.chakra = 0.0
    eng = JutsuEngine(st)
    run(eng, [ts(just_held={"open_palm:Left"}, held={"open_palm:Left"})])
    assert not eng.active


def test_byakugou_needs_stability_then_gives_infinite_chakra():
    st = StateManager()
    eng = JutsuEngine(st)
    face = FakeFace()
    run(eng, [ts(forehead_tap=True, stable=False)], face=face)
    assert "byakugou" not in eng.active
    t = run(eng, [ts(forehead_tap=True, stable=True)], t0=1.0, face=face)
    assert "byakugou" in eng.active and st.infinite_chakra
    assert st.cooldown_left("byakugou", t) == pytest.approx(S.BYAKUGOU_COOLDOWN_S, abs=1)
    before = st.chakra
    run(eng, [ts(just_held={"claw:Right"}, held={"claw:Right"})], t0=t, face=face)
    assert "chidori" in eng.active and st.chakra == before


def test_one_hand_jutsu_at_a_time():
    st = StateManager()
    eng = JutsuEngine(st)
    run(eng, [ts(just_held={"claw:Right", "open_palm:Left"},
                 held={"claw:Right", "open_palm:Left"})])
    assert ("chidori" in eng.active) != ("lion_fist" in eng.active)


def test_dojutsu_classification():
    assert classify_dojutsu(brow_down=0.6, eye_squint=0.5) == "sharingan"
    assert classify_dojutsu(brow_down=0.05, eye_squint=0.1) == "byakugan"
    assert classify_dojutsu(brow_down=0.6, eye_squint=0.5, smile=0.8) == "byakugan"


def test_expression_baseline_is_relative_and_frozen_while_signing():
    from modules.temporal import ExpressionBaseline
    b = ExpressionBaseline(tau_s=1.0)
    for _ in range(100):                     # this user's resting browDown is 0.3
        b.update({"brow_down": 0.3}, 0.05, frozen=False)
    assert b.relative("brow_down", 0.3) == pytest.approx(0.0, abs=0.01)
    for _ in range(100):                     # furrowing during the sign
        b.update({"brow_down": 0.8}, 0.05, frozen=True)
    assert b.relative("brow_down", 0.8) == pytest.approx(0.5, abs=0.01)

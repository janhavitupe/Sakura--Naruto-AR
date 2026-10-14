"""Byakugou seal artwork, modelled on animations/2a5767bf30c9a17bf2fcda407bc3515c.jpg.

The design is authored once, for one side of the face, in a face-local
coordinate system and mirrored to the other side:

    a : horizontal, 0 at the seal, 1 at that side's outer eye corner
    b : vertical (up), in the same units

Each side gets its own horizontal axis (seal -> outer eye corner, with the
vertical component removed), so when the head turns, the far side is
foreshortened naturally. That is a cheap piecewise-affine face warp.

Strokes are filled polygons with a width profile along their length
(tapered brush strokes), drawn in two purple tones with a dark ink outline,
like the cel-shaded reference.
"""
import math

import cv2
import numpy as np

AA = cv2.LINE_AA

# Colours sampled from the reference image (BGR)
INK = (62, 22, 48)
STROKE = (125, 60, 95)
STROKE_LIGHT = (155, 78, 140)
GEM = (205, 95, 245)
GEM_LIGHT = (245, 185, 255)
GEM_INK = (120, 30, 150)
GEM_GLOW = (200, 60, 255)

EYE_OUTER = {"R": 33, "L": 263}       # mesh outer eye corners (image left / right)
BROW_MID, FOREHEAD_CENTER = 9, 151


def _chaikin(pts, iters=3):
    pts = np.asarray(pts, np.float32)
    for _ in range(iters):
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((len(q) * 2, 2), np.float32)
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def _spiral(center, r0, r1, a0_deg, turns, n=28):
    th = np.radians(a0_deg) - np.linspace(0, turns * 2 * np.pi, n)   # clockwise in (a, b)
    r = np.linspace(r0, r1, n)
    return np.stack([center[0] + r * np.cos(th), center[1] + r * np.sin(th)], 1)


# ---------------------------------------------------------------- design
# (control points in (a, b), width profile [(t, width)], reveal start, end)
# Widths are in the same face units.
def _design():
    strokes = []
    # Upper horn: sweeps up and out from the gem to a sharp point.
    strokes.append((_chaikin([(0.0, 0.03), (0.22, 0.2), (0.42, 0.36), (0.56, 0.44), (0.66, 0.46)]),
                    [(0, 0.15), (0.5, 0.1), (1, 0.0)], 0.08, 0.5))
    # Spiral curl beside the horn: lead-in from above, curling inward.
    curl = np.vstack([_chaikin([(0.80, 0.56), (0.76, 0.42), (0.70, 0.30)], 2),
                      _spiral((0.60, 0.25), 0.10, 0.025, 0, 1.15)])
    strokes.append((_chaikin(curl, 1), [(0, 0.0), (0.15, 0.07), (0.55, 0.075), (1, 0.025)], 0.35, 0.8))
    # Lower arm: from the gem down and out along the brow to the temple.
    strokes.append((_chaikin([(0.0, -0.03), (0.2, -0.13), (0.45, -0.17), (0.75, -0.17), (1.02, -0.12)]),
                    [(0, 0.15), (0.45, 0.11), (0.85, 0.06), (1, 0.0)], 0.08, 0.55))
    # Temple stroke: from upper temple down to meet the end of the lower arm.
    strokes.append((_chaikin([(1.0, 0.36), (1.05, 0.16), (1.04, -0.02), (0.98, -0.14)]),
                    [(0, 0.0), (0.4, 0.075), (1, 0.05)], 0.45, 0.8))
    # Under-eye mark: spindle below the outer eye corner.
    strokes.append((_chaikin([(0.86, -0.6), (0.85, -0.74), (0.81, -0.9), (0.76, -1.02)]),
                    [(0, 0.0), (0.3, 0.075), (1, 0.0)], 0.6, 0.95))
    return strokes


DESIGN = _design()

# Body extension (the task asks for markings that spread past the face):
# continues from the under-eye mark out to the face edge, along the jaw, then
# down the neck.
BODY_PATHS = {"R": [93, 132, 58, 172], "L": [323, 361, 288, 397]}


# ---------------------------------------------------------------- geometry
def _frame(face):
    """Per-side mapping (a, b) -> image pixels."""
    pts = face.pts[:, :2]
    seal = 0.55 * pts[BROW_MID] + 0.45 * pts[FOREHEAD_CENTER]
    up = face.up
    axes = {}
    for side, idx in EYE_OUTER.items():
        h = pts[idx] - seal
        axes[side] = h - np.dot(h, up) * up
    unit = 0.5 * sum(np.linalg.norm(v) for v in axes.values())
    return seal, up * unit, axes, unit


def _widths(profile, n):
    t = np.linspace(0, 1, n)
    ts, ws = zip(*profile)
    return np.interp(t, ts, ws)


def _ribbon(pts, widths):
    """Variable-width stroke outline polygon from a centreline."""
    d = np.gradient(pts, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-6
    half = (widths / 2)[:, None]
    return np.vstack([pts + n * half, (pts - n * half)[::-1]])


def _partial(n, frac):
    return max(2, int(round(n * min(max(frac, 0.0), 1.0))))


# ---------------------------------------------------------------- render
def draw_seal(frame, face, age, t, spread_s, body=True):
    """Paint the seal + markings onto `frame`. `age` = seconds since
    activation; strokes are revealed over `spread_s` seconds."""
    prog = min(1.0, age / spread_s)
    seal, vup, axes, unit = _frame(face)
    if unit < 8:
        return

    shapes = []          # (polygon in px, light-core polygon in px)
    for side, hx in axes.items():
        for ctrl, profile, start, end in DESIGN:
            local = (prog - start) / (end - start)
            if local <= 0:
                continue
            w = _widths(profile, len(ctrl))
            k = _partial(len(ctrl), local)
            ab = ctrl[:k]
            px = seal + ab[:, :1] * hx + ab[:, 1:] * vup
            wpx = w[:k] * unit
            shapes.append((_ribbon(px, wpx), _ribbon(px, wpx * 0.45)))

        if body and prog > 0.7:
            # start where the under-eye mark ends, so the line is continuous
            ab = DESIGN[-1][0][-1]
            start = seal + ab[0] * hx + ab[1] * vup
            jaw = [face.pts[i, :2] for i in BODY_PATHS[side]]
            neck = [jaw[-1] - face.up * unit * s for s in (0.6, 1.2)]
            line = _chaikin(np.array([start] + jaw + neck), 3)
            w = _widths([(0, 0.0), (0.15, 0.05), (0.7, 0.04), (1, 0.0)], len(line)) * unit
            k = _partial(len(line), (prog - 0.7) / 0.3)
            shapes.append((_ribbon(line[:k], w[:k]), _ribbon(line[:k], w[:k] * 0.45)))

    # Gem: a rounded square rotated 45 deg, grows in first.
    gem_s = unit * 0.17 * min(1.0, prog / 0.12)
    roll = face.roll
    gem = _rounded_diamond(seal, gem_s, roll)

    allp = np.vstack([s[0] for s in shapes] + [gem]) if shapes else gem
    pad = int(unit * 0.15) + 4
    x0, y0 = np.floor(allp.min(0)).astype(int) - pad
    x1, y1 = np.ceil(allp.max(0)).astype(int) + pad
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(fw, x1), min(fh, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    o = np.array([x0, y0], np.float32)
    roi = frame[y0:y1, x0:x1]
    paint = roi.copy()
    mask = np.zeros(roi.shape[:2], np.uint8)
    ink = max(1, int(unit * 0.008))
    S = 4                                     # sub-pixel precision for fillPoly
    to_i = lambda p: np.round((p - o) * (1 << S)).astype(np.int32)

    for poly, core in shapes:
        cv2.fillPoly(paint, [to_i(poly)], STROKE, AA, shift=S)
        cv2.fillPoly(paint, [to_i(core)], STROKE_LIGHT, AA, shift=S)
        cv2.polylines(paint, [to_i(poly)], True, INK, ink, AA, shift=S)
        cv2.fillPoly(mask, [to_i(poly)], 255, AA, shift=S)
        cv2.polylines(mask, [to_i(poly)], True, 255, ink, AA, shift=S)

    if gem_s > 1:
        pulse = 0.75 + 0.25 * math.sin(t * 3.0)
        # soft glow under the gem
        glow = np.zeros_like(roi)
        cv2.fillPoly(glow, [to_i(_rounded_diamond(seal, gem_s * 1.9, roll))], GEM_GLOW, AA, shift=S)
        sig = max(1.0, gem_s * 0.6)
        glow = cv2.GaussianBlur(glow, (0, 0), sig)
        cv2.fillPoly(paint, [to_i(gem)], GEM, AA, shift=S)
        facet = _rounded_diamond(seal + (vup / np.linalg.norm(vup)) * gem_s * 0.18, gem_s * 0.55, roll)
        cv2.fillPoly(paint, [to_i(facet)], GEM_LIGHT, AA, shift=S)
        cv2.polylines(paint, [to_i(gem)], True, GEM_INK, ink, AA, shift=S)
        cv2.fillPoly(mask, [to_i(gem)], 255, AA, shift=S)
        cv2.polylines(mask, [to_i(gem)], True, 255, ink, AA, shift=S)

    a = (mask.astype(np.float32) / 255.0 * 0.93)[..., None]
    roi[:] = (roi * (1 - a) + paint * a).astype(np.uint8)
    if gem_s > 1:
        cv2.add(roi, cv2.convertScaleAbs(glow, alpha=0.6 * pulse), dst=roi)

    if age < 0.6:                             # activation flash from the gem
        k = 1 - age / 0.6
        r = int(unit * (0.2 + 1.2 * age / 0.6))
        flash = np.zeros_like(roi)
        cv2.circle(flash, tuple(np.round(seal - o).astype(int)), r,
                   tuple(int(c * k) for c in GEM_GLOW), max(2, int(unit * 0.04)), AA)
        cv2.add(roi, cv2.GaussianBlur(flash, (0, 0), max(1.0, unit * 0.03)), dst=roi)


def _rounded_diamond(c, s, roll_deg, n=6):
    """Square rotated 45 deg (+ head roll) with rounded corners."""
    pts = []
    for k in range(4):
        ang = math.radians(roll_deg + 90 * k - 90)
        corner = np.array([math.cos(ang), math.sin(ang)]) * s
        for j in range(n):
            a2 = ang + math.radians((j / (n - 1) - 0.5) * 70)
            pts.append(corner * 0.82 + np.array([math.cos(a2), math.sin(a2)]) * s * 0.18)
    return np.array(pts, np.float32) + c

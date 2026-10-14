"""Chidori lightning, modelled on animations/videoplayback.mp4.

What the reference shows, and how each part is reproduced:
  * small white-hot core with a cyan halo      -> core disc + blurred halo
  * 3-5 long bolts radiating from the core      -> midpoint-displacement bolts
  * bolts thick at the root, hair-thin at tips  -> per-chunk tapered thickness
  * many fine forks and filaments               -> branches + sub-branches
  * dark-blue haze around the thick bolts       -> wide, heavily blurred pass
  * shape changes every 1-2 frames              -> regenerated at REGEN_HZ
  * tiny streak sparks flying outward           -> simple particle system

Bolts are stored relative to the palm centre, in palm units, so they stay
attached to a moving hand between regenerations.
"""
import math

import cv2
import numpy as np

from utils.helpers import add_layer, soft_blur

AA = cv2.LINE_AA
BOLT = (235, 180, 80)         # BGR, sampled cyan of the reference bolts
BOLT_CORE = (255, 245, 225)
HAZE = (130, 70, 15)
SPARK = (255, 220, 150)


class ChidoriFX:
    REGEN_HZ = 15

    def __init__(self, rng):
        self.rng = rng
        self.slot = None
        self.bolts = []           # list of (points (N,2) palm units, root width, is_main)
        self.sparks = np.zeros((0, 5), np.float32)   # x, y, vx, vy, life (palm units)
        self.intensity = 1.0
        self.t_prev = None

    # ------------------------------------------------------------ geometry
    def _bolt(self, p0, p1, disp, depth):
        pts = np.array([p0, p1], np.float32)
        for _ in range(depth):
            d = np.diff(pts, axis=0)
            n = np.stack([-d[:, 1], d[:, 0]], 1)
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-6
            mids = 0.5 * (pts[:-1] + pts[1:]) + n * self.rng.normal(0, disp, (len(d), 1))
            out = np.empty((len(pts) + len(mids), 2), np.float32)
            out[0::2], out[1::2] = pts, mids
            pts, disp = out, disp * 0.55
        return pts

    def _regenerate(self, tips):
        rng = self.rng
        bolts = []
        n_main = int(rng.integers(3, 6))
        base = rng.uniform(0, 2 * np.pi)
        dirs = [base + k * 2 * np.pi / n_main + rng.normal(0, 0.35) for k in range(n_main)]
        # one bolt often runs out along a finger, as in the reference
        if len(tips) and rng.random() < 0.6:
            tip = tips[int(rng.integers(len(tips)))]
            dirs[0] = math.atan2(tip[1], tip[0]) + rng.normal(0, 0.2)
        for a in dirs:
            L = rng.uniform(1.2, 2.4)
            end = np.array([math.cos(a), math.sin(a)]) * L
            main = self._bolt((0.0, 0.0), end, 0.13 * L, 6)
            bolts.append((main, 0.055, True))
            for _ in range(int(rng.integers(3, 8))):
                i = int(rng.integers(len(main) // 6, len(main) * 5 // 6))
                p = main[i]
                d = main[min(i + 2, len(main) - 1)] - main[max(i - 2, 0)]
                ang = math.atan2(d[1], d[0]) + rng.choice([-1, 1]) * rng.uniform(0.35, 1.0)
                bl = (1 - i / len(main)) * L * rng.uniform(0.3, 0.7)
                br = self._bolt(p, p + np.array([math.cos(ang), math.sin(ang)]) * bl, 0.15 * bl, 4)
                bolts.append((br, 0.055 * (1 - i / len(main)) * 0.6, False))
                if rng.random() < 0.5:                      # filament off the branch
                    j = int(rng.integers(len(br) // 3, len(br)))
                    a2 = ang + rng.choice([-1, 1]) * rng.uniform(0.4, 1.0)
                    fl = bl * rng.uniform(0.25, 0.5)
                    q = br[j]
                    bolts.append((self._bolt(q, q + np.array([math.cos(a2), math.sin(a2)]) * fl,
                                             0.15 * fl, 3), 0.0, False))
        self.bolts = bolts
        self.intensity = rng.uniform(0.85, 1.1)

    def _spawn_sparks(self, n):
        a = self.rng.uniform(0, 2 * np.pi, n)
        v = self.rng.uniform(4.0, 9.0, n)
        r = self.rng.uniform(0.0, 0.4, n)
        new = np.stack([np.cos(a) * r, np.sin(a) * r, np.cos(a) * v, np.sin(a) * v,
                        self.rng.uniform(0.2, 0.45, n)], 1).astype(np.float32)
        self.sparks = np.vstack([self.sparks, new])

    # ------------------------------------------------------------ draw
    def draw(self, frame, center, palm, tips, t, env):
        dt = 0.0 if self.t_prev is None else min(0.1, t - self.t_prev)
        self.t_prev = t
        slot = int(t * self.REGEN_HZ)
        if slot != self.slot:
            self.slot = slot
            self._regenerate((np.asarray(tips) - center) / palm)

        # sparks: integrate, age, spawn
        if dt > 0:
            self.sparks[:, 0:2] += self.sparks[:, 2:4] * dt
            self.sparks[:, 4] -= dt
            self.sparks = self.sparks[self.sparks[:, 4] > 0]
        self._spawn_sparks(int(self.rng.integers(1, 4)))

        # Layer covers the effect's reach, clipped to the frame (the bolts
        # can reach 2.4 palms, so an unclipped layer is bigger than the frame).
        reach = palm * 2.6
        fh, fw = frame.shape[:2]
        x0, y0 = max(0, int(center[0] - reach)), max(0, int(center[1] - reach))
        x1, y1 = min(fw, int(center[0] + reach)), min(fh, int(center[1] + reach))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return
        w, h = x1 - x0, y1 - y0
        c = np.array([center[0] - x0, center[1] - y0], np.float32)
        to_px = lambda p: p * palm + c
        q = 4                                           # haze is drawn at 1/4 res
        haze = np.zeros((max(2, h // q), max(2, w // q), 3), np.uint8)
        bolts = np.zeros((h, w, 3), np.uint8)
        S = 3                                           # sub-pixel shift
        ipts = lambda p: np.round(p * (1 << S)).astype(np.int32)

        for pts, w_root, is_main in self.bolts:
            px = to_px(pts)
            if is_main:
                cv2.polylines(haze, [ipts(px / q)], False, HAZE, max(1, int(palm * 0.2 / q)), AA, shift=S)
            n = len(px)
            chunks = 5 if is_main else (3 if w_root > 0 else 1)
            for k in range(chunks):
                a, b = k * (n - 1) // chunks, (k + 1) * (n - 1) // chunks + 1
                taper = (1 - k / chunks) ** 0.9
                th = max(1, int(round(w_root * palm * taper)))
                cv2.polylines(bolts, [ipts(px[a:b])], False, BOLT, th, AA, shift=S)
                if is_main and k < 2:                    # white-hot centre near the root
                    cv2.polylines(bolts, [ipts(px[a:b])], False, BOLT_CORE,
                                  max(1, th // 3), AA, shift=S)

        # core + halo
        ci = tuple(np.round(c).astype(int))
        cv2.circle(haze, (ci[0] // q, ci[1] // q), max(1, int(palm * 0.45 / q)), (200, 140, 40), -1, AA)
        cv2.circle(bolts, ci, max(2, int(palm * 0.1)), BOLT_CORE, -1, AA)

        for x, y, vx, vy, life in self.sparks:
            p0 = to_px(np.array([x, y]))
            p1 = to_px(np.array([x - vx * 0.035, y - vy * 0.035]))
            k = min(1.0, life / 0.2)
            cv2.line(bolts, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)),
                     tuple(int(ch * k) for ch in SPARK), 1, AA)

        haze = cv2.resize(soft_blur(haze, max(0.6, palm * 0.22 / q)), (w, h),
                          interpolation=cv2.INTER_LINEAR)
        layer = cv2.add(haze, cv2.addWeighted(bolts, 1.0, soft_blur(bolts, max(1.0, palm * 0.05)), 0.9, 0))
        k = env * self.intensity
        if abs(k - 1.0) > 1e-3:
            layer = cv2.convertScaleAbs(layer, alpha=k)
        add_layer(frame, layer, x0, y0)

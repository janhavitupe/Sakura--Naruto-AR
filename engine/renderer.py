"""AR Rendering Engine (OpenCV).

Input : frame, active jutsu, hand features by label, face, state, temporal
        state, fps.
Output: frame with effects + HUD drawn in place.

Performance rules followed by every effect:
  * Draw into a small ROI-sized layer, never a full-frame overlay.
  * Energy effects use saturating additive blending (cv2.add) + a blurred
    copy for bloom; this is cheap and looks emissive.
  * Tiny details (eyes are ~6-10 px radius on a 640x480 webcam) are drawn
    as a high-resolution sprite and downsampled with INTER_AREA, which is
    effectively free supersampled antialiasing.
  * Per-activation randomness (swirl tilts, vein shapes) is generated once
    and cached by the jutsu's start time; per-frame randomness is only used
    where flicker is the point (lightning, flames).
"""
import math

import cv2
import numpy as np

from config import settings as S
from engine.byakugou_seal import draw_seal
from engine.chidori_fx import ChidoriFX
from engine.jutsu_engine import DISPLAY
from modules.face_tracker import draw_face
from modules.hand_tracker import draw_hands
from utils.helpers import add_layer, alpha_layer, glow, rotate

AA = cv2.LINE_AA
FONT = cv2.FONT_HERSHEY_SIMPLEX
CHAKRA_BLUE = (255, 170, 60)


def P(v):
    return int(round(float(v[0]))), int(round(float(v[1])))


def envelope(age, duration, fade_in=0.15, fade_out=0.4):
    if age < fade_in:
        return age / fade_in
    return float(np.clip((duration - age) / fade_out, 0.0, 1.0))


class Renderer:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.cache = {}           # per-activation random params
        self.last_hand = {}       # label -> last HandFeatures (dropout cover)
        self.sprite_size = 96

    def _params(self, j, make):
        key = (j.name, j.started)
        if key not in self.cache:
            if len(self.cache) > 32:
                self.cache.clear()
            self.cache[key] = make()
        return self.cache[key]

    # ================================================================ main
    def render(self, frame, active, feats, face, state, ts, fps, t, debug=False):
        for label, f in feats.items():
            self.last_hand[label] = f

        if debug:
            draw_hands(frame, [f.hand for f in feats.values()],
                       {k: "palm" if f.palm_facing else "back" for k, f in feats.items()})
            if face is not None:
                draw_face(frame, face)

        for j in active:
            if j.name == "byakugou" and face is not None:
                self._byakugou(frame, j, face, t)
        for j in active:
            if j.name == "dojutsu" and face is not None:
                self._dojutsu(frame, j, face, t)
            elif j.name == "rasengan":
                self._rasengan(frame, j, feats, t)
            elif j.name == "chidori":
                self._chidori(frame, j, t)
            elif j.name == "lion_fist":
                self._lion_fist(frame, j, t)

        self._hold_feedback(frame, ts, feats, face)
        self._hud(frame, active, state, fps, t)
        if debug:
            self._debug(frame, ts)
        return frame

    # ================================================================ Rasengan
    def _rasengan(self, frame, j, feats, t):
        fs = list(feats.values())
        if len(fs) < 2:
            return
        c = 0.5 * (fs[0].center + fs[1].center)
        palm = 0.5 * (fs[0].palm_size + fs[1].palm_size)
        charge = j.data["charge"]
        ease = 1 - (1 - charge) ** 2
        r = palm * (S.RASENGAN_MIN_RADIUS + (S.RASENGAN_MAX_RADIUS - S.RASENGAN_MIN_RADIUS) * ease)
        r = max(r, 6.0)

        swirls = self._params(j, lambda: [
            (self.rng.uniform(0, 180), self.rng.uniform(0.15, 0.9),
             self.rng.uniform(300, 700) * self.rng.choice([-1, 1]),
             self.rng.uniform(120, 260), self.rng.uniform(0, 360))
            for _ in range(12)])

        half = int(r * 1.8)
        layer = np.zeros((2 * half, 2 * half, 3), np.uint8)
        cc = (half, half)
        cv2.circle(layer, cc, int(r), (95, 40, 5), -1, AA)                # body
        th = max(1, int(r * 0.05))
        for tilt, ratio, speed, arc, phase in swirls:
            start = (phase + t * speed) % 360
            cv2.ellipse(layer, cc, (int(r * 0.95), max(1, int(r * ratio))),
                        tilt + t * speed * 0.15, start, start + arc, (230, 160, 70), th, AA)
        cv2.circle(layer, cc, int(r * 0.3), (255, 220, 170), -1, AA)     # core
        cv2.circle(layer, cc, int(r), (200, 120, 40), max(1, th), AA)    # rim
        for k in range(10):                                              # orbiting wind
            a = t * 7 + k * 0.628
            rr = r * (1.12 + 0.12 * math.sin(t * 5 + k))
            p = (half + rr * math.cos(a), half + rr * math.sin(a) * 0.55)
            cv2.circle(layer, P(p), max(1, th), (230, 180, 110), -1, AA)
        layer = glow(layer, max(2.0, r * 0.25), 0.9)
        add_layer(frame, layer, c[0] - half, c[1] - half)

    # ================================================================ Chidori
    def _chidori(self, frame, j, t):
        f = self.last_hand.get(j.data["hand"])
        if f is None:
            return
        fx = self._params(j, lambda: ChidoriFX(self.rng))
        env = envelope(j.age(t), S.CHIDORI_DURATION_S)
        fx.draw(frame, f.center, f.palm_size, f.hand.landmarks[[4, 8, 12, 16, 20], :2], t, env)

    # ================================================================ Lion Fist
    def _lion_fist(self, frame, j, t):
        f = self.last_hand.get(j.data["hand"])
        if f is None:
            return
        age = j.age(t)
        env = envelope(age, S.LION_FIST_DURATION_S)
        palm = f.palm_size
        pts = f.hand.landmarks[:, :2]
        x0, y0 = pts.min(0) - palm * 1.3
        x1, y1 = pts.max(0) + palm * 1.3
        w, h = int(x1 - x0), int(y1 - y0)
        origin = np.array([x0, y0])
        local = (pts - origin).astype(np.int32)
        hull = cv2.convexHull(local)

        # The aura is soft, so build it at 1/4 resolution and upscale.
        ds = 4
        mask = np.zeros((max(1, h // ds), max(1, w // ds)), np.uint8)
        cv2.fillConvexPoly(mask, hull // ds, 255, AA)
        pulse = 1.0 + 0.15 * math.sin(t * 14)
        k = int(palm * 0.4 / ds) * 2 + 1
        dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        sig = max(0.5, palm * 0.18 * pulse / ds)
        aura = cv2.GaussianBlur(dil, (0, 0), sig).astype(np.float32) / 255.0
        inner = cv2.GaussianBlur(mask, (0, 0), max(0.5, sig * 0.4)).astype(np.float32) / 255.0
        ring = np.clip(aura * 1.3 - inner * 0.8, 0, 1)
        ring = cv2.resize(ring, (w, h), interpolation=cv2.INTER_LINEAR)
        layer = (ring[..., None] * np.array([235, 150, 50], np.float32)).astype(np.uint8)

        # flame tongues licking outward from the hull
        c = f.center - origin
        flames = np.zeros_like(layer)
        for p in hull[:, 0][self.rng.integers(0, len(hull), 14)]:
            d = p - c
            d = d / (np.linalg.norm(d) + 1e-6)
            L = palm * self.rng.uniform(0.25, 0.8)
            q = p + d * L + self.rng.normal(0, palm * 0.08, 2)
            cv2.line(flames, P(p), P(q), (230, 180, 110), max(1, int(palm * 0.07)), AA)
        layer = cv2.add(layer, glow(flames, max(2.0, palm * 0.08), 1.0))

        if age < 0.5:                                  # opening shockwave
            rr = int(palm * (0.8 + age * 5))
            a = 1 - age / 0.5
            cv2.circle(layer, P(c), rr, (int(255 * a), int(230 * a), int(190 * a)),
                       max(1, int(palm * 0.08 * a)), AA)
        if env < 0.999:
            layer = cv2.convertScaleAbs(layer, alpha=env)
        add_layer(frame, layer, x0, y0)

    # ================================================================ Dojutsu
    def _sharingan_sprite(self, t):
        n = self.sprite_size
        R, c = n // 2 - 2, (n // 2, n // 2)
        spr = np.zeros((n, n, 4), np.uint8)
        cv2.circle(spr, c, R, (25, 25, 205, 255), -1, AA)
        cv2.circle(spr, c, R - 4, (10, 10, 140, 255), 5, AA)
        cv2.circle(spr, c, R, (0, 0, 0, 255), 3, AA)
        cv2.circle(spr, c, int(R * 0.56), (0, 0, 0, 255), 2, AA)
        cv2.circle(spr, c, int(R * 0.22), (0, 0, 0, 255), -1, AA)
        for k in range(3):
            a = math.radians(t * 140 + k * 120)
            pos = (c[0] + R * 0.56 * math.cos(a), c[1] + R * 0.56 * math.sin(a))
            cv2.circle(spr, P(pos), int(R * 0.16), (0, 0, 0, 255), -1, AA)
            deg = math.degrees(a)
            cv2.ellipse(spr, c, (int(R * 0.62), int(R * 0.62)), 0, deg, deg + 40,
                        (0, 0, 0, 255), int(R * 0.09), AA)
        return spr

    def _byakugan_sprite(self):
        n = self.sprite_size
        R, c = n // 2 - 2, (n // 2, n // 2)
        spr = np.zeros((n, n, 4), np.uint8)
        cv2.circle(spr, c, R, (238, 228, 235, 230), -1, AA)
        cv2.circle(spr, c, int(R * 0.75), (215, 200, 215, 230), 2, AA)
        cv2.circle(spr, c, int(R * 0.2), (200, 185, 200, 230), -1, AA)
        return spr

    def _place_eye(self, frame, spr, center, radius, contour):
        d = max(3, int(math.ceil(radius * 2.1)))
        small = cv2.resize(spr, (d, d), interpolation=cv2.INTER_AREA)
        x0, y0 = int(round(center[0] - d / 2)), int(round(center[1] - d / 2))
        eye_mask = np.zeros((d, d), np.uint8)
        cv2.fillPoly(eye_mask, [contour - [x0, y0]], 255, AA)     # eyelids occlude
        alpha = (small[..., 3].astype(np.float32) / 255.0) * (eye_mask.astype(np.float32) / 255.0)
        alpha_layer(frame, small[..., :3], alpha, x0, y0)

    def _dojutsu(self, frame, j, face, t):
        eye = j.data["eye"]
        age = j.age(t)
        spr = self._sharingan_sprite(t) if eye == "sharingan" else self._byakugan_sprite()
        for c, r, contour in face.eyes():
            if age < 0.3:                                  # iris "spins up"
                r *= 0.4 + 2.0 * age
            self._place_eye(frame, spr, c, r, contour)
            if eye == "sharingan":                          # faint red bloom
                g = int(r * 4)
                layer = np.zeros((2 * g, 2 * g, 3), np.uint8)
                cv2.circle(layer, (g, g), max(1, int(r * 1.2)), (10, 10, 120), -1, AA)
                add_layer(frame, glow(layer, max(1.5, r * 0.8), 1.0) // 2, c[0] - g, c[1] - g)
        if eye == "byakugan":
            self._veins(frame, j, face, t)

    def _veins(self, frame, j, face, t):
        """Bulging veins radiating from the eye corners toward the temples."""
        templates = self._params(j, lambda: [
            [(self.rng.uniform(-70, 70), [self.rng.uniform(-35, 35) for _ in range(5)])
             for _ in range(5)] for _ in range(2)])
        grow = min(1.0, j.age(t) / 0.6)
        lines = []
        eyes = list(face.eyes())
        for (c, r, contour), veins in zip(eyes, templates):
            ew = float(np.linalg.norm(contour[0] - contour[8]))
            outward = np.sign(c[0] - face.anchor[0]) or 1.0
            base_dir = rotate(np.array([outward, 0.0]), face.roll)
            corner = contour[0] if abs(contour[0][0] - face.anchor[0]) > abs(contour[8][0] - face.anchor[0]) else contour[8]
            for spread, wiggles in veins:
                p = corner.astype(np.float32) + base_dir * ew * 0.1
                d = rotate(base_dir, spread * outward)
                pts = [p.copy()]
                for wig in wiggles[:max(1, int(len(wiggles) * grow))]:
                    d = rotate(d, wig)
                    p = p + d * ew * 0.2
                    pts.append(p.copy())
                lines.append(np.array(pts, np.int32))
        if not lines:
            return
        allp = np.vstack(lines)
        x0, y0 = allp.min(0) - 4
        x1, y1 = allp.max(0) + 4
        fh, fw = frame.shape[:2]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(fw, x1), min(fh, y1)
        if x1 <= x0 or y1 <= y0:
            return
        roi = frame[y0:y1, x0:x1]
        dark = roi.copy()
        shifted = [l - [x0, y0] for l in lines]
        cv2.polylines(dark, shifted, False, (55, 35, 75), max(1, int(face.width * 0.013)), AA)
        cv2.addWeighted(dark, 0.75, roi, 0.25, 0, dst=roi)

    # ================================================================ Byakugou
    def _byakugou(self, frame, j, face, t):
        draw_seal(frame, face, j.age(t), t, S.BYAKUGOU_SPREAD_S, body=S.BYAKUGOU_BODY_MARKS)

    # ================================================================ HUD
    @staticmethod
    def _text(frame, text, org, scale=0.5, color=(255, 255, 255), thick=1):
        cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), thick + 2, AA)
        cv2.putText(frame, text, org, FONT, scale, color, thick, AA)

    def _hold_feedback(self, frame, ts, feats, face):
        """Progress ring while a gesture is being held, so the user knows
        it was recognized and how long to keep it."""
        for key, p in ts.progress.items():
            if not 0.05 < p < 1.0:
                continue
            base, _, label = key.partition(":")
            if label and label in feats:
                pos = feats[label].center
            elif base in ("rasengan_cup", "prayer") and len(feats) == 2:
                pos = np.mean([f.center for f in feats.values()], axis=0)
            else:
                continue
            cv2.ellipse(frame, P(pos), (26, 26), -90, 0, 360 * p, (255, 255, 255), 2, AA)
            self._text(frame, base.replace("_", " "), P(pos + [-30, 44]), 0.42)

    def _infinite_bar(self, bw, bh):
        if "bar" not in self.__dict__:
            hue = (130 + 30 * np.sin(np.linspace(0, 2 * np.pi, bw))).astype(np.uint8)
            hsv = np.stack([hue, np.full(bw, 200, np.uint8), np.full(bw, 255, np.uint8)], -1)
            row = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)
            self.bar = np.repeat(row, bh, axis=0)
        return self.bar

    def _hud(self, frame, active, state, fps, t):
        h, w = frame.shape[:2]
        # chakra bar
        x, y, bw, bh = 14, 14, 220, 16
        cv2.rectangle(frame, (x - 2, y - 2), (x + bw + 2, y + bh + 2), (0, 0, 0), -1)
        if state.infinite_chakra:
            shift = int(t * 80) % bw
            frame[y:y + bh, x:x + bw] = np.roll(self._infinite_bar(bw, bh), shift, axis=1)
            label = "CHAKRA  INFINITE (Byakugou)"
        else:
            frac = state.chakra / S.CHAKRA_MAX
            col = CHAKRA_BLUE if frac > 0.3 else (0, 80, 255)
            cv2.rectangle(frame, (x, y), (x + int(bw * frac), y + bh), col, -1)
            label = f"CHAKRA  {state.chakra:5.1f}"
        self._text(frame, label, (x, y + bh + 18), 0.5)

        # cooldowns
        yy = h - 14 - 18 * 4
        for name in ("rasengan", "chidori", "lion_fist", "dojutsu", "byakugou"):
            left = state.cooldown_left(name, t)
            running = any(j.name == name for j in active)
            if running:
                txt, col = "ACTIVE", (0, 255, 255)
            elif left > 0:
                txt, col = f"{int(left // 60)}:{int(left % 60):02d}", (0, 140, 255)
            else:
                txt, col = "ready", (120, 255, 120)
            self._text(frame, f"{DISPLAY[name]:<14}", (14, yy), 0.45)
            self._text(frame, txt, (140, yy), 0.45, col)
            yy += 18

        self._text(frame, f"{fps:4.1f} FPS", (w - 100, 26), 0.55,
                   (120, 255, 120) if fps >= 20 else (0, 0, 255))

        names = [DISPLAY[j.name] + (f" ({j.data['eye'].title()})" if j.name == "dojutsu" else "")
                 for j in active]
        if names:
            banner = "  |  ".join(names)
            (tw, _), _ = cv2.getTextSize(banner, FONT, 0.6, 2)
            self._text(frame, banner, (max(8, (w - tw) // 2), 76), 0.6, (255, 255, 255), 2)

    def draw_messages(self, frame, messages):
        h, w = frame.shape[:2]
        y = h - 40
        for text, _, color in messages[-3:]:
            (tw, _), _ = cv2.getTextSize(text, FONT, 0.6, 2)
            self._text(frame, text, ((w - tw) // 2, y), 0.6, color, 2)
            y -= 26

    def _debug(self, frame, ts):
        w = frame.shape[1]
        y = 56
        lines = [f"stable {ts.stable}  head {ts.head_speed:.2f} fw/s",
                 f"browDown {ts.brow_down:.2f} squint {ts.eye_squint:.2f} smile {ts.smile:.2f}"]
        lines += [f"{k:<18}{v:.2f}{' HELD' if k in ts.held else ''}"
                  for k, v in sorted(ts.conf.items()) if v > 0.02]
        for line in lines:
            self._text(frame, line, (w - 260, y), 0.42, (200, 255, 255))
            y += 16

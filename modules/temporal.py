"""Temporal logic: landmark filtering, confidence smoothing, hold detection,
tap detection and motion stability.

Everything is time-based (seconds), never frame-based, so behaviour stays the
same whether the loop runs at 15 or 60 FPS.
"""
import math
from collections import deque

import numpy as np

from config import settings as S


# ------------------------------------------------------------------ filters
class OneEuroFilter:
    """One-Euro filter over an arbitrary numpy array (e.g. 21x3 landmarks).

    Adaptive low-pass: heavy smoothing when the signal is slow (kills jitter),
    light smoothing when it moves fast (kills lag)."""

    def __init__(self, min_cutoff=S.ONE_EURO_MIN_CUTOFF, beta=S.ONE_EURO_BETA,
                 d_cutoff=S.ONE_EURO_D_CUTOFF):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = None

    def __call__(self, x, t):
        if self.x_prev is None or t <= self.t_prev:
            self.x_prev, self.dx_prev, self.t_prev = x, np.zeros_like(x), t
            return x
        dt = t - self.t_prev
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


# ------------------------------------------------------------------ gestures
class GestureTracker:
    """Turns noisy per-frame booleans into a stable, held gesture.

    Pipeline per gesture:
      raw bool -> EMA confidence (time-constant based)
               -> hysteresis (CONF_ON / CONF_OFF) => `active`
               -> hold timer => `held` once active for `hold_s`
               -> grace period: short dropouts don't reset the hold
    """

    def __init__(self, name, hold_s):
        self.name = name
        self.hold_s = hold_s
        self.conf = 0.0
        self.active = False
        self.active_since = None
        self.last_seen = -1e9
        self.held = False
        self.just_held = False       # rising edge, true for one update
        self.just_released = False   # falling edge of `held`

    def update(self, raw, t, dt):
        alpha = 1.0 - math.exp(-dt / S.SMOOTHING_TAU_S) if dt > 0 else 1.0
        self.conf += alpha * ((1.0 if raw else 0.0) - self.conf)
        if raw:
            self.last_seen = t

        was_held = self.held
        if not self.active and self.conf >= S.CONF_ON:
            self.active = True
            self.active_since = t
        elif self.active and self.conf <= S.CONF_OFF \
                and t - self.last_seen > S.RELEASE_GRACE_S:
            self.active = False
            self.active_since = None

        self.held = self.active and (t - self.active_since) >= self.hold_s
        self.just_held = self.held and not was_held
        self.just_released = was_held and not self.held
        return self

    def progress(self, t):
        """0..1 progress toward `held`, for UI feedback."""
        if not self.active or self.active_since is None:
            return 0.0
        return min(1.0, (t - self.active_since) / self.hold_s)


class TapDetector:
    """Detects a tap: contact that starts, lasts [min, max] seconds, then
    ends. A long press beyond TAP_HOLD_COUNTS_AS_TAP_S also counts, so users
    who 'hold' their finger on the forehead are not punished."""

    def __init__(self):
        self.contact_since = None
        self.fired_this_contact = False

    def update(self, touching, t):
        tapped = False
        if touching:
            if self.contact_since is None:
                self.contact_since = t
                self.fired_this_contact = False
            elif not self.fired_this_contact and \
                    t - self.contact_since >= S.TAP_HOLD_COUNTS_AS_TAP_S:
                tapped = True
                self.fired_this_contact = True
        elif self.contact_since is not None:
            dur = t - self.contact_since
            if not self.fired_this_contact and \
                    S.TAP_MIN_CONTACT_S <= dur <= S.TAP_MAX_CONTACT_S:
                tapped = True
            self.contact_since = None
        return tapped


class StabilityTracker:
    """Measures 'low body movement' from the head anchor position.

    Keeps a time window of (t, position, scale) and reports mean speed in
    face-widths/second, which is distance- and resolution-invariant."""

    def __init__(self, window_s=S.STABILITY_WINDOW_S):
        self.window_s = window_s
        self.samples = deque()

    def update(self, pos, scale, t):
        if pos is None:
            self.samples.clear()
            return
        self.samples.append((t, np.asarray(pos, dtype=np.float32), scale))
        while self.samples and t - self.samples[0][0] > self.window_s:
            self.samples.popleft()

    def speed(self):
        if len(self.samples) < 5:
            return float("inf")
        path = 0.0
        for (t0, p0, _), (t1, p1, _) in zip(self.samples, list(self.samples)[1:]):
            path += float(np.linalg.norm(p1 - p0))
        span = self.samples[-1][0] - self.samples[0][0]
        scale = np.mean([s for _, _, s in self.samples])
        return path / max(span, 1e-3) / max(scale, 1e-3)

    def is_stable(self):
        covered = self.samples and \
            self.samples[-1][0] - self.samples[0][0] >= 0.8 * self.window_s
        return bool(covered) and self.speed() <= S.STABILITY_MAX_SPEED


class RollingMean:
    """Time-windowed mean of a dict of scalar signals (expression scores)."""

    def __init__(self, window_s):
        self.window_s = window_s
        self.samples = deque()

    def update(self, values, t):
        self.samples.append((t, values))
        while self.samples and t - self.samples[0][0] > self.window_s:
            self.samples.popleft()

    def mean(self, key):
        vals = [v.get(key, 0.0) for _, v in self.samples]
        return float(np.mean(vals)) if vals else 0.0


class ExpressionBaseline:
    """Per-user neutral face. Blendshape resting values differ a lot between
    people (some faces read browDown 0.3 at rest), so expressions are judged
    relative to a slow EMA of the user's own face. The EMA is frozen while
    the user is making a sign, so the expression being measured never leaks
    into its own baseline."""

    def __init__(self, tau_s=S.EXPRESSION_BASELINE_TAU_S):
        self.tau_s = tau_s
        self.values = None

    def update(self, expr, dt, frozen):
        if self.values is None:
            self.values = dict(expr)
            return
        if frozen or dt <= 0:
            return
        a = 1.0 - math.exp(-dt / self.tau_s)
        for k, v in expr.items():
            self.values[k] = self.values.get(k, v) + a * (v - self.values.get(k, v))

    def relative(self, key, value):
        if self.values is None:
            return value
        return max(0.0, value - self.values.get(key, 0.0))


# ------------------------------------------------------------------ facade
class TemporalLogic:
    """Owns all temporal state.

    Input : per-frame raw gesture dict from GestureClassifier, face data, t.
    Output: TemporalState with held gestures, rising/falling edges, hold
            progress, head stability, smoothed + baseline-relative expression."""

    def __init__(self):
        self.trackers = {}
        self.tap = TapDetector()
        self.stability = StabilityTracker()
        self.expressions = RollingMean(S.DOJUTSU_SAMPLE_S)
        self.baseline = ExpressionBaseline()
        self.t_prev = None

    def _tracker(self, key):
        if key not in self.trackers:
            base = key.split(":")[0]
            self.trackers[key] = GestureTracker(key, S.HOLD_S.get(base, S.DEFAULT_HOLD_S))
        return self.trackers[key]

    def update(self, raw, face, t):
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = t

        # Head stability is measured from the nose bridge + face width.
        if face is not None:
            self.stability.update(face.anchor, face.width, t)
            self.expressions.update(face.expression, t)
            signing = "prayer" in self.trackers and self.trackers["prayer"].active
            self.baseline.update(face.expression, dt, frozen=signing)
        else:
            self.stability.update(None, 0, t)

        tapped = self.tap.update(raw.get("forehead_touch", False), t)

        # Every key seen in this or previous frames gets updated, so trackers
        # decay naturally when their gesture disappears.
        keys = set(self.trackers) | {k for k, v in raw.items()
                                     if k != "forehead_touch" and v}
        for k in keys:
            self._tracker(k).update(bool(raw.get(k, False)), t, dt)

        return TemporalState(
            held={k for k, tr in self.trackers.items() if tr.held},
            just_held={k for k, tr in self.trackers.items() if tr.just_held},
            just_released={k for k, tr in self.trackers.items() if tr.just_released},
            progress={k: tr.progress(t) for k, tr in self.trackers.items() if tr.active},
            conf={k: tr.conf for k, tr in self.trackers.items()},
            forehead_tap=tapped,
            stable=self.stability.is_stable(),
            head_speed=self.stability.speed(),
            # expression scores relative to the user's neutral face
            brow_down=self.baseline.relative("brow_down", self.expressions.mean("brow_down")),
            eye_squint=self.baseline.relative("eye_squint", self.expressions.mean("eye_squint")),
            eye_wide=self.baseline.relative("eye_wide", self.expressions.mean("eye_wide")),
            smile=self.expressions.mean("smile"),
        )


class TemporalState:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def held_any(self, base):
        """Return the first held key for a gesture base name, e.g. 'claw' ->
        'claw:Right'."""
        for k in self.held:
            if k.split(":")[0] == base:
                return k
        return None

    def just_held_any(self, base):
        for k in self.just_held:
            if k.split(":")[0] == base:
                return k
        return None

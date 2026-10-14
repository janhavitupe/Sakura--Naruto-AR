"""Jutsu Engine: maps (held gestures + face + state) -> jutsu lifecycle.

Input : TemporalState, HandFeatures by label, Face | None, StateManager, t.
Output: list of active Jutsu objects (consumed by the renderer) and short
        on-screen messages (blocked / cooldown feedback).

Lifecycle per jutsu type:
  instant    (Chidori, Lion Fist): fire on the gesture's rising `held` edge,
             pay once, show the effect for a fixed duration.
  continuous (Rasengan): start on the edge, drain chakra while the gesture is
             held, end on release or empty chakra.
  mode       (Dojutsu, Byakugou): start on a trigger, persist on their own
             until a timeout, toggle-off or empty chakra.

Rule: only one *hand* jutsu (Rasengan / Chidori / Lion Fist) at a time, since
the hands can't make two signs at once. Modes stack with hand jutsu.
"""
from config import settings as S

HAND_JUTSU = ("rasengan", "chidori", "lion_fist")
DISPLAY = {
    "rasengan": "Rasengan", "chidori": "Chidori", "lion_fist": "Lion Fist",
    "dojutsu": "Dojutsu", "byakugou": "Byakugou Seal",
}


class Jutsu:
    def __init__(self, name, t, **data):
        self.name = name
        self.started = t
        self.ends_at = None          # for timed jutsu
        self.data = data             # hand label, eye type, charge, anchor...

    def age(self, t):
        return t - self.started


class JutsuEngine:
    def __init__(self, state):
        self.state = state
        self.active = {}             # name -> Jutsu
        self.messages = []           # (text, expires_at, color)
        self.t_prev = None

    # ------------------------------------------------------------ helpers
    def _msg(self, text, t, color=(255, 255, 255), dur=1.6):
        self.messages = [m for m in self.messages if m[0] != text]
        self.messages.append((text, t + dur, color))

    def _try_start(self, name, cost, t):
        """Shared gate: cooldown, then chakra. Returns True if allowed and paid."""
        st = self.state
        if st.on_cooldown(name, t):
            left = st.cooldown_left(name, t)
            self._msg(f"{DISPLAY[name]} on cooldown {int(left // 60)}:{int(left % 60):02d}",
                      t, (0, 165, 255))
            return False
        if not st.spend(cost):
            self._msg("Not enough chakra", t, (0, 0, 255))
            return False
        return True

    def _end(self, name, t):
        if self.active.pop(name, None) is not None and name != "byakugou":
            self.state.start_cooldown(name, t)

    @property
    def hand_busy(self):
        return any(n in self.active for n in HAND_JUTSU)

    @property
    def idle(self):
        return not self.active or list(self.active) == ["byakugou"]

    # ------------------------------------------------------------ update
    def update(self, ts, feats_by_label, face, t):
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = t
        st = self.state
        self.messages = [m for m in self.messages if m[1] > t]

        self._update_byakugou(ts, face, t)
        self._update_dojutsu(ts, face, t, dt)
        self._update_rasengan(ts, feats_by_label, t, dt)
        self._update_timed(ts, "chidori", "claw", S.COST["chidori"],
                           S.CHIDORI_DURATION_S, t)
        self._update_timed(ts, "lion_fist", "open_palm", S.COST["lion_fist"],
                           S.LION_FIST_DURATION_S, t)

        # Chakra at zero blocks everything, including running jutsu.
        if st.depleted:
            for name in list(self.active):
                if name != "byakugou":
                    self._end(name, t)

        st.tick(t, self.idle, dt)
        return list(self.active.values())

    def _update_byakugou(self, ts, face, t):
        st = self.state
        if "byakugou" in self.active and not st.byakugou_active:
            self._end("byakugou", t)
            self._msg("Byakugou faded", t, (200, 120, 255))
        if ts.forehead_tap and "byakugou" not in self.active:
            if not ts.stable:
                self._msg("Stay still to release the seal", t, (200, 120, 255))
            elif st.on_cooldown("byakugou", t):
                left = st.cooldown_left("byakugou", t)
                self._msg(f"Byakugou on cooldown {int(left // 60)}:{int(left % 60):02d}",
                          t, (0, 165, 255))
            else:
                st.activate_byakugou(t)
                self.active["byakugou"] = Jutsu("byakugou", t)
                self._msg("BYAKUGOU NO IN - infinite chakra", t, (220, 120, 255), 2.5)

    def _update_dojutsu(self, ts, face, t, dt):
        st = self.state
        prayer_edge = "prayer" in ts.just_held
        j = self.active.get("dojutsu")
        if j is not None:
            if prayer_edge:
                self._end("dojutsu", t)               # sign again to release
                return
            if j.age(t) > S.DOJUTSU_MAX_DURATION_S or \
                    not st.drain(S.COST["dojutsu_per_s"], dt):
                self._end("dojutsu", t)
            return
        if not prayer_edge:
            return
        if face is None:
            self._msg("Face not visible", t, (0, 0, 255))
            return
        if self._try_start("dojutsu", S.COST["dojutsu_start"], t):
            eye = classify_dojutsu(ts.brow_down, ts.eye_squint, ts.smile)
            self.active["dojutsu"] = Jutsu("dojutsu", t, eye=eye)
            self._msg(eye.upper() + "!", t, (0, 0, 255) if eye == "sharingan"
                      else (255, 230, 230), 2.0)

    def _update_rasengan(self, ts, feats, t, dt):
        st = self.state
        j = self.active.get("rasengan")
        if j is not None:
            if "rasengan_cup" not in ts.held or len(feats) < 2:
                self._end("rasengan", t)
                return
            if not st.drain(S.COST["rasengan_per_s"], dt):
                self._end("rasengan", t)
                self._msg("Chakra exhausted", t, (0, 0, 255))
                return
            j.data["charge"] = min(1.0, j.data["charge"] + dt / S.RASENGAN_FULL_CHARGE_S)
            return
        if "rasengan_cup" in ts.just_held and not self.hand_busy:
            if self._try_start("rasengan", S.COST["rasengan_start"], t):
                self.active["rasengan"] = Jutsu("rasengan", t, charge=0.0)

    def _update_timed(self, ts, name, gesture, cost, duration, t):
        j = self.active.get(name)
        if j is not None:
            if t >= j.ends_at:
                self._end(name, t)
            return
        key = ts.just_held_any(gesture)
        if key and not self.hand_busy and self._try_start(name, cost, t):
            j = Jutsu(name, t, hand=key.split(":")[1])
            j.ends_at = t + duration
            self.active[name] = j


def classify_dojutsu(brow_down, eye_squint, smile=0.0):
    """Sharingan = focused eyes + furrowed brows, Byakugan = neutral face.

    Inputs are averaged over the last DOJUTSU_SAMPLE_S seconds (while the
    prayer sign was held) and measured relative to the user's neutral face.
    A smile also raises squint/brow scores, so it vetoes Sharingan."""
    if smile > S.SHARINGAN_MAX_SMILE:
        return "byakugan"
    score = 0.6 * brow_down + 0.4 * eye_squint
    if brow_down >= S.SHARINGAN_MIN_BROW_DOWN or score >= S.SHARINGAN_MIN_SCORE:
        return "sharingan"
    return "byakugan"

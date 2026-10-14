"""State Manager: chakra, cooldowns and active modes.

Input : spend / drain / cooldown requests from the JutsuEngine, time.
Output: queries such as can_afford(), cooldown_left(), infinite_chakra.

This module holds *no* gesture logic. It only enforces resource rules, so the
rules stay in one place and are easy to test.
"""
from config import settings as S

STANDARD = "standard"
BYAKUGOU = "byakugou"


class StateManager:
    def __init__(self, demo=False):
        self.chakra = S.CHAKRA_MAX
        self.cooldown_until = {}          # jutsu name -> absolute time
        self.byakugou_until = -1.0
        self.demo = demo
        self.t = 0.0

    # ------------------------------------------------------------ modes
    def tick(self, t, idle, dt):
        self.t = t
        if not self.infinite_chakra and idle:
            self.chakra = min(S.CHAKRA_MAX, self.chakra + S.CHAKRA_REGEN_PER_S * dt)

    @property
    def byakugou_active(self):
        return self.t < self.byakugou_until

    @property
    def infinite_chakra(self):
        return self.byakugou_active

    def activate_byakugou(self, t):
        self.byakugou_until = t + S.BYAKUGOU_DURATION_S
        self.chakra = S.CHAKRA_MAX
        self.start_cooldown("byakugou", t, BYAKUGOU)

    # ------------------------------------------------------------ chakra
    @property
    def depleted(self):
        return not self.infinite_chakra and self.chakra <= 0.0

    def can_afford(self, cost):
        if self.infinite_chakra:
            return True
        return self.chakra > 0.0 and self.chakra >= cost

    def spend(self, cost):
        """Instant cost. Returns False (and spends nothing) if unaffordable."""
        if not self.can_afford(cost):
            return False
        if not self.infinite_chakra:
            self.chakra -= cost
        return True

    def drain(self, per_s, dt):
        """Continuous cost. Returns False once chakra hits zero."""
        if self.infinite_chakra:
            return True
        self.chakra = max(0.0, self.chakra - per_s * dt)
        return self.chakra > 0.0

    # ------------------------------------------------------------ cooldowns
    def cooldown_duration(self, kind):
        if kind == BYAKUGOU:
            return S.DEMO_BYAKUGOU_COOLDOWN_S if self.demo else S.BYAKUGOU_COOLDOWN_S
        return S.DEMO_STANDARD_COOLDOWN_S if self.demo else S.STANDARD_COOLDOWN_S

    def start_cooldown(self, name, t, kind=STANDARD):
        self.cooldown_until[name] = t + self.cooldown_duration(kind)

    def cooldown_left(self, name, t=None):
        t = self.t if t is None else t
        return max(0.0, self.cooldown_until.get(name, 0.0) - t)

    def on_cooldown(self, name, t=None):
        return self.cooldown_left(name, t) > 0.0

    # ------------------------------------------------------------ debug keys
    def reset_cooldowns(self):
        self.cooldown_until.clear()

    def refill(self):
        self.chakra = S.CHAKRA_MAX

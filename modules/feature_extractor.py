"""Feature Extraction: landmarks -> scale/rotation invariant features.

Input : Hand (21 pixel landmarks + handedness).
Output: HandFeatures: joint bend angles, normalized tip distances, spread,
        palm-facing flag, index direction. All distances are divided by palm
        size (wrist -> middle MCP), so features don't depend on how far the
        user stands from the camera.
"""
import numpy as np

WRIST = 0
THUMB = [1, 2, 3, 4]
FINGERS = {               # MCP, PIP, DIP, TIP
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}
PALM_POINTS = [0, 5, 9, 13, 17]


def distance(p1, p2):
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


def bend(a, b, c):
    """Bend at joint b in degrees: 0 = straight line, 180 = folded back."""
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return 180.0 - float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def get_hand_center(hand_pts):
    return hand_pts[PALM_POINTS, :2].mean(axis=0)


class HandFeatures:
    def __init__(self, hand):
        p = hand.landmarks
        self.hand = hand
        self.label = hand.label
        self.center = get_hand_center(p)
        self.palm_size = max(distance(p[WRIST, :2], p[9, :2]), 1.0)

        # Per finger: PIP+DIP bend (finger straightness) and MCP bend (cup).
        self.bend = {}
        self.mcp_bend = {}
        self.tip_dist = {}
        for name, (mcp, pip, dip, tip) in FINGERS.items():
            self.bend[name] = bend(p[mcp], p[pip], p[dip]) + bend(p[pip], p[dip], p[tip])
            self.mcp_bend[name] = bend(p[WRIST], p[mcp], p[pip])
            self.tip_dist[name] = distance(p[tip, :2], self.center) / self.palm_size

        self.thumb_bend = bend(p[1], p[2], p[3]) + bend(p[2], p[3], p[4])
        self.thumb_out = distance(p[4, :2], p[5, :2]) / self.palm_size
        self.spread = distance(p[8, :2], p[20, :2]) / self.palm_size

        # Palm orientation: 2D cross product of (index MCP - wrist) and
        # (pinky MCP - wrist) gives the hand's chirality as seen on screen;
        # combined with the handedness label it tells palm from back. Showing
        # the back flips the sign, in-plane rotation does not. Mirroring the
        # image flips both the sign and MediaPipe's label, so the product is
        # mirror-invariant. Sign verified on real images (palm -> True).
        v1 = p[5, :2] - p[WRIST, :2]
        v2 = p[17, :2] - p[WRIST, :2]
        cross = float(v1[0] * v2[1] - v1[1] * v2[0]) / (self.palm_size ** 2)
        self.palm_score = cross * (-1.0 if self.label == "Right" else 1.0)
        self.palm_facing = self.palm_score > 0.08

        d = p[8, :2] - p[5, :2]
        self.index_dir = d / (np.linalg.norm(d) + 1e-9)
        self.index_tip = p[8, :2]
        self.middle_tip = p[12, :2]
        self.wrist = p[WRIST, :2]

    def extended(self, name, max_bend):
        return self.bend[name] < max_bend

    def mean_curl(self):
        """Mean total curl (MCP+PIP+DIP) of the four fingers."""
        return float(np.mean([self.bend[f] + self.mcp_bend[f] for f in FINGERS]))

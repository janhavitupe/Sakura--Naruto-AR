import numpy as np

def get_hand_center(hand):
    xs = [p[0] for p in hand]
    ys = [p[1] for p in hand]
    return np.mean(xs), np.mean(ys)

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))
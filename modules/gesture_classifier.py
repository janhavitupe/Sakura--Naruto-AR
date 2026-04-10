from modules.feature_extractor import get_hand_center, distance

def detect_rasengan(hands):
    if len(hands) != 2:
        return False

    c1 = get_hand_center(hands[0])
    c2 = get_hand_center(hands[1])

    dist = distance(c1, c2)

    # threshold tuned experimentally
    if dist < 0.15:
        return True

    return False
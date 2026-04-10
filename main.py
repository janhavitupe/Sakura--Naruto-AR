import cv2
from modules.hand_tracker import HandTracker
from modules.gesture_classifier import detect_rasengan
from modules.feature_extractor import get_hand_center

cap = cv2.VideoCapture(0)
hand_tracker = HandTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    
    hands_data = hand_tracker.process(frame)

    
    if detect_rasengan(hands_data):
        c1 = get_hand_center(hands_data[0])
        c2 = get_hand_center(hands_data[1])

        center = (
            int((c1[0] + c2[0]) / 2 * frame.shape[1]),
            int((c1[1] + c2[1]) / 2 * frame.shape[0])
        )

        cv2.circle(frame, center, 30, (255, 0, 0), -1)

    cv2.imshow("Sakura", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
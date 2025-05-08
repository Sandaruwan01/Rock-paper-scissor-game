import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_hand_gesture(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    for hand_landmarks in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow("Hand Landmark Outline", image)

    landmarks = result.multi_hand_landmarks[0].landmark
    tips_ids = [8, 12, 16, 20]
    count = 0
    for tip_id in tips_ids:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            count += 1
    if landmarks[4].x < landmarks[3].x:
        count += 1

    hands.close()

    if count == 0:
        return "rock"
    elif count == 2 or count == 1:
        return "scissors"
    elif count >= 4:
        return "paper"
    else:
        return "unknown"
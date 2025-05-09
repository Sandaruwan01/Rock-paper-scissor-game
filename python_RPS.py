import cv2
import numpy as np
import random
import mediapipe as mp

# Define gestures
GESTURES = ["rock", "paper", "scissors"]

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def get_computer_choice():
    return random.choice(GESTURES)


def detect_hand_gesture(image):
    # Convert to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        hands.close()
        return None

    # Draw landmarks
    for hand_landmarks in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Landmark Outline", image)

    landmarks = result.multi_hand_landmarks[0].landmark

    # Finger tips: Index, Middle, Ring, Pinky
    tips_ids = [8, 12, 16, 20]
    count = 0
    for tip_id in tips_ids:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            count += 1

    # Thumb detection (more reliable)
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    if abs(thumb_tip.x - thumb_mcp.x) > 0.1:
        count += 1

    hands.close()

    # Classify gesture
    if count == 0:
        return "rock"
    elif count == 2 or count == 1:
        return "scissors"
    elif count >= 4:
        return "paper"
    else:
        return "unknown"


def determine_winner(player, computer):
    if player == computer:
        return "It's a tie!"
    elif (
        (player == "rock" and computer == "scissors") or
        (player == "scissors" and computer == "paper") or
        (player == "paper" and computer == "rock")
    ):
        return "You win!"
    else:
        return "Computer wins!"


def show_processing_stages(image):
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", gray)

    # Threshold
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("Threshold", thresh)

    # Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    outline = image.copy()
    cv2.drawContours(outline, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Contours", outline)


def main():
    print("Get ready... Say 'Rock, Paper, Scissors, Shoot!'")
    print("Press 's' to shoot and capture hand gesture. Press 'q' to quit.")

    cap = cv2.VideoCapture(0)

    total_games = 0
    player_wins = 0
    computer_wins = 0
    ties = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Live Feed - Press 's' to capture", frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            show_processing_stages(frame.copy())

            # Detect gesture
            player_choice = detect_hand_gesture(frame.copy())
            print(f"Your gesture: {player_choice}")

            if player_choice == "unknown" or player_choice is None:
                print("Couldn't recognize your gesture. Try again.")
                continue

            computer_choice = get_computer_choice()
            print(f"Computer chose: {computer_choice}")

            result = determine_winner(player_choice, computer_choice)
            print(result)

            # Update score
            total_games += 1
            if result == "You win!":
                player_wins += 1
            elif result == "Computer wins!":
                computer_wins += 1
            else:
                ties += 1

            # Show result and summary
            font = cv2.FONT_HERSHEY_SIMPLEX
            result_image = frame.copy()
            cv2.putText(result_image, f"You: {player_choice}", (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(result_image, f"Computer: {computer_choice}", (10, 70), font, 1, (255, 255, 255), 2)
            cv2.putText(result_image, result, (10, 110), font, 1, (0, 255, 0), 2)

            # Summary
            cv2.putText(result_image, f"Games: {total_games}", (10, 160), font, 0.7, (0, 255, 255), 2)
            cv2.putText(result_image, f"You: {player_wins} | Computer: {computer_wins} | Ties: {ties}",
                        (10, 190), font, 0.7, (0, 255, 255), 2)

            cv2.imshow("Result", result_image)
            cv2.waitKey(3000)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

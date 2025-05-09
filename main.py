from gesture_detection import detect_hand_gesture
from image_processing import show_processing_stages
from computer_choice import get_computer_choice
from game_logic import determine_winner

import cv2

def main():
    print("Get ready... Say 'Rock, Paper, Scissors, Shoot!'")
    print("Press 's' to shoot and capture hand gesture. Press 'q' to quit.")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Live Feed - Press 's' to capture", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            show_processing_stages(frame.copy())
            player_choice = detect_hand_gesture(frame.copy())
            print(f"Your gesture: {player_choice}")

            if player_choice == "unknown" or player_choice is None:
                print("Couldn't recognize your gesture. Try again.")
                continue

            computer_choice = get_computer_choice()
            print(f"Computer chose: {computer_choice}")

            result = determine_winner(player_choice, computer_choice)
            print(result)

            font = cv2.FONT_HERSHEY_SIMPLEX
            result_image = frame.copy()
            cv2.putText(result_image, f"You: {player_choice}", (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(result_image, f"Computer: {computer_choice}", (10, 70), font, 1, (255, 255, 255), 2)
            cv2.putText(result_image, result, (10, 110), font, 1, (0, 255, 0), 2)
            cv2.imshow("Result", result_image)
            cv2.waitKey(3000)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
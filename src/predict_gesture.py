import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load trained model
model = joblib.load("hand_gesture_model.pkl")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

print("Press ESC to exit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Flatten landmark coordinates
            flattened_landmarks = []
            for lm in hand_landmarks.landmark:
                flattened_landmarks.extend([lm.x, lm.y, lm.z])

            # Predict gesture
            X_input = np.array(flattened_landmarks).reshape(1, -1)
            prediction = model.predict(X_input)[0]

            # Display predicted gesture
            cv2.putText(frame, f"Gesture: {prediction}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

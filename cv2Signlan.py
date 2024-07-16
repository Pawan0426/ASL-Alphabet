import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load the trained model
with open('asl_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define a reverse label map
label_map = {i: chr(i + 65) for i in range(26)}  # 0-25 to A-Z
label_map[26] = 'space'
label_map[27] = 'del'
label_map[28] = 'nothing'

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])

                features = np.array(features).reshape(1, -1)
                prediction = model.predict(features)
                predicted_label = label_map[prediction[0]]

                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

                cv2.putText(frame, predicted_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('ASL Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

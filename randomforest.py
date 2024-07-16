import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Path to the dataset
data_dir = 'asl_alphabet_train'

# Prepare the dataset
X = []
y = []

# Map labels to numeric values
label_map = {chr(i): i - 65 for i in range(65, 91)}  # A-Z
label_map['space'] = 26
label_map['del'] = 27
label_map['nothing'] = 28

# Iterate over each label directory
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    if not os.path.isdir(label_dir):
        continue
    for img_name in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
            X.append(features)
            y.append(label_map[label])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save the model using pickle
with open('asl_rf_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

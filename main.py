import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model/infant_classifier.h5")

# Define class names
classes = ["Non-Infant", "Infant"]

# Function to preprocess image
def preprocess(frame):
    resized = cv2.resize(frame, (224, 224))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

# Start webcam
cap = cv2.VideoCapture(0)

print("Starting camera. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess(frame)
    prediction = model.predict(input_data)[0]
    label = classes[np.argmax(prediction)]

    # Display the result
    cv2.putText(frame, f"Detected: {label}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # If infant is detected, simulate bin lock
    if label == "Infant":
        cv2.putText(frame, "⚠️ Bin Locked!", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Smart Trash Bin - Infant Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

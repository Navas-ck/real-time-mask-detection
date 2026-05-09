import cv2
import numpy as np
import tensorflow as tf
import time
import os

# ==============================
# Load trained PPE model
# ==============================
model = tf.keras.models.load_model(
    "ppe_detector.h5",
    compile=False
)

class_names = ["WITH PPE", "WITHOUT PPE"]

# Get model input shape
input_shape = model.input_shape
height, width, channels = input_shape[1:4]
print("✅ Model expects input shape:", input_shape)

# ==============================
# Use UPPER BODY Haarcascade (WORKS ON WEBCAM)
# ==============================
body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_upperbody.xml"
)

# Create a folder to save captured images
save_dir = "ppe_captured_images"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

# Improve camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bodies = body_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(100, 150)
    )

    # ==============================
    # Debug: show detection count
    # ==============================
    cv2.putText(
        frame,
        f"Detections: {len(bodies)}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    for (x, y, w, h) in bodies:
        person = frame[y:y + h, x:x + w]

        try:
            # ==============================
            # Preprocess ROI
            # ==============================
            if channels == 1:
                person = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
                person = cv2.resize(person, (width, height))
                person = np.expand_dims(person, axis=-1)
            else:
                person = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
                person = cv2.resize(person, (width, height))

            person = person / 255.0
            person = np.expand_dims(person, axis=0)

            # ==============================
            # Predict PPE
            # ==============================
            pred = model.predict(person, verbose=0)[0][0]

            if pred < 0.5:
                label = f"WITH ({1 - pred:.2f})"
                color = (0, 255, 0)   # GREEN
            else:
                label = f"WITHOUT PPE ({pred:.2f})"
                color = (0, 0, 255)   # RED

            # ==============================
            # Draw bounding box
            # ==============================
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

        except Exception as e:
            print("⚠️ PPE processing error:", e)

    cv2.imshow("PPE Real-Time Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(save_dir, f"ppe_capture_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"📸 Saved: {filename}")

cap.release()
cv2.destroyAllWindows()

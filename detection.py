
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import os

model = load_model("mask_detector_strong.h5")
class_names = ["with_mask", "without_mask"]

input_shape = model.input_shape
height, width, channels = input_shape[1:4]
print("✅ Model expects input shape:", input_shape)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        try:
            if channels == 1:  
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (width, height))
                face_input = np.expand_dims(face_resized, axis=-1)
            else: 
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (width, height))
                face_input = face_resized

            face_input = face_input / 255.0
            face_input = np.expand_dims(face_input, axis=0)

            pred = model.predict(face_input, verbose=0)[0][0]

            if pred > 0.5:
                class_index = 1
                confidence = pred
            else:
                class_index = 0
                confidence = 1 - pred

            label = f"{class_names[class_index]} ({confidence:.2f})"
            color = (0, 255, 0) if class_index == 0 else (0, 0, 255)

            # Draw results
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception as e:
            print("⚠️ Error in face processing:", e)

    cv2.imshow("Mask Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"): 
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(save_dir, f"capture_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"📸 Saved: {filename}")

cap.release()

cv2.destroyAllWindows()

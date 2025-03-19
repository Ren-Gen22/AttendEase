import cv2
import os
import pandas as pd
import time
import face_recognition
from datetime import datetime
from deepface import DeepFace

# Create dataset directory if not exists
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

# Initialize attendance file
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Time"])
    df.to_csv(attendance_file, index=False)

# Dictionary to track last attendance marking time
last_marked_time = {}

# Cooldown time in seconds
CHECK_COOLDOWN = 30  
MARK_COOLDOWN = 30  

# Load known face encodings
known_faces = {}
image_extensions = {".jpg", ".jpeg", ".png"}
for file in os.listdir(dataset_dir):
    if not any(file.lower().endswith(ext) for ext in image_extensions):
        continue
    img_path = os.path.join(dataset_dir, file)
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        known_faces[file] = encodings[0]

def mark_attendance(name):
    """Marks attendance if cooldown has passed."""
    global last_marked_time
    current_time = time.time()

    if name in last_marked_time and (current_time - last_marked_time[name]) < MARK_COOLDOWN:
        print(f"Cooldown active for {name}, skipping attendance.")
        return  

    last_marked_time[name] = current_time  # Update last marked time
    
    df = pd.read_csv(attendance_file)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if name not in df["Name"].values:
        df = pd.concat([df, pd.DataFrame([{"Name": name, "Time": now}])], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        print(f"✅ Attendance marked for {name}")
    else:
        print(f"🔹 {name} is already marked present.")

# Open webcam
cap = cv2.VideoCapture(0)
print("📷 Press 'q' to quit.")

frame_count = 0  # To skip frames

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    if frame_count % CHECK_COOLDOWN != 0:
        continue  # Skip frames to reduce processing load
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        for file, known_encoding in known_faces.items():
            match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.6)
            if match[0]:  # Face matched
                name = os.path.splitext(file)[0]
                mark_attendance(name)

                # Verify with DeepFace for high accuracy
                temp_img = "temp.jpg"
                cv2.imwrite(temp_img, frame)
                try:
                    result = DeepFace.verify(img1_path=os.path.join(dataset_dir, file), img2_path=temp_img, model_name='VGG-Face')
                    if result["verified"]:
                        print(f"🔍 Verified: {name}")
                except Exception as e:
                    print("⚠️ DeepFace error:", e)
                break  # Stop checking once a match is found

    # Display video feed
    cv2.imshow("Facial Attendance", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


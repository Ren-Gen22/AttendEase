import cv2
import os
import pandas as pd
import time
from datetime import datetime
from deepface import DeepFace

# Create dataset directory if not exists
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

# Initialize attendance file (appends daily logs)
attendance_file = "attendance.csv"

# Dictionary to track last attendance marking time
last_marked_time = {}

# Cooldown times
MARK_COOLDOWN = 5  # Cooldown per user (in seconds)
FRAME_COOLDOWN = 2  # Global cooldown to reduce CPU load

# Define valid image extensions
valid_extensions = {".jpg", ".jpeg", ".png"}

def mark_attendance(name):
    """Marks attendance in a CSV file with a cooldown."""
    global last_marked_time
    current_time = time.time()
    
    # Enforce cooldown per user
    if name in last_marked_time and (current_time - last_marked_time[name]) < MARK_COOLDOWN:
        print(f"Cooldown active for {name}, waiting...")
        return  
    
    last_marked_time[name] = current_time  # Update last marked time
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([{"Name": name, "Time": now}])
    
    # Append attendance log instead of overwriting
    file_exists = os.path.exists(attendance_file)
    df.to_csv(attendance_file, mode='a', header=not file_exists, index=False)
    
    print(f"✅ Attendance marked for {name} at {now}")

# Load webcam
cap = cv2.VideoCapture(0)
print("📷 Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Save current frame as temp image
    temp_img = "temp.jpg"
    cv2.imwrite(temp_img, frame)
    
    try:
        # Iterate through dataset to find a match
        for file in os.listdir(dataset_dir):
            if not any(file.lower().endswith(ext) for ext in valid_extensions):
                continue  # Skip non-image files
            
            result = DeepFace.verify(
                img1_path=os.path.join(dataset_dir, file),
                img2_path=temp_img,
                model_name='VGG-Face',
                enforce_detection=False
            )

            if result["verified"]:
                name = os.path.splitext(file)[0]
                mark_attendance(name)
                break  # Stop checking further once a match is found

    except Exception as e:
        print("❌ Error:", e)
    
    cv2.imshow("Facial Attendance", frame)

    # Global cooldown before checking the next frame
    time.sleep(FRAME_COOLDOWN)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


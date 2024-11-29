
###------Video with alert sound------###


from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from playsound import playsound  # Import for playing sound
import threading  # To avoid sound blocking the main loop

# cap = cv2.VideoCapture(0)  # Change camera index if necessary
# cap.set(3, 1280)  # Set frame width
# cap.set(4, 720)   # Set frame height

cap = cv2.VideoCapture("../Videos/ppe-3.mp4") 

# Load the YOLO model
model = YOLO("best.pt")

# Class names
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

prev_frame_time = 0
new_frame_time = 0
fps = 0  # Initialize fps variable
alert_playing = False  # Prevent multiple alert sounds simultaneously

# Check if camera is opened
if not cap.isOpened():
    print("Error: Camera not found or could not be opened.")
    exit()

# Function to play alert sound
def play_alert():
    global alert_playing
    if not alert_playing:  # Avoid overlapping sounds
        alert_playing = True
        playsound('alert.mp3')  # Replace with your alert sound file
        alert_playing = False

while True:
    success, img = cap.read()

    if not success:
        print("Failed to capture image")
        break

    alert_triggered = False  # Reset alert flag for each frame

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            # Choose color based on class
            if classNames[cls] in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                color = (0, 0, 255)  # Red for violations
                alert_triggered = True  # Flag an alert condition
            else:
                color = (0, 255, 0)  # Green for compliance

            # Draw bounding box with chosen color
            cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=5, colorR=color)
            # Add text
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=color)

    # Trigger alert sound if a violation is detected
    if alert_triggered:
        threading.Thread(target=play_alert).start()  # Play alert sound in a separate thread

    # Calculate FPS
    new_frame_time = time.time()
    if prev_frame_time != 0:
        fps = 1 / (new_frame_time - prev_frame_time)
        fps = int(fps)  # Convert to integer
    prev_frame_time = new_frame_time

    # Display FPS on the image
    cvzone.putTextRect(img, f'FPS: {fps}', (10, 50), scale=1, thickness=1)

    print(f'FPS: {fps}')
    
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' is pressed
        break

cap.release()  # Release the camera when done
cv2.destroyAllWindows()  # Close all OpenCV windows

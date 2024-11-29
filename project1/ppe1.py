

#--- Only viedo config---#


from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Video source
cap = cv2.VideoCapture("../Videos/ppe-3.mp4")  # Replace with video file or use live feed

# Load YOLO model
model = YOLO("best.pt")

# Class names for detected objects
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

prev_frame_time = 0
new_frame_time = 0
fps = 0  # Initialize fps variable

# Check if camera or video is opened
if not cap.isOpened():
    print("Error: Camera or video file could not be opened.")
    exit()

while True:
    success, img = cap.read()

    if not success:
        print("Failed to capture frame")
        break

    results = model(img, stream=True)
    detected_items = []  # Track detected items for compliance check
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence and class index
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            detected_items.append(classNames[cls])  # Track detected class names

            # Choose color based on detected class
            if classNames[cls] in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                color = (0, 0, 255)  # Red for violations
            else:
                color = (0, 255, 0)  # Green for compliance

            # Draw bounding box and label
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5, colorR=color)
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=color)

    # Check compliance: If a "Person" is detected without "NO-Hardhat", "NO-Mask", or "NO-Safety Vest"
    if 'Person' in detected_items:
        if any(item in detected_items for item in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']):
            cvzone.putTextRect(img, 'Violation Detected!', (10, 100), scale=2, thickness=2, colorR=(0, 0, 255))
        else:
            cvzone.putTextRect(img, 'Compliance: All Safety Gear Present', (10, 100), scale=2, thickness=2, colorR=(0, 255, 0))

    # Calculate FPS
    new_frame_time = time.time()
    if prev_frame_time != 0:
        fps = 1 / (new_frame_time - prev_frame_time)
        fps = int(fps)  # Convert to integer
    prev_frame_time = new_frame_time

    # Display FPS on the image
    cvzone.putTextRect(img, f'FPS: {fps}', (10, 50), scale=1, thickness=1)

    # Show the processed frame
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' is pressed
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows

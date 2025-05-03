import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys

# Path to local YOLOv12 weights
model_path = os.path.join(os.path.expanduser("~/workspace/hkn/CS671-HACKATHON"), "yolov12n.pt")

# Initialize YOLO model
try:
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        sys.exit(1)
    model = YOLO(model_path)
    print(f"Loaded YOLOv12 model: {model_path}")
except Exception as e:
    print(f"Error loading YOLOv12 model: {e}")
    sys.exit(1)

# Initialize camera
cap = None
for index in [0, 1, 2]:  # Try webcam indices 0, 1, 2
    try:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            # Test read a frame to confirm it works
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                frame_h, frame_w = test_frame.shape[:2]
                print(f"Camera opened on index {index}, resolution: {frame_w}x{frame_h}")
                break
            else:
                cap.release()
                cap = None
        else:
            cap = None
    except Exception as e:
        print(f"Error trying camera index {index}: {e}")
        cap = None

if cap is None:
    print("Error: Could not open any camera. Ensure a webcam or DroidCam is connected and working.")
    sys.exit(1)

def process_frame(frame):
    # Run YOLO detection
    try:
        results = model(frame, conf=0.3)  # Detect with confidence threshold 0.3
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        return frame
    
    # Process detections
    person_count = 0
    if results and len(results) > 0 and hasattr(results[0], 'boxes'):
        for box in results[0].boxes:
            if hasattr(box, 'cls') and int(box.cls) == 0:  # Class 0 is 'person'
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add person count overlay
    cv2.putText(frame, f"People: {person_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame

def main():
    print("Starting person detection...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Process frame for detection
        frame = process_frame(frame)
        
        # Display frame
        cv2.imshow('Person Detection', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
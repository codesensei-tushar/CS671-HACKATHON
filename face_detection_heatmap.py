import cv2
import numpy as np
import mediapipe as mp
import os
import sys
import base64

# Path to local YOLOv12 weights (for reference, not used in this script)
yolov12_weights = os.path.join(os.path.expanduser("~/workspace/hkn/CS671-HACKATHON"), "yolov12n.pt")

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Initialize camera
cap = None
for index in [0, 1, 2]:
    try:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
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
    print("Error: Could not open any camera. Ensure DroidCam is connected via USB with debugging enabled.")
    sys.exit(1)

# Initialize heatmap
heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)
decay_factor = 0.1

def process_frame(frame):
    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run face detection
    results = face_detection.process(frame_rgb)
    
    # Process detections
    face_count = 0
    face_centers = []
    if results.detections:
        for detection in results.detections:
            face_count += 1
            bboxC = detection.location_data.relative_bounding_box
            x1 = int(bboxC.xmin * frame_w)
            y1 = int(bboxC.ymin * frame_h)
            w = int(bboxC.width * frame_w)
            h = int(bboxC.height * frame_h)
            x2 = x1 + w
            y2 = y1 + h
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Store face center for heatmap
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            face_centers.append((center_x, center_y))
    
    # Update heatmap
    global heatmap
    heatmap *= (1 - decay_factor)  # Decay existing heatmap
    for cx, cy in face_centers:
        # Add Gaussian blob for each face
        for y in range(max(0, cy-50), min(frame_h, cy+50)):
            for x in range(max(0, cx-50), min(frame_w, cx+50)):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < 50:
                    heatmap[y, x] += np.exp(-dist**2 / (2 * 25**2))
    
    # Normalize and apply colormap
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    # Overlay heatmap on frame
    frame_with_heatmap = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
    
    # Add face count overlay
    cv2.putText(frame_with_heatmap, f"Faces: {face_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Encode frame for web
    try:
        _, buffer = cv2.imencode('.jpg', frame_with_heatmap)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error encoding frame: {e}")
        frame_b64 = None
    
    return frame_with_heatmap, frame_b64, face_count

def test_locally():
    print("Starting local face detection with heatmap...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Process frame
        frame, _, face_count = process_frame(frame)
        
        # Display frame
        cv2.imshow('Face Detection with Heatmap', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_locally()
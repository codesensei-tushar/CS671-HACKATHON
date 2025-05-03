import cv2
import numpy as np
from ultralytics import YOLO, solutions
import os
import sys
import base64
import time
from collections import defaultdict

# Path to local YOLOv8m-face weights
model_path = os.path.join(os.path.expanduser("~/workspace/hkn/CS671-HACKATHON"), "yolov8m-face.pt")

# Initialize YOLO model for detection
try:
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Download from https://huggingface.co/arnabdhar/YOLOv8-Face-Detection or fine-tune yolov8m.pt on WIDER FACE.")
        sys.exit(1)
    model = YOLO(model_path)
    print(f"Loaded YOLOv8m-face model: {model_path}")
except Exception as e:
    print(f"Error loading YOLOv8m-face model: {e}")
    sys.exit(1)

# Initialize camera with reconnection logic
def init_camera():
    for index in [0, 1, 2]:  # Try webcam indices 0, 1, 2
        try:
            cap = cv2.VideoCapture(index)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduced buffer to minimize latency
            cap.set(cv2.CAP_PROP_FPS, 30)  # Request higher FPS if available
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPG format for better performance
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    frame_h, frame_w = test_frame.shape[:2]
                    print(f"Camera opened on index {index}, resolution: {frame_w}x{frame_h}")
                    return cap, frame_h, frame_w
                else:
                    cap.release()
            else:
                cap.release()
        except Exception as e:
            print(f"Error trying camera index {index}: {e}")
    return None, None, None

# Initialize camera first - this is a global variable used by multiple functions
cap, frame_h, frame_w = init_camera()
if cap is None:
    print("Error: Could not open any camera. Ensure a webcam is connected or try running with a video file.")
    sys.exit(1)

# Global heatmap variable
heatmap = None

# Initialize heatmap only after we have a valid frame to work with
def initialize_heatmap(shape):
    global heatmap
    try:
        # Custom heatmap implementation instead of using solutions.Heatmap
        # This creates a blank heatmap that we'll update manually
        h, w = shape
        heatmap_overlay = np.zeros((h, w), dtype=np.float32)
        print(f"Created custom heatmap with shape {shape}")
        return heatmap_overlay
    except Exception as e:
        print(f"Error initializing custom heatmap: {e}")
        return None

# Create a function to update and apply heatmap
def apply_heatmap(frame, detected_boxes, heatmap_overlay, decay_factor=0.15):
    try:
        # Decay existing heatmap values
        heatmap_overlay *= (1 - decay_factor)
        
        # Update heatmap with new detections
        for box in detected_boxes:
            x1, y1, x2, y2 = box
            # Create a weighted gaussian blob around the face center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            # Size of gaussian based on face size
            face_width = x2 - x1
            face_height = y2 - y1
            sigma = max(face_width, face_height) * 0.3  # Adjust this factor as needed
            
            # Create a grid for the gaussian
            y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
            # Calculate distance from center
            dist_from_center = ((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2)
            # Create gaussian mask
            mask = np.exp(-dist_from_center)
            # Normalize to 0-1
            mask = mask / mask.max()
            # Add to heatmap with some gain (0.5)
            heatmap_overlay += mask * 0.5
        
        # Clip values to 0-1 range
        heatmap_overlay = np.clip(heatmap_overlay, 0, 1)
        
        # Convert heatmap to colormap (uint8)
        heatmap_colored = cv2.applyColorMap((heatmap_overlay * 255).astype(np.uint8), cv2.COLORMAP_PARULA)
        
        # Blend with original frame
        alpha = 0.6  # Transparency factor
        blended = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
        
        return blended, heatmap_overlay
    except Exception as e:
        print(f"Error applying heatmap: {e}")
        return frame, heatmap_overlay

# Create custom heatmap overlay
heatmap_overlay = initialize_heatmap((frame_h, frame_w))

# Simple tracker for faces
class FaceTracker:
    def __init__(self, max_disappeared=10, max_distance=100):
        self.next_id = 1  # Start IDs from 1
        self.faces = {}  # {id: {box: (x1, y1, x2, y2), disappeared: count}}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.id_color_map = {}  # For consistent color per ID
    
    def register(self, box):
        face_id = f"p{self.next_id}"
        self.faces[face_id] = {"box": box, "disappeared": 0}
        # Generate a consistent color for this ID
        color = tuple(map(int, np.random.randint(50, 230, 3).tolist()))
        self.id_color_map[face_id] = color
        self.next_id += 1
        return face_id
    
    def deregister(self, face_id):
        del self.faces[face_id]
        del self.id_color_map[face_id]
    
    def update(self, boxes):
        # If no faces, mark all as disappeared
        if len(boxes) == 0:
            for face_id in list(self.faces.keys()):
                self.faces[face_id]["disappeared"] += 1
                if self.faces[face_id]["disappeared"] > self.max_disappeared:
                    self.deregister(face_id)
            return self.faces
        
        # If no existing faces, register all new ones
        if len(self.faces) == 0:
            for box in boxes:
                self.register(box)
            return self.faces
        
        # Calculate IoU between existing and new boxes
        matched_face_ids = set()
        matched_box_indices = set()
        
        # Try to match based on proximity
        for face_id in list(self.faces.keys()):
            if face_id in matched_face_ids:
                continue
            
            existing_box = self.faces[face_id]["box"]
            existing_center = ((existing_box[0] + existing_box[2]) // 2, 
                             (existing_box[1] + existing_box[3]) // 2)
            
            best_distance = self.max_distance
            best_box_idx = None
            
            for i, box in enumerate(boxes):
                if i in matched_box_indices:
                    continue
                
                new_center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
                distance = np.sqrt((existing_center[0] - new_center[0])**2 + 
                                 (existing_center[1] - new_center[1])**2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_box_idx = i
            
            if best_box_idx is not None:
                # Update with matched box
                self.faces[face_id]["box"] = boxes[best_box_idx]
                self.faces[face_id]["disappeared"] = 0
                matched_face_ids.add(face_id)
                matched_box_indices.add(best_box_idx)
        
        # Check for disappeared faces
        for face_id in list(self.faces.keys()):
            if face_id not in matched_face_ids:
                self.faces[face_id]["disappeared"] += 1
                if self.faces[face_id]["disappeared"] > self.max_disappeared:
                    self.deregister(face_id)
        
        # Register new faces
        for i, box in enumerate(boxes):
            if i not in matched_box_indices:
                self.register(box)
        
        return self.faces

# Initialize face tracker
tracker = FaceTracker(max_disappeared=8, max_distance=150)

# Performance metrics
fps_counter = 0
fps_start_time = time.time()
fps = 0

def process_frame(frame, processing_scale=1.0):
    global fps, fps_counter, fps_start_time, heatmap_overlay
    
    original_frame = frame.copy()
    
    # Resize for faster processing if scale < 1.0
    if processing_scale < 1.0:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * processing_scale), int(h * processing_scale)))
    
    # Run YOLO face detection
    try:
        results = model(frame, conf=0.3)  # Slightly higher confidence for stability
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        return original_frame, None, 0
    
    # Process detections
    face_boxes = []
    if results and len(results) > 0 and hasattr(results[0], 'boxes'):
        for box in results[0].boxes:
            if hasattr(box, 'cls') and int(box.cls) == 0:  # Face class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Scale back coordinates if needed
                if processing_scale < 1.0:
                    scale_factor = 1.0 / processing_scale
                    x1, y1, x2, y2 = int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(y2 * scale_factor)
                
                face_boxes.append((x1, y1, x2, y2))
    
    # Update face tracker
    tracked_faces = tracker.update(face_boxes)
    
    # Get boxes of currently visible faces only
    visible_boxes = [face_data["box"] for face_id, face_data in tracked_faces.items() 
                    if face_data["disappeared"] == 0]
    
    # Generate and apply heatmap
    try:
        if heatmap_overlay is None:
            heatmap_overlay = initialize_heatmap(original_frame.shape[:2])
        
        frame_with_heatmap, heatmap_overlay = apply_heatmap(
            original_frame, visible_boxes, heatmap_overlay, decay_factor=0.15
        )
    except Exception as e:
        print(f"Error generating custom heatmap: {e}")
        frame_with_heatmap = original_frame
    
    # Draw tracked boxes with IDs
    for face_id, face_data in tracked_faces.items():
        if face_data["disappeared"] == 0:  # Only draw visible faces
            x1, y1, x2, y2 = face_data["box"]
            color = tracker.id_color_map[face_id]
            
            # Draw bounding box
            cv2.rectangle(frame_with_heatmap, (x1, y1), (x2, y2), color, 2)
            
            # Add ID label
            label = f"{face_id}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame_with_heatmap, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame_with_heatmap, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Calculate FPS
    fps_counter += 1
    if fps_counter >= 10:  # Update FPS every 10 frames
        current_time = time.time()
        fps = fps_counter / (current_time - fps_start_time)
        fps_start_time = current_time
        fps_counter = 0
    
    # Add face count and FPS overlay
    cv2.putText(frame_with_heatmap, f"Faces: {len([f for f in tracked_faces.values() if f['disappeared'] == 0])}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame_with_heatmap, f"FPS: {fps:.1f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Encode frame for web
    try:
        # Use lower quality for faster encoding
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', frame_with_heatmap, encode_param)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error encoding frame: {e}")
        frame_b64 = None
    
    return frame_with_heatmap, frame_b64, len([f for f in tracked_faces.values() if f['disappeared'] == 0])

def test_locally(processing_scale=0.75):
    """Run face detection locally with visualization"""
    global cap  # Use the global cap variable
    
    print("Starting local face detection with tracking and heatmap...")
    max_retries = 3
    retry_delay = 2  # Seconds
    retries = 0
    
    # Ensure we have a valid camera
    if cap is None or not cap.isOpened():
        print("Error: Camera not initialized or opened. Trying to reinitialize...")
        cap, frame_h, frame_w = init_camera()
        if cap is None:
            print("Error: Failed to initialize camera. Exiting.")
            return
    
    while True:  # Changed from cap.isOpened() to handle camera failures better
        if not cap or not cap.isOpened():
            print("Camera disconnected. Attempting to reconnect...")
            if retries < max_retries:
                retries += 1
                print(f"Retrying camera connection (attempt {retries}/{max_retries})...")
                cap, frame_h, frame_w = init_camera()
                if cap is None:
                    print(f"Reconnection attempt {retries} failed. Waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print("Camera reconnected successfully!")
            else:
                print("Error: Max retries reached. Exiting.")
                break
        
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to read frame")
            if retries < max_retries:
                retries += 1
                print(f"Trying to recover (attempt {retries}/{max_retries})...")
                cap.release()  # Release the current capture
                time.sleep(retry_delay)
                cap, frame_h, frame_w = init_camera()  # Try to reinitialize
                continue
            else:
                print("Error: Max retries reached. Exiting.")
                break
        
        retries = 0  # Reset retries on successful frame read
        
        start_time = time.time()
        
        # Process frame with optional scaling for speed
        frame, _, face_count = process_frame(frame, processing_scale)
        
        # Calculate processing time
        process_time = time.time() - start_time
        latency = int(process_time * 1000)  # ms
        
        # Add latency information
        cv2.putText(frame, f"Latency: {latency}ms", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow('Face Detection with Tracking & Heatmap', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

# For web integration (Flask or FastAPI example)
def get_processed_frame_for_web(frame=None, processing_scale=0.75):
    """Process a frame and return base64 encoded image and metadata for web streaming"""
    global cap  # Use the global cap variable
    
    if frame is None:
        if cap is None or not cap.isOpened():
            return None, 0, 0
        
        ret, frame = cap.read()
        if not ret or frame is None:
            return None, 0, 0
    
    _, frame_b64, face_count = process_frame(frame, processing_scale)
    
    return frame_b64, face_count, fps

def cleanup():
    """Properly cleanup resources"""
    global cap
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        test_locally(processing_scale=0.75)  # Use 0.75 scale for better performance
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        cleanup()  # Ensure proper cleanup even if an exception occurs
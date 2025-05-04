#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import time
import logging
import sys
import os
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def open_droidcam(url, max_retries=5, delay=2, timeout=10):
    """Try USB cams 0–2 first, then DroidCam URL with retries."""
    cv2.setUseOptimized(True)

    # 1) USB camera priority
    logger.info("Trying USB cameras 0, 1, 2 first")
    for idx in (0, 1, 2):
        try:
            cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPG format for better performance
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    logger.info(f"Connected to USB camera index {idx}")
                    return cap
            cap.release()
        except Exception as e:
            logger.warning(f"Error trying USB camera {idx}: {e}")
        
    # 2) DroidCam fallback
    logger.info(f"USB failed — connecting to DroidCam at {url}")
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"DroidCam connection attempt {attempt}/{max_retries}")
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout * 1000)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            if cap.isOpened():
                # Try reading a test frame to verify connection
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    logger.info(f"Connected to DroidCam on attempt {attempt}")
                    return cap
                
            cap.release()
            logger.warning(f"DroidCam attempt {attempt}/{max_retries} failed — retrying in {delay}s")
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Error connecting to DroidCam: {e}")
            time.sleep(delay)

    logger.error("All camera connection attempts failed.")
    return None

def get_color_by_confidence(conf):
    if conf < 0.3:
        return (0, 0, 255)
    elif conf < 0.6:
        return (0, 165, 255)
    else:
        return (0, 255, 0)

def draw_text_with_background(frame, text, org, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(frame, (org[0], org[1]-h-10), (org[0]+w+10, org[1]), (0,0,0), -1)
    cv2.putText(frame, text, (org[0]+5, org[1]-5), font, scale, color, thick)

def annotate_frame(frame, detections, mode='boxes'):
    annotated = frame.copy()
    for det in detections:
        conf = det['conf']
        color = get_color_by_confidence(conf)
        if mode == 'dots':
            cx, cy = det['centroid']
            cv2.circle(annotated, (cx, cy), 5, color, -1)
            draw_text_with_background(annotated, f"{conf:.2f}", (cx+10, cy+10), color)
        else:
            x1, y1, x2, y2 = det['bbox']
            thick = 2 if conf > 0.7 else 1
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, thick)
            draw_text_with_background(annotated, f"{conf:.2f}", (x1, y1), color)
    return annotated

class FaceTracker:
    def __init__(self, max_disappeared=10, max_distance=100):
        self.next_id = 1
        self.faces = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.colors = {}

    def register(self, box):
        fid = f"p{self.next_id}"
        self.faces[fid] = {'box': box, 'disappeared': 0}
        self.colors[fid] = tuple(map(int, np.random.randint(50,230,3)))
        self.next_id += 1
        return fid

    def deregister(self, fid):
        del self.faces[fid]
        del self.colors[fid]

    def update(self, boxes):
        if not boxes:
            for fid in list(self.faces):
                self.faces[fid]['disappeared'] += 1
                if self.faces[fid]['disappeared'] > self.max_disappeared:
                    self.deregister(fid)
            return self.faces
        if not self.faces:
            for b in boxes:
                self.register(b)
            return self.faces
        matched_fids, matched_idxs = set(), set()
        for fid in list(self.faces):
            eb = self.faces[fid]['box']
            ec = ((eb[0]+eb[2])//2, (eb[1]+eb[3])//2)
            best_d, best_i = self.max_distance, None
            for i, b in enumerate(boxes):
                if i in matched_idxs:
                    continue
                nc = ((b[0]+b[2])//2, (b[1]+b[3])//2)
                d = np.hypot(ec[0]-nc[0], ec[1]-nc[1])
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is not None:
                self.faces[fid]['box'] = boxes[best_i]
                self.faces[fid]['disappeared'] = 0
                matched_fids.add(fid)
                matched_idxs.add(best_i)
        for fid in list(self.faces):
            if fid not in matched_fids:
                self.faces[fid]['disappeared'] += 1
                if self.faces[fid]['disappeared'] > self.max_disappeared:
                    self.deregister(fid)
        for i, b in enumerate(boxes):
            if i not in matched_idxs:
                self.register(b)
        return self.faces

def initialize_heatmap(shape):
    return np.zeros(shape, dtype=np.float32)

def apply_heatmap(frame, boxes, heatmap, decay=0.15):
    heatmap *= (1 - decay)
    h, w = frame.shape[:2]
    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1+x2)//2, (y1+y2)//2
        face_size = max(x2-x1, y2-y1)
        sigma = face_size * 0.3
        yg, xg = np.ogrid[:h, :w]
        d = ((xg-cx)**2 + (yg-cy)**2) / (2*sigma**2)
        mask = np.exp(-d)
        mask /= mask.max()
        heatmap += mask * 0.5
    np.clip(heatmap, 0, 1, out=heatmap)
    hm_col = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_PARULA)
    return cv2.addWeighted(frame, 0.4, hm_col, 0.6, 0), heatmap

def process_frame(frame, model, tracker, heatmap, processing_scale=1.0):
    orig_frame = frame.copy()
    
    # Resize for faster processing if scale < 1.0
    if processing_scale < 1.0:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * processing_scale), int(h * processing_scale)))
    
    # Run YOLO model
    results = model(frame, conf=0.3)
    
    # Extract boxes
    boxes = []
    for r in results:
        for b in r.boxes:
            if int(b.cls) == 0:  # Class 0 is person/face
                coords = b.xyxy.cpu().numpy().flatten()
                x1, y1, x2, y2 = map(int, coords)
                
                # Scale back coordinates if needed
                if processing_scale < 1.0:
                    scale_factor = 1.0 / processing_scale
                    x1, y1, x2, y2 = int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(y2 * scale_factor)
                
                boxes.append((x1, y1, x2, y2))
    
    # Update tracker
    faces = tracker.update(boxes)
    
    # Get visible face boxes only
    visible = [f['box'] for f in faces.values() if f['disappeared'] == 0]
    
    # Apply heatmap to original frame
    frame_with_heatmap, heatmap = apply_heatmap(orig_frame, visible, heatmap)
    
    # Draw bounding boxes and IDs
    for fid, data in faces.items():
        if data['disappeared'] == 0:
            x1, y1, x2, y2 = data['box']
            c = tracker.colors[fid]
            cv2.rectangle(frame_with_heatmap, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame_with_heatmap, fid, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    return frame_with_heatmap, heatmap, len(visible)

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Face Detection with YOLOv8-Face + Heatmap + Tracking")
    parser.add_argument("--url", type=str, default="http://100.69.5.85:4747/video",
                        help="DroidCam URL (default: http://100.69.5.85:4747/video)")
    parser.add_argument("--model", type=str, default="medium.pt",
                        help="Path to YOLOv8 face detection model (default: medium.pt)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Processing scale for faster performance (0.5–1.0, default: 1.0)")
    parser.add_argument("--usb", action="store_true",
                        help="Try USB cameras first before DroidCam")

    # Parse arguments
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        model_paths = [
            args.model,
            os.path.join(os.path.expanduser("~"), "workspace/hkn/CS671-HACKATHON", args.model),
            os.path.join(os.getcwd(), args.model)
        ]
        found = False
        for path in model_paths:
            if os.path.exists(path):
                args.model = path
                found = True
                break
        if not found:
            logger.error(f"Model {args.model} not found. Please download a YOLOv8 face detection model.")
            logger.info("You can download from: https://huggingface.co/arnabdhar/YOLOv8-Face-Detection")
            sys.exit(1)

    # Log settings
    logger.info("Starting face tracker with settings:")
    logger.info(f"  - Try USB first: {args.usb}")
    logger.info(f"  - URL: {args.url}")
    logger.info(f"  - Model: {args.model}")
    logger.info(f"  - Processing scale: {args.scale}")

    # Try USB cam if requested
    cap = None
    if args.usb:
        logger.info("Trying USB camera...")
        for cam_id in [0, 1]:
            cap = cv2.VideoCapture(cam_id)
            if cap is not None and cap.isOpened():
                logger.info(f"Connected to USB camera at ID {cam_id}")
                break
        if cap is None or not cap.isOpened():
            logger.warning("USB camera not found, falling back to DroidCam.")
            cap = open_droidcam(args.url)
    else:
        cap = open_droidcam(args.url)

    if not cap or not cap.isOpened():
        logger.error("Failed to connect to any camera. Exiting.")
        sys.exit(1)

    # Initialize model
    try:
        logger.info(f"Loading YOLOv8 model: {args.model}")
        model = YOLO(args.model)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

    # Initialize tracker and heatmap
    tracker = FaceTracker(max_disappeared=8, max_distance=150)
    heatmap = None
    last_shape = None

    # Performance metrics
    frame_count = 0
    fps = 0
    start = time.time()

    logger.info("Starting main loop. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Stream lost, reconnecting...")
            cap.release()
            time.sleep(1)
            if args.usb:
                for cam_id in [0, 1]:
                    cap = cv2.VideoCapture(cam_id)
                    if cap.isOpened():
                        logger.info(f"Reconnected to USB camera ID {cam_id}")
                        break
                else:
                    cap = open_droidcam(args.url)
            else:
                cap = open_droidcam(args.url)

            heatmap = None
            last_shape = None
            continue

        h, w = frame.shape[:2]
        if heatmap is None or last_shape != (h, w):
            heatmap = initialize_heatmap((h, w))
            last_shape = (h, w)

        # Process frame
        frame_proc, heatmap, count = process_frame(frame, model, tracker, heatmap, args.scale)

        # FPS calculation
        frame_count += 1
        if time.time() - start >= 1.0:
            fps = frame_count
            frame_count = 0
            start = time.time()

        latency = int((time.time() - start) * 1000) if frame_count == 0 else 0
        cv2.putText(frame_proc, f"Faces: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame_proc, f"FPS: {fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if latency > 0:
            cv2.putText(frame_proc, f"Latency: {latency}ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display frame
        cv2.imshow("Face Detection + Heatmap", frame_proc)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # Cleanup
    logger.info("Cleaning up resources...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

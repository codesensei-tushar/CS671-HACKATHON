import cv2
import numpy as np
import os
import sys
import base64
import time
from collections import defaultdict
import random
import math
import torch
from pathlib import Path

# Add YOLOv5-CrowdHuman repo to the Python path
sys.path.append('/home/atharva/workspace/hkn/CS671-HACKATHON/yolov5-crowdhuman')

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Path to your yolov5-crowdhuman repository
YOLOV5_CROWD_REPO = os.path.expanduser("~/workspace/hkn/CS671-HACKATHON/yolov5-crowdhuman")
if not os.path.isdir(YOLOV5_CROWD_REPO):
    print(f"Error: YOLOv5-CrowdHuman repo not found at {YOLOV5_CROWD_REPO}")
    sys.exit(1)

# Path to local YOLOv5m-CrowdHuman weights
model_path = os.path.expanduser("~/workspace/hkn/CS671-HACKATHON/models/crowdhuman_yolov5m.pt")

# Define the target class for detection (adjust based on model.names)
TARGET_CLASS = 1  # Assuming 'head' is class 1

# Initialize YOLOv5 model for detection
device = select_device('cpu')  # Force CPU to bypass GPU issues temporarily
try:
    if not os.path.isfile(model_path):
        print(f"Error: weights not found at {model_path}")
        sys.exit(1)
    model = attempt_load(model_path, map_location=device)
    model.eval()
    print(f"Loaded YOLOv5-CrowdHuman model: {model_path}")
    print(f"Model class names: {model.names}")  # Check class names
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Initialize camera
def init_camera():
    for index in [0, 1, 2]:
        try:
            cap = cv2.VideoCapture(index)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    h, w = test_frame.shape[:2]
                    print(f"Camera {index} opened at {w}×{h}")
                    return cap, h, w
                cap.release()
        except Exception:
            pass
    return None, None, None

cap, height, width = init_camera()
if cap is None:
    print("Failed to open a camera.")
    sys.exit(1)

# Simple head tracker
class HeadTracker:
    def __init__(self, max_disappeared=8, max_distance=50):
        self.next_id = 1
        self.heads = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.id_color_map = {}
    
    def register(self, box):
        head_id = f"h{self.next_id}"
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        self.heads[head_id] = {
            "box": box,
            "position_history": [(center_x, center_y)],
            "disappeared": 0,
            "velocity": (0, 0)
        }
        color = tuple(np.random.randint(50, 230, 3).tolist())
        self.id_color_map[head_id] = color
        self.next_id += 1
        return head_id
    
    def deregister(self, head_id):
        del self.heads[head_id]
        del self.id_color_map[head_id]
    
    def update(self, boxes):
        if len(boxes) == 0:
            for head_id in list(self.heads.keys()):
                self.heads[head_id]["disappeared"] += 1
                if self.heads[head_id]["disappeared"] > self.max_disappeared:
                    self.deregister(head_id)
            return self.heads
        
        if len(self.heads) == 0:
            for box in boxes:
                self.register(box)
            return self.heads
        
        matched_head_ids = set()
        matched_box_indices = set()
        
        for head_id in list(self.heads.keys()):
            if head_id in matched_head_ids:
                continue
            existing_box = self.heads[head_id]["box"]
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
                new_box = boxes[best_box_idx]
                new_center_x = (new_box[0] + new_box[2]) // 2
                new_center_y = (new_box[1] + new_box[3]) // 2
                if len(self.heads[head_id]["position_history"]) > 0:
                    prev_center = self.heads[head_id]["position_history"][-1]
                    vx = new_center_x - prev_center[0]
                    vy = new_center_y - prev_center[1]
                    self.heads[head_id]["velocity"] = (vx, vy)
                self.heads[head_id]["position_history"].append((new_center_x, new_center_y))
                if len(self.heads[head_id]["position_history"]) > 10:
                    self.heads[head_id]["position_history"].pop(0)
                self.heads[head_id]["box"] = new_box
                self.heads[head_id]["disappeared"] = 0
                matched_head_ids.add(head_id)
                matched_box_indices.add(best_box_idx)
        
        for head_id in list(self.heads.keys()):
            if head_id not in matched_head_ids:
                self.heads[head_id]["disappeared"] += 1
                if self.heads[head_id]["disappeared"] > self.max_disappeared:
                    self.deregister(head_id)
        
        for i, box in enumerate(boxes):
            if i not in matched_box_indices:
                self.register(box)
        
        return self.heads

# Initialize tracker
tracker = HeadTracker(max_disappeared=8, max_distance=50)

def process_frame(frame, processing_scale=1.0):
    original_frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Resize with processing_scale
    if processing_scale < 1.0:
        frame = cv2.resize(frame, (int(w * processing_scale), int(h * processing_scale)))
    
    # Ensure dimensions are divisible by 32 (YOLOv5 stride)
    stride = 32
    h, w = frame.shape[:2]
    new_h = ((h + stride - 1) // stride) * stride
    new_w = ((w + stride - 1) // stride) * stride
    padded_frame = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    padded_frame[:h, :w, :] = frame
    
    try:
        img = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(device).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        with torch.no_grad():
            pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=[TARGET_CLASS])
        head_boxes = []
        if pred[0] is not None:
            pred = pred[0]
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], original_frame.shape).round()
            for *xyxy, conf, cls in pred:
                if int(cls) == TARGET_CLASS:
                    x1, y1, x2, y2 = map(int, xyxy)
                    head_boxes.append((x1, y1, x2, y2))
    except Exception as e:
        print(f"Error in YOLOv5 detection: {e}")
        return original_frame, None, 0
    
    tracked_heads = tracker.update(head_boxes)
    
    for head_id, head_data in tracked_heads.items():
        if head_data["disappeared"] == 0:
            x1, y1, x2, y2 = head_data["box"]
            color = tracker.id_color_map[head_id]
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{head_id}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(original_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(original_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    head_count = len([h for h in tracked_heads.values() if h['disappeared'] == 0])
    cv2.putText(original_frame, f"Heads: {head_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', original_frame, encode_param)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error encoding frame: {e}")
        frame_b64 = None
    
    return original_frame, frame_b64, head_count

def test_locally(processing_scale=0.75):
    print("Starting head detection system...")
    print("Press 'q' to quit")
    
    while True:
        if not cap or not cap.isOpened():
            print("Camera disconnected. Exiting.")
            break
        
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to read frame")
            break
        
        processed_frame, _, head_count = process_frame(frame, processing_scale)
        cv2.imshow("Head Detection System", processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Head detection system stopped")

if __name__ == "__main__":
    test_locally()
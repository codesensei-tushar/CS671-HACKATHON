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

# Path to your yolov5-crowdhuman repository
YOLOV5_CROWD_REPO = os.path.expanduser("~/workspace/hkn/CS671-HACKATHON/yolov5-crowdhuman")
if not os.path.isdir(YOLOV5_CROWD_REPO):
    print(f"Error: YOLOv5-CrowdHuman repo not found at {YOLOV5_CROWD_REPO}")
    sys.exit(1)

# Ensure the YOLOv5-CrowdHuman code is importable
sys.path.insert(0, YOLOV5_CROWD_REPO)

# Import YOLOv5 loader components
from yolov5-crowdhuman.models.experimental import attempt_load
from yolov5-crowdhuman.utils.general import non_max_suppression, scale_coords
from yolov5-crowdhuman.utils.torch_utils import select_device

# Path to local YOLOv5m-CrowdHuman weights
model_path = os.path.expanduser(
    "~/workspace/hkn/CS671-HACKATHON/models/crowdhuman_yolov5m.pt"
)

# Initialize YOLOv5 model for detection
device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    if not os.path.isfile(model_path):
        print(f"Error: weights not found at {model_path}")
        sys.exit(1)

    model = attempt_load(model_path, map_location=device)
    model.eval()
    print(f"Loaded YOLOv5-CrowdHuman model: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Initialize camera with reconnection logic
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


# Initialize camera first - this is a global variable used by multiple functions
cap, frame_h, frame_w = init_camera()
if cap is None:
    print("Error: Could not open any camera. Ensure a webcam is connected or try running with a video file.")
    sys.exit(1)

# Simple tracker for people
class PersonTracker:
    def __init__(self, max_disappeared=10, max_distance=100):
        self.next_id = 1  # Start IDs from 1
        self.persons = {}  # {id: {box: (x1, y1, x2, y2), position_history: [], disappeared: count, velocity: (vx, vy)}}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.id_color_map = {}  # For consistent color per ID
        self.panic_agents = set()  # IDs of agents in panic state
        self.panic_level = 0  # Global panic level from 0 to 1
    
    def register(self, box):
        person_id = f"p{self.next_id}"
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        
        self.persons[person_id] = {
            "box": box, 
            "position_history": [(center_x, center_y)],
            "disappeared": 0,
            "velocity": (0, 0),
            "in_panic": False,
            "panic_level": 0.0  # Individual panic level
        }
        
        # Generate a consistent color for this ID
        color = tuple(map(int, np.random.randint(50, 230, 3).tolist()))
        self.id_color_map[person_id] = color
        self.next_id += 1
        return person_id
    
    def deregister(self, person_id):
        if person_id in self.panic_agents:
            self.panic_agents.remove(person_id)
        del self.persons[person_id]
        del self.id_color_map[person_id]
    
    def update(self, boxes):
        # If no people, mark all as disappeared
        if len(boxes) == 0:
            for person_id in list(self.persons.keys()):
                self.persons[person_id]["disappeared"] += 1
                if self.persons[person_id]["disappeared"] > self.max_disappeared:
                    self.deregister(person_id)
            return self.persons
        
        # If no existing people, register all new ones
        if len(self.persons) == 0:
            for box in boxes:
                self.register(box)
            return self.persons
        
        # Try to match based on proximity
        matched_person_ids = set()
        matched_box_indices = set()
        
        for person_id in list(self.persons.keys()):
            if person_id in matched_person_ids:
                continue
            
            existing_box = self.persons[person_id]["box"]
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
                # Get the new box and calculate velocity
                new_box = boxes[best_box_idx]
                new_center_x = (new_box[0] + new_box[2]) // 2
                new_center_y = (new_box[1] + new_box[3]) // 2
                
                # Calculate velocity if we have history
                if len(self.persons[person_id]["position_history"]) > 0:
                    prev_center = self.persons[person_id]["position_history"][-1]
                    vx = new_center_x - prev_center[0]
                    vy = new_center_y - prev_center[1]
                    self.persons[person_id]["velocity"] = (vx, vy)
                
                # Update position history (keep last 10 positions)
                self.persons[person_id]["position_history"].append((new_center_x, new_center_y))
                if len(self.persons[person_id]["position_history"]) > 10:
                    self.persons[person_id]["position_history"].pop(0)
                
                # Update with matched box
                self.persons[person_id]["box"] = new_box
                self.persons[person_id]["disappeared"] = 0
                matched_person_ids.add(person_id)
                matched_box_indices.add(best_box_idx)
        
        # Check for disappeared people
        for person_id in list(self.persons.keys()):
            if person_id not in matched_person_ids:
                self.persons[person_id]["disappeared"] += 1
                if self.persons[person_id]["disappeared"] > self.max_disappeared:
                    self.deregister(person_id)
        
        # Register new people
        for i, box in enumerate(boxes):
            if i not in matched_box_indices:
                self.register(box)
        
        return self.persons
    
    def get_density_map(self, frame_shape, grid_size=32):
        """Generate a density map of person positions"""
        height, width = frame_shape[:2]
        density_map = np.zeros((height // grid_size, width // grid_size), dtype=np.float32)
        
        for person_id, person_data in self.persons.items():
            if person_data["disappeared"] == 0:  # Only count visible people
                x1, y1, x2, y2 = person_data["box"]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Convert to grid coordinates
                grid_x = min(center_x // grid_size, density_map.shape[1] - 1)
                grid_y = min(center_y // grid_size, density_map.shape[0] - 1)
                
                # Increment density
                density_map[grid_y, grid_x] += 1
        
        # Normalize density map
        if np.max(density_map) > 0:
            density_map = density_map / np.max(density_map)
            
        return density_map
    
    def get_flow_map(self, frame_shape, grid_size=32):
        """Generate a flow map based on person velocities"""
        height, width = frame_shape[:2]
        flow_map_x = np.zeros((height // grid_size, width // grid_size), dtype=np.float32)
        flow_map_y = np.zeros((height // grid_size, width // grid_size), dtype=np.float32)
        count_map = np.zeros((height // grid_size, width // grid_size), dtype=np.float32)
        
        for person_id, person_data in self.persons.items():
            if person_data["disappeared"] == 0 and len(person_data["position_history"]) >= 2:
                center_x = person_data["position_history"][-1][0]
                center_y = person_data["position_history"][-1][1]
                vx, vy = person_data["velocity"]
                
                # Convert to grid coordinates
                grid_x = min(center_x // grid_size, flow_map_x.shape[1] - 1)
                grid_y = min(center_y // grid_size, flow_map_x.shape[0] - 1)
                
                # Accumulate velocity vectors
                flow_map_x[grid_y, grid_x] += vx
                flow_map_y[grid_y, grid_x] += vy
                count_map[grid_y, grid_x] += 1
        
        # Average velocities
        mask = count_map > 0
        flow_map_x[mask] /= count_map[mask]
        flow_map_y[mask] /= count_map[mask]
        
        return flow_map_x, flow_map_y
    
    def detect_stampede_risk(self, frame_shape):
        """Detect potential stampede risks based on crowd density and movement patterns"""
        # Calculate density map
        density_map = self.get_density_map(frame_shape)
        flow_map_x, flow_map_y = self.get_flow_map(frame_shape)
        
        # Metrics for stampede risk
        risk_factors = {
            "high_density": False,
            "flow_convergence": False,
            "panic_spreading": False,
            "irregular_movement": False
        }
        
        # 1. High density areas
        if np.max(density_map) > 0.7:  # 70% of max possible density
            risk_factors["high_density"] = True
        
        # 2. Flow convergence (people moving toward the same point)
        if np.any(flow_map_x) and np.any(flow_map_y):
            # Calculate flow convergence (negative divergence)
            convergence = self._calculate_flow_convergence(flow_map_x, flow_map_y)
            if np.max(convergence) > 0.5:  # Threshold for concerning convergence
                risk_factors["flow_convergence"] = True
        
        # 3. Panic spreading (number of agents in panic state)
        if len(self.panic_agents) > 0 and len(self.persons) > 0:
            panic_ratio = len(self.panic_agents) / len(self.persons)
            if panic_ratio > 0.3:  # More than 30% of persons in panic
                risk_factors["panic_spreading"] = True
        
        # 4. Irregular movement patterns
        movement_irregularity = self._calculate_movement_irregularity()
        if movement_irregularity > 0.5:  # Threshold for irregular movement
            risk_factors["irregular_movement"] = True
        
        # Overall risk score (0 to 1)
        risk_score = sum(1 for factor in risk_factors.values() if factor) / len(risk_factors)
        
        return risk_score, risk_factors
    
    def _calculate_flow_convergence(self, flow_x, flow_y):
        """Calculate flow convergence (negative divergence)"""
        # A simple approximation of divergence
        dx = np.gradient(flow_x, axis=1)
        dy = np.gradient(flow_y, axis=0)
        
        # Negative divergence = convergence
        convergence = -(dx + dy)
        
        # Normalize to [0, 1]
        if np.max(convergence) != np.min(convergence):
            convergence = (convergence - np.min(convergence)) / (np.max(convergence) - np.min(convergence))
        
        return convergence
    
    def _calculate_movement_irregularity(self):
        """Calculate irregularity in movement patterns"""
        if len(self.persons) < 2:
            return 0.0
        
        # Calculate variance in movement directions
        directions = []
        for person_id, person_data in self.persons.items():
            if person_data["disappeared"] == 0 and person_data["velocity"] != (0, 0):
                vx, vy = person_data["velocity"]
                direction = math.atan2(vy, vx)
                directions.append(direction)
        
        if not directions:
            return 0.0
        
        # Calculate circular variance (for angular data)
        sin_sum = sum(math.sin(d) for d in directions)
        cos_sum = sum(math.cos(d) for d in directions)
        r = math.sqrt(sin_sum**2 + cos_sum**2) / len(directions)
        
        # Convert to irregularity (1 - r)
        irregularity = 1 - r
        
        return irregularity
    
    def simulate_panic(self, frame_shape, trigger_probability=0.01, spread_factor=0.3):
        """Simulate panic behavior in the crowd based on Social Force Model"""
        # Randomly trigger panic in some agents
        if random.random() < trigger_probability and len(self.persons) > 0:
            # Select a random agent to start panic
            available_ids = [person_id for person_id, person_data in self.persons.items() 
                          if person_data["disappeared"] == 0 and person_id not in self.panic_agents]
            
            if available_ids:
                panic_id = random.choice(available_ids)
                self.persons[panic_id]["in_panic"] = True
                self.persons[panic_id]["panic_level"] = 0.7 + random.random() * 0.3  # Random high panic level
                self.panic_agents.add(panic_id)
                print(f"Panic triggered in agent {panic_id}")
        
        # Spread panic to nearby agents
        panic_spread = []
        for panic_id in self.panic_agents:
            if panic_id not in self.persons:  # Skip if agent was deregistered
                continue
                
            panic_data = self.persons[panic_id]
            if panic_data["disappeared"] > 0:  # Skip disappeared agents
                continue
                
            # Get position of panic agent
            p_x1, p_y1, p_x2, p_y2 = panic_data["box"]
            panic_center = ((p_x1 + p_x2) / 2, (p_y1 + p_y2) / 2)
            
            # Check which agents are nearby
            for person_id, person_data in self.persons.items():
                if person_id in self.panic_agents or person_data["disappeared"] > 0:
                    continue
                    
                # Get position of potential agent
                f_x1, f_y1, f_x2, f_y2 = person_data["box"]
                person_center = ((f_x1 + f_x2) / 2, (f_y1 + f_y2) / 2)
                
                # Calculate distance
                distance = math.sqrt((panic_center[0] - person_center[0])**2 + 
                                     (panic_center[1] - person_center[1])**2)
                
                # Distance threshold based on frame size
                threshold = min(frame_shape[0], frame_shape[1]) * 0.2  # 20% of frame size
                
                # Probability of spreading panic decreases with distance
                if distance < threshold:
                    spread_prob = spread_factor * (1 - distance/threshold) * panic_data["panic_level"]
                    if random.random() < spread_prob:
                        panic_spread.append(person_id)
        
        # Add newly panicked agents
        for person_id in panic_spread:
            if person_id not in self.panic_agents and person_id in self.persons:
                self.persons[person_id]["in_panic"] = True
                self.persons[person_id]["panic_level"] = 0.5 + random.random() * 0.3  # Random medium panic level
                self.panic_agents.add(person_id)
        
        # Update global panic level
        if self.persons:
            total_panic = sum(person_data["panic_level"] for person_id, person_data in self.persons.items()
                            if person_id in self.panic_agents)
            self.panic_level = total_panic / len(self.persons) if len(self.persons) > 0 else 0
        else:
            self.panic_level = 0
        
        # Decay panic levels over time
        for panic_id in list(self.panic_agents):
            if panic_id in self.persons:
                self.persons[panic_id]["panic_level"] *= 0.97  # Decay factor
                
                # If panic level falls below threshold, remove from panic state
                if self.persons[panic_id]["panic_level"] < 0.1:
                    self.persons[panic_id]["in_panic"] = False
                    self.persons[panic_id]["panic_level"] = 0
                    self.panic_agents.remove(panic_id)
        
        return self.panic_level, len(self.panic_agents)

# Initialize person tracker
tracker = PersonTracker(max_disappeared=8, max_distance=150)

# Performance metrics
fps_counter = 0
fps_start_time = time.time()
fps = 0

# Flags for simulation modes
enable_panic_simulation = False
enable_stampede_detection = True

def process_frame(frame, processing_scale=1.0):
    global fps, fps_counter, fps_start_time, enable_panic_simulation, enable_stampede_detection
    
    original_frame = frame.copy()
    
    # Resize for faster processing if scale < 1.0
    if processing_scale < 1.0:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * processing_scale), int(h * processing_scale)))
    
    # Run YOLOv5 person detection
    try:
        # YOLOv5 detection
        results = model(frame)
        
        # Extract detections with 'person' class (class 0 in COCO dataset)
        person_boxes = []
        # Process the results
        if results is not None and hasattr(results, 'pandas') and hasattr(results.pandas(), 'xyxy'):
            detections = results.pandas().xyxy[0]
            for _, detection in detections.iterrows():
                # Check if the detected class is 'person' (class 0 in COCO)
                # For CrowdHuman model, check appropriate class index
                if detection['class'] == 0 or detection['name'] == 'person':  # Adjust based on model output
                    x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                    
                    # Scale back coordinates if needed
                    if processing_scale < 1.0:
                        scale_factor = 1.0 / processing_scale
                        x1, y1, x2, y2 = int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(y2 * scale_factor)
                    
                    person_boxes.append((x1, y1, x2, y2))
    except Exception as e:
        print(f"Error in YOLOv5 detection: {e}")
        return original_frame, None, 0, 0, {}
    
    # Update person tracker
    tracked_persons = tracker.update(person_boxes)
    
    # Run panic simulation if enabled
    panic_level = 0
    panic_agents_count = 0
    if enable_panic_simulation:
        panic_level, panic_agents_count = tracker.simulate_panic(original_frame.shape)
    
    # Run stampede risk detection if enabled
    stampede_risk = 0
    risk_factors = {}
    if enable_stampede_detection:
        stampede_risk, risk_factors = tracker.detect_stampede_risk(original_frame.shape)
    
    # Draw tracked boxes with IDs
    for person_id, person_data in tracked_persons.items():
        if person_data["disappeared"] == 0:  # Only draw visible people
            x1, y1, x2, y2 = person_data["box"]
            
            # Determine color based on panic state
            if person_id in tracker.panic_agents:
                # Red for panic state - intensity based on panic level
                panic_level = person_data["panic_level"]
                color = (0, 0, int(255 * min(1, panic_level * 1.5)))  # Brighter red for higher panic
            else:
                color = tracker.id_color_map[person_id]
            
            # Draw bounding box
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add ID label
            label = f"{person_id}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(original_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(original_frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display flow vectors for visualization
    if len(tracked_persons) > 0:
        for person_id, person_data in tracked_persons.items():
            if person_data["disappeared"] == 0 and person_data["velocity"] != (0, 0):
                # Get center of person
                x1, y1, x2, y2 = person_data["box"]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Get velocity vector
                vx, vy = person_data["velocity"]
                
                # Draw flow vector
                end_x = int(center_x + vx * 3)  # Scale for better visibility
                end_y = int(center_y + vy * 3)
                
                # Determine color (green for normal, red for panic)
                if person_id in tracker.panic_agents:
                    vector_color = (0, 0, 255)  # Red for panic
                else:
                    vector_color = (0, 255, 0)  # Green for normal
                
                cv2.arrowedLine(original_frame, (center_x, center_y), (end_x, end_y), vector_color, 2)
    
    # Calculate FPS
    fps_counter += 1
    if fps_counter >= 10:  # Update FPS every 10 frames
        current_time = time.time()
        fps = fps_counter / (current_time - fps_start_time)
        fps_start_time = current_time
        fps_counter = 0
    
    # Add person count and FPS overlay
    cv2.putText(original_frame, f"People: {len([p for p in tracked_persons.values() if p['disappeared'] == 0])}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(original_frame, f"FPS: {fps:.1f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Display panic information if simulation is active
    if enable_panic_simulation:
        cv2.putText(original_frame, f"Panic Level: {panic_level:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(original_frame, f"Panic Agents: {panic_agents_count}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Display stampede risk information
    if enable_stampede_detection:
        risk_color = (0, 255, 0)  # Green for low risk
        if stampede_risk > 0.3:
            risk_color = (0, 165, 255)  # Orange for medium risk
        if stampede_risk > 0.6:
            risk_color = (0, 0, 255)  # Red for high risk
            
        cv2.putText(original_frame, f"Stampede Risk: {stampede_risk:.2f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)
    
    # Create risk dashboard overlay
    if enable_stampede_detection and risk_factors:
        dashboard_y = 180
        for factor, is_active in risk_factors.items():
            factor_color = (0, 0, 255) if is_active else (0, 255, 0)
            cv2.putText(original_frame, f"- {factor.replace('_', ' ').title()}: {'YES' if is_active else 'NO'}", 
                        (20, dashboard_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, factor_color, 2)
            dashboard_y += 30
    
    # Encode frame for web
    try:
        # Use lower quality for faster encoding
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', original_frame, encode_param)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error encoding frame: {e}")
        frame_b64 = None
    
    # Prepare data for web interface
    data = {
        "person_count": len([p for p in tracked_persons.values() if p['disappeared'] == 0]),
        "fps": fps,
        "panic_level": panic_level if enable_panic_simulation else 0,
        "panic_agents": panic_agents_count if enable_panic_simulation else 0,
        "stampede_risk": stampede_risk if enable_stampede_detection else 0,
        "risk_factors": risk_factors if enable_stampede_detection else {}
    }
    
    return original_frame, frame_b64, data["person_count"], data["stampede_risk"], data

def test_locally(processing_scale=0.75):
    """Run person detection locally with visualization"""
    global cap, enable_panic_simulation, enable_stampede_detection, tracker
    
    print("Starting local crowd monitoring system...")
    print("Press 'p' to toggle panic simulation")
    print("Press 's' to toggle stampede detection")
    print("Press 'r' to reset panic simulation")
    print("Press 'q' to quit")
    
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
    
    while True:
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
        
        retries = 0  # Reset retries counter on successful frame read
        
        # Process the frame
        processed_frame, _, person_count, stampede_risk, _ = process_frame(frame, processing_scale)
        
        # Display the processed frame with annotations
        cv2.imshow("Crowd Monitoring System", processed_frame)
        
        # Check for keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        
        # Key controls
        if key == ord('q'):  # Quit
            break
        elif key == ord('p'):  # Toggle panic simulation
            enable_panic_simulation = not enable_panic_simulation
            print(f"Panic simulation {'enabled' if enable_panic_simulation else 'disabled'}")
        elif key == ord('s'):  # Toggle stampede detection
            enable_stampede_detection = not enable_stampede_detection
            print(f"Stampede detection {'enabled' if enable_stampede_detection else 'disabled'}")
        elif key == ord('r'):  # Reset panic simulation
            tracker.panic_agents.clear()
            tracker.panic_level = 0
            for person_id in tracker.persons:
                tracker.persons[person_id]["in_panic"] = False
                tracker.persons[person_id]["panic_level"] = 0
            print("Panic simulation reset")
    
    # Clean up
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Crowd monitoring system stopped")

if __name__ == "__main__":
    # Start the local testing mode
    test_locally()
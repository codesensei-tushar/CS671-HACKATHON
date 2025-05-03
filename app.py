# --------------------------------------------------------
# Based on yolov10
# https://github.com/THU-MIG/yolov10/app.py
# --------------------------------------------------------'

import streamlit as st
import time
import cv2
from PIL import Image
import numpy as np
import random
from ultralytics import YOLO

# Page Configuration
st.set_page_config(page_title="Crowd Monitoring with YOLOv12", layout="wide")

# Cached Model Loading
@st.cache_resource
def load_yolo_model(model_path='yolov12m.pt'):
    """
    Load YOLO model with caching to improve performance
    
    Args:
        model_path (str): Path to YOLO model weights
    
    Returns:
        YOLO model instance
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Advanced Inference Function
def run_yolo_inference(model, image, conf_threshold=0.25, imgsz=640):
    """
    Run YOLOv12 inference with advanced processing
    
    Args:
        model (YOLO): Loaded YOLO model
        image (Image/np.ndarray): Input image
        conf_threshold (float): Confidence threshold
        imgsz (int): Image size for inference
    
    Returns:
        tuple: Annotated image, people count, additional metrics
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Run inference
    results = model.predict(
        source=image, 
        conf=conf_threshold, 
        imgsz=imgsz
    )
    
    # Get annotated image
    annotated_image = results[0].plot()
    
    # Advanced people counting and analysis
    people_boxes = [box for box in results[0].boxes if int(box.cls) == 0]
    people_count = len(people_boxes)
    
    # Calculate additional metrics
    metrics = {
        'total_objects': len(results[0].boxes),
        'people_count': people_count,
        'confidence_distribution': np.mean([box.conf.item() for box in people_boxes]) if people_boxes else 0
    }
    
    return annotated_image, people_count, metrics

# Crowd Density Estimation
def estimate_crowd_density(people_count, image_area):
    """
    Estimate crowd density based on people count and image area
    
    Args:
        people_count (int): Number of detected people
        image_area (float): Area of the image
    
    Returns:
        str: Crowd density description
    """
    density_per_sqm = people_count / image_area
    
    if density_per_sqm < 0.1:
        return "Low Density"
    elif density_per_sqm < 0.5:
        return "Medium Density"
    else:
        return "High Density"

# Main Streamlit App
def main():
    # Sidebar Configuration
    st.sidebar.title("YOLOv12 Crowd Monitoring")
    
    # Mode Selection
    app_mode = st.sidebar.selectbox(
        "Choose Mode", 
        ["Home", "Upload Image", "Live Webcam"]
    )
    
    # Model Selection
    model_choice = st.sidebar.selectbox(
        "Select Model", 
        ["yolov12n.pt", "yolov12s.pt", "yolov12m.pt", "yolov12l.pt", "yolov12x.pt"]
    )
    
    # Confidence Threshold
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.25, 
        step=0.05
    )
    
    # Image Size
    image_size = st.sidebar.slider(
        "Image Size", 
        min_value=320, 
        max_value=1280, 
        value=640, 
        step=32
    )
    
    # Load Selected Model
    model = load_yolo_model(model_choice)
    
    # Main Content
    st.title("👥 YOLOv12 Crowd Monitoring Dashboard")
    
    if app_mode == "Home":
        st.write("Welcome to Advanced Crowd Monitoring")
        st.markdown("""
        ### Key Features:
        - Real-time people detection
        - Crowd density estimation
        - Multiple model scales
        - Configurable detection parameters
        """)
        
        # Display Model Performance
        st.subheader("Model Scales")
        performance_data = {
            "Nano (n)": "40.4% mAP, 1.60ms",
            "Small (s)": "47.6% mAP, 2.42ms",
            "Medium (m)": "52.5% mAP, 4.27ms",
            "Large (l)": "53.8% mAP, 5.83ms",
            "Extra Large (x)": "55.4% mAP, 10.38ms"
        }
        
        for model, perf in performance_data.items():
            st.text(f"{model}: {perf}")
    
    elif app_mode == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image", 
            type=["jpg", "png", "jpeg"]
        )
        
        if uploaded_file is not None and model is not None:
            # Open and display original image
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="Uploaded Image", use_column_width=True)
            
            # Detect button
            if st.button("Detect Objects"):
                with st.spinner('Analyzing image...'):
                    # Run YOLOv12 inference
                    annotated_image, people_count, metrics = run_yolo_inference(
                        model, 
                        original_image, 
                        conf_threshold, 
                        image_size
                    )
                    
                    # Image area (approximation)
                    image_area = original_image.width * original_image.height / 10000  # in 10k pixels
                    
                    # Crowd density estimation
                    crowd_density = estimate_crowd_density(people_count, image_area)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("People Count", people_count)
                        st.metric("Crowd Density", crowd_density)
                    
                    with col2:
                        st.metric("Total Objects", metrics['total_objects'])
                        st.metric("Avg Confidence", f"{metrics['confidence_distribution']:.2f}")
                    
                    # Display annotated image
                    st.image(
                        annotated_image, 
                        caption="Detected Objects", 
                        channels="BGR",
                        use_column_width=True
                    )
    
    elif app_mode == "Live Webcam":
        st.write("🎥 Live Webcam Detection")
        
        start_webcam = st.button("Start Webcam Detection")
        
        if start_webcam and model is not None:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            people_count_placeholder = st.empty()
            
            # Webcam simulation for 100 frames
            for _ in range(100):
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                try:
                    # Run YOLOv12 inference on each frame
                    annotated_frame, people_count, metrics = run_yolo_inference(
                        model, 
                        frame, 
                        conf_threshold, 
                        image_size
                    )
                    
                    # Display frame and people count
                    stframe.image(
                        annotated_frame, 
                        channels="BGR"
                    )
                    
                    people_count_placeholder.metric("People Count", people_count)
                    
                    time.sleep(0.1)  # Control frame rate
                
                except Exception as e:
                    st.error(f"Error in processing: {e}")
                    break
            
            cap.release()

# Run the app
if __name__ == "__main__":
    main()

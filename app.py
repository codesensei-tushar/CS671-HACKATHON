# --------------------------------------------------------
# Based on YOLOv10/YOLOv12 Gradio Implementation
# --------------------------------------------------------

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

def yolov12_inference(image, model_path='yolov12m.pt', image_size=640, conf_threshold=0.25):
    """
    Perform YOLOv12 inference on an image
    
    Args:
        image (np.ndarray/PIL.Image): Input image
        model_path (str): Path to YOLOv12 model
        image_size (int): Inference image size
        conf_threshold (float): Confidence threshold
    
    Returns:
        np.ndarray: Annotated image
    """
    try:
        # Ensure image is in the right format
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert image to 3-channel if it's not already
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Ensure image is in the correct format for YOLO
        if image.shape[2] != 3:
            st.error(f"Unexpected image format. Shape: {image.shape}")
            return None
        
        # Load model
        model = YOLO(model_path)
        model.model.classes = [0]
        # Run inference
        results = model.predict(
            source=image, 
            imgsz=image_size, 
            conf=conf_threshold,classes=[0]
        )
        
        # Plot annotated image
        annotated_image = results[0].plot()
        
        return annotated_image
    except Exception as e:
        st.error(f"Inference error: {e}")
        return None

def main():
    st.title("YOLOv12: Attention-Centric Object Detection")
    
    # Sidebar for configuration
    st.sidebar.header("Detection Settings")
    
    # Model selection
    model_options = [
        "yolov12n.pt", 
        "yolov12s.pt", 
        "yolov12m.pt", 
        "yolov12l.pt", 
        "yolov12x.pt"
    ]
    selected_model = st.sidebar.selectbox("Select Model", model_options, index=2)
    
    # Image size and confidence threshold
    image_size = st.sidebar.slider(
        "Image Size", 
        min_value=320, 
        max_value=1280, 
        value=640, 
        step=32
    )
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.25, 
        step=0.05
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Read the image
        try:
            image = Image.open(uploaded_file)
            
            # Display image information for debugging
            st.write("Image Information:")
            st.write(f"Image Mode: {image.mode}")
            st.write(f"Image Size: {image.size}")
            
            # Convert to numpy array for display
            image_np = np.array(image)
            st.write(f"Numpy Array Shape: {image_np.shape}")
            
            # Display original image
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Detect button
            if st.button("Detect Objects"):
                with st.spinner('Running inference...'):
                    # Perform inference
                    annotated_image = yolov12_inference(
                        image, 
                        model_path=selected_model, 
                        image_size=image_size, 
                        conf_threshold=conf_threshold
                    )
                    
                    if annotated_image is not None:
                        # Convert BGR to RGB for Streamlit display
                        annotated_image_rgb = annotated_image[:, :, ::-1]
                        st.image(
                            annotated_image_rgb, 
                            caption=f"Detected Objects (Model: {selected_model})", 
                            use_column_width=True
                        )
                    else:
                        st.error("Failed to perform object detection")
        
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()

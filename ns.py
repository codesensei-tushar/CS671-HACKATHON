# --------------------------------------------------------
# Based on YOLOv10/YOLOv12 Gradio Implementation
# --------------------------------------------------------

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# ...existing code...

def yolov12_inference(image=None, video=None, model_path='yolov12m.pt', image_size=640, conf_threshold=0.25):
    """
    Perform YOLOv12 inference on an image or video with heatmap overlay.
    """
    try:
        # Load model
        model = YOLO(model_path)
        model.model.classes = [0]

        # Image processing
        if image is not None:
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

            # Run inference on image
            results = model.predict(
                source=image, 
                imgsz=image_size, 
                conf=conf_threshold,
                classes=[0]
            )
            
            # Plot annotated image
            annotated_image = results[0].plot()

            # Generate heatmap
            heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
                confidence = result.conf[0]  # Confidence score
                heatmap[y1:y2, x1:x2] += confidence  # Add confidence to heatmap region
            
            # Normalize heatmap to range [0, 255]
            heatmap = np.clip(heatmap / heatmap.max() * 255, 0, 255).astype(np.uint8)
            
            # Apply colormap to heatmap
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Overlay heatmap on the original image
            overlayed_image = cv2.addWeighted(annotated_image, 0.7, heatmap_colored, 0.3, 0)
            
            return overlayed_image[:, :, ::-1], heatmap_colored[:, :, ::-1]  # Convert BGR to RGB
 
        # Video processing
        elif video is not None:
            # Open the video
            cap = cv2.VideoCapture(video)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create temporary output video file
            output_video_path = tempfile.mktemp(suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            
            # Process video frames
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference on each frame
                results = model.predict(
                    source=frame, 
                    imgsz=image_size, 
                    conf=conf_threshold,
                    classes=[0]
                )
                
                # Plot annotated frame
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
            
            # Release resources
            cap.release()
            out.release()
            
            return output_video_path
        
        else:
            st.error("No image or video provided")
            return None
    
    except Exception as e:
        st.error(f"Inference error: {e}")
        return None

def check_backend_availability():
    """
    Check if backend services are available
    
    Returns:
        bool: True if backend is available, False otherwise
    """
    # Placeholder for backend availability check
    # In a real-world scenario, this would check actual backend services
    return False

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
    
    # Detection mode selection
    detection_mode = st.sidebar.selectbox(
        "Detection Mode", 
        ["Image", "Video"]
    )
    
    # Backend availability check
    backend_available = check_backend_availability()
    
    if detection_mode == "Image":
        # File uploader for image
        uploaded_file = st.file_uploader(
            "Choose an image", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            # Read the image
            try:
                image = Image.open(uploaded_file)
                
                # Display original image
                st.image(image, caption="Original Image", use_column_width=True)
                
                # Detect button
                if st.button("Detect Objects"):
                    with st.spinner('Running inference...'):
                        # Perform inference
                        annotated_image, heatmap = yolov12_inference(
                            image=image, 
                            model_path=selected_model, 
                            image_size=image_size, 
                            conf_threshold=conf_threshold
                        )
                        
                        if annotated_image is not None and heatmap is not None:
                            st.image(
                                annotated_image, 
                                caption=f"Detected Objects (Model: {selected_model})", 
                                use_column_width=True
                            )
                            st.image(
                                heatmap, 
                                caption="Heatmap", 
                                use_column_width=True
                            )
                        else:
                            st.error("Failed to perform object detection")
            
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    elif detection_mode == "Video":
        # Video file uploader
        uploaded_video = st.file_uploader(
            "Choose a video", 
            type=["mp4", "avi", "mov"]
        )
        
        if uploaded_video is not None:
            # Temporary save the uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_video.getvalue())
                video_path = tmp_file.name
            
            # Display original video
            st.video(video_path)
            
            if st.button("Detect Objects in Video"):
                with st.spinner('Running video inference...'):
                    # Perform video inference
                    output_video_path = yolov12_inference(
                        video=video_path, 
                        model_path=selected_model, 
                        image_size=image_size, 
                        conf_threshold=conf_threshold
                    )
                    
                    if output_video_path is not None:
                        # Display annotated video
                        st.video(output_video_path)
                        
                        # Provide download option
                        with open(output_video_path, "rb") as file:
                            st.download_button(
                                label="Download Annotated Video",
                                data=file,
                                file_name="annotated_video.mp4",
                                mime="video/mp4"
                            )
                        
                        # Clean up temporary files
                        os.unlink(video_path)
                        os.unlink(output_video_path)
                    else:
                        st.error("Failed to perform video object detection")

if __name__ == "__main__":
    main()

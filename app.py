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
import time

def yolov12_inference(image=None, video=None, model_path='yolov12m.pt', image_size=640, conf_threshold=0.25):
    """
    Perform YOLOv12 inference on an image or video
    
    Args:
        image (np.ndarray/PIL.Image, optional): Input image
        video (str, optional): Path to input video
        model_path (str): Path to YOLOv12 model
        image_size (int): Inference image size
        conf_threshold (float): Confidence threshold
    
    Returns:
        tuple: Annotated image/video (depends on input type)
    """
    try:
        # Validate input video
        if video is not None:
            # Detailed video validation for MP4
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                st.error(f"Cannot open MP4 video file: {video}")
                return None
            
            # Detailed video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Validate MP4 video properties
            st.write("MP4 Video Validation:")
            st.write(f"- Resolution: {frame_width}x{frame_height}")
            st.write(f"- FPS: {fps}")
            st.write(f"- Total Frames: {total_frames}")
            
            # Strict validation for MP4
            if fps <= 0 or frame_width <= 0 or frame_height <= 0 or total_frames == 0:
                st.error("Invalid MP4 video properties. Cannot process the video.")
                cap.release()
                return None
        
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
            return annotated_image[:, :, ::-1]  # Convert BGR to RGB
        
        # Video processing for MP4
        elif video is not None:
            # Open the video
            cap = cv2.VideoCapture(video)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create temporary output video file with full path
            output_video_path = os.path.join(tempfile.gettempdir(), f"annotated_video_{int(time.time())}.mp4")
            
            # Try multiple codecs for better compatibility
            codecs = [
                ('H264', cv2.VideoWriter_fourcc(*'X264')),  # H.264
                ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # MPEG-4
                ('avc1', cv2.VideoWriter_fourcc(*'avc1'))   # Another H.264 variant
            ]
            
            out = None
            for codec_name, fourcc in codecs:
                try:
                    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
                    if out.isOpened():
                        st.write(f"Using {codec_name} codec for video writing")
                        break
                except Exception as codec_err:
                    st.write(f"Failed to use {codec_name} codec: {codec_err}")
            
            if out is None or not out.isOpened():
                st.error("Could not create video writer with any available codec")
                cap.release()
                return None
            
            # Process video frames
            frame_count = 0
            max_frames = 300  # Limit to prevent extremely long processing
            processed_frames = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break
                
                try:
                    # Run inference on each frame
                    results = model.predict(
                        source=frame, 
                        imgsz=image_size, 
                        conf=conf_threshold,
                        classes=[0]
                    )
                    
                    # Plot annotated frame
                    annotated_frame = results[0].plot()
                    
                    # Ensure the frame is in the correct color space for writing
                    if annotated_frame.shape[2] == 4:  # RGBA
                        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGBA2RGB)
                    
                    out.write(annotated_frame)
                    processed_frames.append(annotated_frame)
                    frame_count += 1
                
                except Exception as frame_err:
                    st.error(f"Error processing MP4 frame {frame_count}: {frame_err}")
                    break
            
            # Release resources
            cap.release()
            out.release()
            
            # Verify MP4 file was created and has content
            if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                st.write(f"Successfully processed {frame_count} frames in MP4")
                return output_video_path
            else:
                st.error(f"Failed to create MP4 video. Processed frames: {len(processed_frames)}")
                return None
        
        else:
            st.error("No image or video provided")
            return None
    
    except Exception as e:
        st.error(f"MP4 Video Processing Error: {e}")
        import traceback
        st.error(traceback.format_exc())
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
                        annotated_image = yolov12_inference(
                            image=image, 
                            model_path=selected_model, 
                            image_size=image_size, 
                            conf_threshold=conf_threshold
                        )
                        
                        if annotated_image is not None:
                            st.image(
                                annotated_image, 
                                caption=f"Detected Objects (Model: {selected_model})", 
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
            
            try:
                # Validate video file
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Unable to read the video file. Please check the file format.")
                    cap.release()
                    os.unlink(video_path)
                    st.stop()
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # Display video information
                st.write(f"Video Details:")
                st.write(f"- FPS: {fps}")
                st.write(f"- Total Frames: {frame_count}")
                
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
                        
                        if output_video_path is not None and os.path.exists(output_video_path):
                            # Verify video file
                            video_size = os.path.getsize(output_video_path)
                            st.write(f"Annotated Video Size: {video_size / 1024:.2f} KB")
                            
                            # Read video file for display
                            with open(output_video_path, 'rb') as video_file:
                                video_bytes = video_file.read()
                            
                            # Display annotated video
                            st.video(video_bytes)
                            
                            # Provide download option
                            st.download_button(
                                label="Download Annotated Video",
                                data=video_bytes,
                                file_name="annotated_video.mp4",
                                mime="video/mp4"
                            )
                            
                            # Clean up temporary files
                            try:
                                os.unlink(video_path)
                                os.unlink(output_video_path)
                            except Exception as cleanup_err:
                                st.error(f"Error cleaning up temporary files: {cleanup_err}")
                        else:
                            st.error("Failed to create annotated video. Please try again with a different video or settings.")
            
            except Exception as e:
                st.error(f"Error processing video: {e}")
                # Ensure temporary files are cleaned up
                if 'video_path' in locals() and os.path.exists(video_path):
                    os.unlink(video_path)

if __name__ == "__main__":
    main()

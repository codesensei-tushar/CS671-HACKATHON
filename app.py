#!/usr/bin/env python3
import threading
import gradio as gr
import cv2
import tempfile
from ultralytics import YOLO
from ultralytics.solutions.heatmap import Heatmap
from PIL import Image
import numpy as np
from collections import defaultdict

def ensure_numpy_image(image):
    if hasattr(image, 'read') or hasattr(image, 'name'):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    return image

def draw_dots_on_frame(frame, results, min_conf=0.3, min_box_area=0, track_history=None, tracking_enabled=False):
    annotated = frame.copy()
    for box in results[0].boxes:
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if conf < min_conf or area < min_box_area:
            continue
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 7, (0, 0, 255), -1)
        # Draw trajectory if tracking is enabled and ID is present
        if tracking_enabled and hasattr(box, 'id') and box.id is not None and track_history is not None:
            track_id = int(box.id.item())
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((cx, cy))
            if len(track_history[track_id]) > 30:  # Limit history length
                track_history[track_id].pop(0)
            # Draw the trajectory line
            for i in range(1, len(track_history[track_id])):
                cv2.line(
                    annotated,
                    track_history[track_id][i - 1],
                    track_history[track_id][i],
                    (0, 255, 0),
                    2
                )
    return annotated

def yolov12_tracker_inference(image, video, model_id, image_size, conf_threshold, mode, viz_mode, use_tracking):
    model = YOLO(model_id)
    model.model.classes = [0]
    output_video_path = None
    count = 0
    info = "Processed"

    if mode == "Original":
        if video is not None:
            video_path = tempfile.mktemp(suffix=".mp4")
            with open(video_path, "wb") as f:
                with open(video.name, "rb") as g:
                    f.write(g.read())
            return None, video_path, 0, "Original video"
        return None, None, 0, "Original mode requires video input"

    if mode == "Detection":
        if image is not None:
            image = ensure_numpy_image(image)
            results = model.predict(source=image, imgsz=image_size, conf=conf_threshold, classes=[0])
            if viz_mode == "Dots":
                annotated_image = draw_dots_on_frame(image, results)
            else:
                annotated_image = results[0].plot()
            count = len(results[0].boxes)
            return annotated_image[:, :, ::-1], None, count, f"Max pedestrians in a frame: {count}"
        elif video is not None:
            video_path = tempfile.mktemp(suffix=".mp4")
            with open(video_path, "wb") as f:
                with open(video.name, "rb") as g:
                    f.write(g.read())
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_path = tempfile.mktemp(suffix="_det.mp4")
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            max_count = 0
            track_history = {}  # Only for Dots+Detection+video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold, classes=[0])
                count = len(results[0].boxes)
                if count > max_count:
                    max_count = count
                if viz_mode == "Dots":
                    annotated_frame = draw_dots_on_frame(frame, results, track_history=track_history, tracking_enabled=True)
                else:
                    annotated_frame = results[0].plot()
                out.write(annotated_frame)
            cap.release()
            out.release()
            return None, output_video_path, max_count, f"Max pedestrians in a frame: {max_count}"
    elif mode == "Tracking":
        if image is not None:
            image = ensure_numpy_image(image)
            results = model.predict(source=image, imgsz=image_size, conf=conf_threshold, classes=[0], tracker="bytetrack.yaml")
            if viz_mode == "Dots":
                annotated_image = draw_dots_on_frame(image, results)
            else:
                annotated_image = results[0].plot()
            count = len(results[0].boxes)
            return annotated_image[:, :, ::-1], None, count, f"Max pedestrians in a frame: {count}"
        elif video is not None:
            video_path = tempfile.mktemp(suffix=".mp4")
            with open(video_path, "wb") as f:
                with open(video.name, "rb") as g:
                    f.write(g.read())
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_path = tempfile.mktemp(suffix="_track.mp4")
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            track_history = defaultdict(list)
            max_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.track(frame, persist=True, imgsz=image_size, conf=conf_threshold, classes=[0], tracker="bytetrack.yaml")[0]
                if results.boxes and results.boxes.id is not None:
                    boxes = results.boxes.xywh.cpu()
                    track_ids = results.boxes.id.int().cpu().tolist()
                    annotated_frame = results.plot()
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((int(x), int(y)))  # x, y center point
                        if len(track) > 30:
                            track.pop(0)
                        # Draw the tracking lines
                        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                        if len(points) > 1:
                            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
                    out.write(annotated_frame)
                    if len(track_ids) > max_count:
                        max_count = len(track_ids)
                else:
                    out.write(frame)
            cap.release()
            out.release()
            return None, output_video_path, max_count, f"Max pedestrians in a frame: {max_count}"
    elif mode == "Heatmap":
        if image is not None:
            image = ensure_numpy_image(image)
            return image, None, 0, "Heatmap for image not implemented"
        elif video is not None:
            video_path = tempfile.mktemp(suffix=".mp4")
            with open(video_path, "wb") as f:
                with open(video.name, "rb") as g:
                    f.write(g.read())
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_path = tempfile.mktemp(suffix="_heatmap.mp4")
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            # Initialize Heatmap with your model
            heatmap = Heatmap(model=model_id, imgsz=image_size, conf=conf_threshold, classes=[0])
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # This will run detection+tracking and overlay the heatmap
                heatmap_frame = heatmap.generate_heatmap(frame)
                out.write(heatmap_frame)
            cap.release()
            out.release()
            return None, output_video_path, 0, "Heatmap complete"
    return None, None, 0, "No input provided"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        # Sidebar
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("## Controls")
            model_id = gr.Dropdown(
                label="Model",
                choices=[
                    "yolov12n.pt",
                    "yolov12m.pt",
                    "best.pt",
                    "medium.pt"
                ],
                value="medium.pt",
            )
            image_size = gr.Slider(
                label="Image Size",
                minimum=320,
                maximum=1280,
                step=32,
                value=640,
            )
            conf_threshold = gr.Slider(
                label="Confidence Threshold",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.25,
            )
            input_type = gr.Radio(["Image", "Video"], value="Image", label="Input Type")
            viz_mode = gr.Radio(["Boxes", "Dots"], value="Boxes", label="Visualization Mode")
            use_tracking = gr.Checkbox(label="Enable Tracking", value=False)
            upload = gr.File(label="Upload an image or video", file_types=[".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".webm"])
            run_btn = gr.Button("Run Inference")

        # Main content
        with gr.Column(scale=2):
            gr.Markdown("<h1 style='text-align: center; color: #20e3c2;'>YOLOV12 PEDESTRIAN DETECTION</h1>")
            
            # Output area
            output_image = gr.Image(type="numpy", label="Processed Image", visible=True, height=600, width=800)
            
            # Video outputs in a 5x5 grid, reordered as requested
            with gr.Row(visible=True):
                # First column
                with gr.Column(scale=1):
                    with gr.Row():
                        original_video = gr.Video(label="Original", visible=False, height=250, width=300, autoplay=False, elem_id="original_video")
                    with gr.Row():
                        tracking_video = gr.Video(label="Tracking", visible=False, height=250, width=300, autoplay=False, elem_id="tracking_video")
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                # Second column
                with gr.Column(scale=1):
                    with gr.Row():
                        detection_video = gr.Video(label="Detection", visible=False, height=250, width=300, autoplay=False, elem_id="detection_video")
                    with gr.Row():
                        heatmap_video = gr.Video(label="Heatmap", visible=False, height=250, width=300, autoplay=False, elem_id="heatmap_video")
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                # Third column
                with gr.Column(scale=1):
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                # Fourth column
                with gr.Column(scale=1):
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                # Fifth column
                with gr.Column(scale=1):
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                    with gr.Row():
                        gr.Markdown("")  # Empty slot
                    with gr.Row():
                        gr.Markdown("")  # Empty slot

            # Play All button
            play_all_btn = gr.Button("Play All Videos", visible=False)

            metrics = gr.Textbox(label="Metrics / Info", interactive=False)
            count_box = gr.Number(value=0, label="Pedestrian Count", interactive=False, precision=0)

    def update_outputs(input_type):
        return (
            gr.update(visible=input_type == "Image"),
            gr.update(visible=input_type == "Video"),
            gr.update(visible=input_type == "Video"),
            gr.update(visible=input_type == "Video"),
            gr.update(visible=input_type == "Video"),
            gr.update(visible=input_type == "Video"),
        )

    input_type.change(
        fn=update_outputs,
        inputs=[input_type],
        outputs=[output_image, detection_video, tracking_video, heatmap_video, original_video, play_all_btn],
    )

    def run_all(upload, model_id, image_size, conf_threshold, input_type, viz_mode, use_tracking):
        if input_type == "Image":
            if upload is not None and upload.name.lower().endswith((".jpg", ".jpeg", ".png")):
                image = upload
                img, _, count, info = yolov12_tracker_inference(image, None, model_id, image_size, conf_threshold, "Detection", viz_mode, use_tracking)
                return img, None, None, None, None, info, count, gr.update(visible=False)
            else:
                return None, None, None, None, None, "No image uploaded", 0, gr.update(visible=False)
        else:
            if upload is not None and upload.name.lower().endswith((".mp4", ".avi", ".mov", ".webm")):
                video = upload
                # Process for Original
                _, original_path, _, original_info = yolov12_tracker_inference(None, video, model_id, image_size, conf_threshold, "Original", viz_mode, use_tracking)
                # Process for Detection
                _, detection_path, detection_count, detection_info = yolov12_tracker_inference(None, video, model_id, image_size, conf_threshold, "Detection", viz_mode, use_tracking)
                # Process for Tracking
                _, tracking_path, tracking_count, tracking_info = yolov12_tracker_inference(None, video, model_id, image_size, conf_threshold, "Tracking", viz_mode, use_tracking)
                # Process for Heatmap
                _, heatmap_path, _, heatmap_info = yolov12_tracker_inference(None, video, model_id, image_size, conf_threshold, "Heatmap", viz_mode, use_tracking)
                # Combine info
                info = f"{original_info}\n{detection_info}\n{tracking_info}\n{heatmap_info}"
                # Use detection count as primary count
                count = detection_count
                return None, detection_path, tracking_path, heatmap_path, original_path, info, count, gr.update(visible=True)
            else:
                return None, None, None, None, None, "No video uploaded", 0, gr.update(visible=False)

    # JavaScript to play all videos
    play_all_js = """
    () => {
        const videos = [
            document.getElementById("original_video").querySelector("video"),
            document.getElementById("detection_video").querySelector("video"),
            document.getElementById("tracking_video").querySelector("video"),
            document.getElementById("heatmap_video").querySelector("video")
        ];
        videos.forEach(video => {
            if (video) {
                video.play().catch(e => console.error("Error playing video:", e));
            }
        });
    }
    """

    run_btn.click(
        fn=run_all,
        inputs=[upload, model_id, image_size, conf_threshold, input_type, viz_mode, use_tracking],
        outputs=[output_image, detection_video, tracking_video, heatmap_video, original_video, metrics, count_box, play_all_btn],
    )

    play_all_btn.click(
        fn=None,
        inputs=[],
        outputs=[],
        js=play_all_js
    )

demo.launch()

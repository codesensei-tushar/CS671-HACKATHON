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
    # Only boxes/dots for now, viz_mode is passed for future use
    if mode == "Detection":
        if image is not None:
            image = ensure_numpy_image(image)
            results = model.predict(source=image, imgsz=image_size, conf=conf_threshold, classes=[0])
            if viz_mode == "Dots":
                annotated_image = draw_dots_on_frame(image, results)
            else:
                annotated_image = results[0].plot()
            count = len(results[0].boxes)
            return annotated_image[:, :, ::-1], None, count, f"Detected: {count} people"
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
            return annotated_image[:, :, ::-1], None, count, f"Tracked: {count} people"
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
            # For images, just run normal detection (optional: add heatmap for single image)
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

def yolov12_inference_for_examples(image, model_path, image_size, conf_threshold):
    annotated_image, _ = yolov12_tracker_inference(image, None, model_path, image_size, conf_threshold, "Detection", "Boxes", False)
    return annotated_image

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        # Sidebar
        with gr.Column(scale=0.3, min_width=320):
            gr.Markdown("## Controls")
            model_id = gr.Dropdown(
                label="Model",
                choices=[
                    "yolov12n.pt",
                    "yolov12s.pt",
                    "yolov12m.pt",
                    "yolov12l.pt",
                    "yolov12x.pt",
                    "best.pt",
                    "medium.pt",
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
            mode = gr.Radio(["Detection", "Tracking", "Heatmap"], value="Detection", label="Mode")
            input_type = gr.Radio(["Image", "Video"], value="Image", label="Input Type")
            viz_mode = gr.Radio(["Boxes", "Dots"], value="Boxes", label="Visualization Mode")
            use_tracking = gr.Checkbox(label="Enable Tracking", value=False)
            run_btn = gr.Button("Run Inference")
        # Main content
        with gr.Column(scale=0.7):
            gr.Markdown("<h1 style='text-align: center; color: #20e3c2;'>YOLOV12 PEDESTRIAN DETECTION</h1>")
            upload = gr.File(label="Upload an image or video", file_types=[".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".webm"])
            output_image = gr.Image(type="numpy", label="Processed Image", visible=True)
            output_video = gr.Video(label="Processed Video", visible=False)
            metrics = gr.Textbox(label="Metrics / Info", interactive=False)
            count_box = gr.Number(value=0, label="Pedestrian Count", interactive=False, precision=0)

    def update_outputs(input_type):
        return (
            gr.update(visible=input_type == "Image"),
            gr.update(visible=input_type == "Video"),
        )

    input_type.change(
        fn=update_outputs,
        inputs=[input_type],
        outputs=[output_image, output_video],
    )

    def run_all(upload, model_id, image_size, conf_threshold, mode, input_type, viz_mode, use_tracking):
        if input_type == "Image":
            if upload is not None:
                image = upload if upload.name.lower().endswith((".jpg", ".jpeg", ".png")) else None
                img, vid, count, info = yolov12_tracker_inference(image, None, model_id, image_size, conf_threshold, mode, viz_mode, use_tracking)
                return img, None, info, count
            else:
                return None, None, "No image uploaded", 0
        else:
            if upload is not None:
                video = upload if upload.name.lower().endswith((".mp4", ".avi", ".mov", ".webm")) else None
                img, vid, count, info = yolov12_tracker_inference(None, video, model_id, image_size, conf_threshold, mode, viz_mode, use_tracking)
                return None, vid, info, count
            else:
                return None, None, "No video uploaded", 0

    run_btn.click(
        fn=run_all,
        inputs=[upload, model_id, image_size, conf_threshold, mode, input_type, viz_mode, use_tracking],
        outputs=[output_image, output_video, metrics, count_box],
    )

    gr.Examples(
        examples=[
            [
                "ultralytics/assets/bus.jpg",
                "yolov12s.pt",
                640,
                0.25,
            ],
            [
                "ultralytics/assets/zidane.jpg",
                "yolov12x.pt",
                640,
                0.25,
            ],
        ],
        fn=yolov12_inference_for_examples,
        inputs=[
            upload,
            model_id,
            image_size,
            conf_threshold,
        ],
        outputs=[output_image],
        cache_examples='lazy',
    )

demo.launch()  

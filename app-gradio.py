# --------------------------------------------------------
# Based on yolov10
# https://github.com/THU-MIG/yolov10/app.py
# --------------------------------------------------------'
import threading
import gradio as gr
import cv2
import tempfile
from ultralytics import YOLO


def yolov12_tracker_inference(image, video, model_id, image_size, conf_threshold):
    model = YOLO(model_id)
    model.model.classes = [0]
    if image:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold,classes=[0],tracker="bytetrack.yaml")
        # results = model.predict(source=image, imgsz=image_size, conf=conf_threshold,classes=[0],tracker="botsort.yaml")
        annotated_image = results[0].plot()
        return annotated_image[:, :, ::-1], None
    else:
        video_path = tempfile.mktemp(suffix=".webm")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix=".webm")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))
        track_history = {}
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(source=frame, imgsz=image_size, conf=conf_threshold, classes=[0], tracker="bytetrack.yaml")
            annotated_frame = results[0].plot()

            # Draw tracking lines and IDs
            for box in results[0].boxes:
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id.item())
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    center = ((xyxy[0] + xyxy[2]) // 2, (xyxy[1] + xyxy[3]) // 2)

                    if track_id not in track_history:
                        track_history[track_id] = []
                    track_history[track_id].append(center)

                    # Limit history length
                    if len(track_history[track_id]) > 30:
                        track_history[track_id].pop(0)

                    # Draw path line
                    for i in range(1, len(track_history[track_id])):
                        cv2.line(
                            annotated_frame,
                            track_history[track_id][i - 1],
                            track_history[track_id][i],
                            (0, 255, 0),
                            2)
                    cv2.putText(
                        annotated_frame,
                        f'ID: {track_id}',
                        (center[0] - 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )
            out.write(annotated_frame)

        cap.release()
        out.release()

        return None, output_video_path


def yolov12_inference_for_examples(image, model_path, image_size, conf_threshold):
    annotated_image, _ = yolov12_tracker_inference(image, None, model_path, image_size, conf_threshold)
    return annotated_image


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Image",
                    label="Input Type",
                )
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yolov12n.pt",
                        "yolov12s.pt",
                        "yolov12m.pt",
                        "yolov12l.pt",
                        "yolov12x.pt",
                    ],
                    value="yolov12m.pt",
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
                yolov12_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)

        def update_visibility(input_type):
            image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            output_image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            output_video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)

            return image, video, output_image, output_video

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video],
        )

        def run_inference(image, video, model_id, image_size, conf_threshold, input_type):
            if input_type == "Image":
                return yolov12_tracker_inference(image, None, model_id, image_size, conf_threshold)
            else:
                return yolov12_tracker_inference(None, video, model_id, image_size, conf_threshold)


        yolov12_infer.click(
            fn=run_inference,
            inputs=[image, video, model_id, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video],
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
                image,
                model_id,
                image_size,
                conf_threshold,
            ],
            outputs=[output_image],
            cache_examples='lazy',
        )

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv12: Attention-Centric Real-Time Object Detectors
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2502.12524' target='_blank'>arXiv</a> | <a href='https://github.com/sunsmarterjie/yolov12' target='_blank'>github</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()
if __name__ == '__main__':
    gradio_app.launch(share=True)

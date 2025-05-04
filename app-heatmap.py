# --------------------------------------------------------
# Based on yolov10
# https://github.com/THU-MIG/yolov10/app.py
# --------------------------------------------------------'

import gradio as gr
import cv2
import tempfile
from ultralytics import YOLO
from ultralytics.solutions.heatmap import Heatmap
import numpy as np
import cv2

def draw_colorbar(frame, colormap=cv2.COLORMAP_JET, min_val=0, max_val=255, width=30, margin=10):
    h, w = frame.shape[:2]
    bar_height = h // 2
    bar = np.linspace(min_val, max_val, bar_height).astype(np.uint8)
    bar = np.expand_dims(bar, axis=1)
    bar = cv2.applyColorMap(bar, colormap)
    bar = cv2.resize(bar, (width, bar_height))
    # Place the colorbar on the right side
    frame[margin:margin+bar_height, w-width-margin:w-margin] = bar
    # Add min/max text
    cv2.putText(frame, f'Low', (w-width-5*margin, margin+bar_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f'High', (w-width-5*margin, margin+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return frame


def yolov12_inference(image, video, model_id, image_size, conf_threshold):
    model = YOLO(model_id)
    model.model.classes = [0]
    if image:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold,classes=[0])
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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold,classes=[0])
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        return None, output_video_path


def yolov12_heatmap_inference(image, video, model_id, image_size, conf_threshold):
    if image:
        # For images, just run normal detection (optional: add heatmap for single image)
        return image, None
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

        # Initialize Heatmap with your model
        heatmap = Heatmap(model=model_id, imgsz=image_size, conf=conf_threshold, classes=[0])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # This will run detection+tracking and overlay the heatmap
            heatmap_frame = heatmap.generate_heatmap(frame)
            num_people = len(heatmap.tracker.tracks) if hasattr(heatmap, 'tracker') else 0
            cv2.putText(
                heatmap_frame,
                f'Count: {num_people}',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )
            # Add color bar
            heatmap_frame = draw_colorbar(heatmap_frame)
            out.write(heatmap_frame)

        cap.release()
        out.release()

        return None, output_video_path


def yolov12_inference_for_examples(image, model_path, image_size, conf_threshold):
    annotated_image, _ = yolov12_inference(image, None, model_path, image_size, conf_threshold)
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
                return yolov12_inference(image, None, model_id, image_size, conf_threshold)
            else:
                return yolov12_heatmap_inference(None, video, model_id, image_size, conf_threshold)


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

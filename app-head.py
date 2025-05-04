# --------------------------------------------------------
# Based on yolov10
# https://github.com/THU-MIG/yolov10/app.py
# --------------------------------------------------------'

import gradio as gr
import cv2
import tempfile
from ultralytics import YOLO
import subprocess
import os
import shutil

 
def yolov12_inference(image, video, model_id, image_size, conf_threshold):
    model = YOLO(model_id)
    # Only filter for class 0 if not using the face or crowdhuman model
    if model_id not in ["yolov8m-face.pt", "crowdhuman_yolov5m.pt"]:
        model.model.classes = [0]
    if image:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
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

            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        return None, output_video_path


def yolov12_inference_for_examples(image, model_path, image_size, conf_threshold):
    annotated_image, _ = yolov12_inference(image, None, model_path, image_size, conf_threshold)
    return annotated_image

def crowdhuman_inference(image, video, weights_path, image_size, conf_threshold, detect_heads=True):
    repo_dir = "/home/tushar/Documents/CS671-HACKATHON/yolov5-crowdhuman"
    detect_py = os.path.join(repo_dir, "detect.py")
    output_dir = tempfile.mkdtemp()
    args = [
        "python", detect_py,
        "--weights", weights_path,
        "--img", str(image_size),
        "--conf", str(conf_threshold),
        "--save-txt",
        "--project", output_dir,
        "--name", "result",
        "--exist-ok"
    ]
    if detect_heads:
        args.append("--heads")
    else:
        args.append("--person")

    if image is not None:
        # Save PIL/numpy image to temp file if needed
        if isinstance(image, str):
            image_path = image
        else:
            image_path = os.path.join(output_dir, "input.jpg")
            if hasattr(image, "save"):
                image.save(image_path)
            else:
                import cv2
                cv2.imwrite(image_path, image[:, :, ::-1])
        args += ["--source", image_path]
    else:
        # Video
        video_path = tempfile.mktemp(suffix=".mp4")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())
        args += ["--source", video_path]

    # Run detect.py
    subprocess.run(args, cwd=repo_dir, check=True)

    # Find output file
    result_dir = os.path.join(output_dir, "result")
    files = os.listdir(result_dir)
    image_files = [f for f in files if f.lower().endswith((".jpg", ".png"))]
    video_files = [f for f in files if f.lower().endswith((".mp4", ".avi", ".mov", ".webm"))]

    output_image = None
    output_video = None
    if image_files:
        output_image = os.path.join(result_dir, image_files[0])
    if video_files:
        output_video = os.path.join(result_dir, video_files[0])

    return output_image, output_video

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
                        "yolov8m-face.pt",
                        "best.pt",
                        "crowdhuman_yolov5m.pt"
                    ],
                    value="crowdhuman_yolov5m.pt",
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
            if model_id == "crowdhuman_yolov5m.pt":
                weights_path = "/home/tushar/Documents/CS671-HACKATHON/yolov5-crowdhuman/crowdhuman_yolov5m.pt"
                output_image, output_video = crowdhuman_inference(
                    image, video, weights_path, image_size, conf_threshold, detect_heads=True
                )
                # Load and return the output for Gradio
                if input_type == "Image":
                    import cv2
                    img = cv2.imread(output_image)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return img, None
                else:
                    return None, output_video
            else:
                if input_type == "Image":
                    return yolov12_inference(image, None, model_id, image_size, conf_threshold)
                else:
                    return yolov12_inference(None, video, model_id, image_size, conf_threshold)


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

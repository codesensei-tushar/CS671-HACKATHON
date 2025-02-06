# [YOLOv12: Attention-Centric Real-Time Object Detector](https://arxiv.org/abs/2502.xxxxx)


Official PyTorch implementation of **YOLOv12**.

<p align="center">
  <img src="assets/latency.svg" width=48%>
  <img src="assets/params.svg" width=48%> <br>
  Comparisons with others in terms of latency-accuracy (left) and parameter-accuracy (right) trade-offs.
</p>

[YOLOv12: Attention-Centric Real-Time Object Detector](https://arxiv.org/abs/2502.xxxxx).\
Yunjie Tian, Qixiang Ye, and David Doermann\
[![arXiv](https://img.shields.io/badge/arXiv-2405.14458-b31b1b.svg)](https://arxiv.org/abs/2502.xxxxx) <a href="https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov12-object-detection-on-custom-dataset.ipynb#scrollTo=SaKTSzSWnG7s"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/collections/sunsmarterjie/yolov12-xxxxxxxxxxxxxxxxxxxxxx)

## UPDATES 🔥
- 2025/02/15: Add [colab demo](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov10-object-detection-on-custom-dataset.ipynb#scrollTo=SaKTSzSWnG7s), [HuggingFace Demo](https://huggingface.co/spaces/kadirnar/Yolov10), and [HuggingFace Model Page](https://huggingface.co/kadirnar/Yolov10). Thanks to [SkalskiP](https://github.com/SkalskiP) and [kadirnar](https://github.com/kadirnar)! 


<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Enhancing the network architecture of the YOLO framework has been long crucial yet focused on CNN-based improvements, despite the proven superiority of attention mechanisms in modeling capabilities. This is because attention-based models cannot match the speed of CNN-based models. This paper proposes an attention-centric YOLO framework, namely YOLOv12, that matches the speed of CNN-based ones while harnessing the performance benefits of attention mechanisms.

YOLOv12 surpasses all popular real-time object detectors in both speed and accuracy. For example, YOLOv12-N achieves $40.4$ mAP with an inference latency of $1.4$ ms on a T4 GPU, outperforming the advanced YOLOv10-N/YOLOv11-N by $1.9/1.0$ mAP and being $x.x\%/x.x\%$ faster. This advantage extends to other model scales. Furthermore, YOLOv12-S achieves comparable accuracy to RT-DETR-R18/xxxx while running $86\%/xx\%$ faster, using only $35\%/xx\%$ of the computation and $45\%/xx\%$ of the parameters
</details>


## Main Results
COCO

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>T4 TensorRT10<br> | params<br><sup>(M) | FLOPs<br><sup>(B) |
| :----------------------------------------------------------------------------------- | :-------------------: | :-------------------:| :------------------------------:| :-----------------:| :---------------:|
| [YOLO12n](https://github.com/sunsmarterjie/assets/releases/download/v1.0/yolo12n.pt) | 640                   | xx.x                 | x.xx                            | 2.5                | 6.6               |
| [YOLO12s](https://github.com/sunsmarterjie/assets/releases/download/v1.0/yolo12s.pt) | 640                   | xx.x                 | x.xx                            | 8.9                | 22.0              |
| [YOLO12m](https://github.com/sunsmarterjie/assets/releases/download/v1.0/yolo12m.pt) | 640                   | xx.x                 | x.xx                            | 19.9               | 69.7              |
| [YOLO12l](https://github.com/sunsmarterjie/assets/releases/download/v1.0/yolo12l.pt) | 640                   | xx.x                 | x.xx                            | 28.3               | 97.2              |
| [YOLO12x](https://github.com/sunsmarterjie/assets/releases/download/v1.0/yolo12x.pt) | 640                   | xx.x                 | xx.x                            | 63.2               | 216.5             |

## Installation
```
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
conda create -n yolov12 python=3.11
source activate yolov12
pip install -r requirements.txt
pip install -e .
```

## Validation
[`yolov12n`]([https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov10n.pt)
[`yolov12s`]([https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov10s.pt)
[`yolov12m`]([https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov10m.pt)
[`yolov12l`]([https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov10l.pt)
[`yolov12x`]([https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov10x.pt)

```python
from ultralytics import YOLO

model = YOLO.from_pretrained('sunsmarterjie/yolov12{n/s/m/b/l/x}')
# or
# wget https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12{n/s/m/l/x}.pt
model = YOLO('yolov12{n/s/m/b/l/x}.pt')

model.val(data='coco.yaml', batch=128)
```

## Training 
```python
from ultralytics import YOLO

model = YOLO('yolov12n.yaml')

# Train the model
results = model.train(
  data='coco.yaml',
  epochs=600, 
  batch=128, 
  imgsz=640,
  device="0,1,2,3",
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
```

## Finetuning
```python

# If you want to finetune the model with pretrained weights, you could load the 
# pretrained weights like below
# model = YOLO.from_pretrained('sunsmarterjie/yolov12{n/s/m/l/x}')
# or
# wget https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12{n/s/m/l/x}.pt
# model = YOLO('yolov12{n/s/m/b/l/x}.pt')

```

## Prediction
```python
from ultralytics import YOLO

model = YOLO.from_pretrained('sunsmarterjie/yolov12{n/s/m/l/x}')
# or
# wget https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12{n/s/m/l/x}.pt
model = YOLO('yolov12{n/s/m/b/l/x}.pt')

model.predict()
```

## Export
```python
from ultralytics import YOLO

model = YOLO.from_pretrained('sunsmarterjie/yolov12{n/s/m/l/x}')
# or
# wget https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12{n/s/m/l/x}.pt
model = YOLOv10('yolov12{n/s/m/b/l/x}.pt')

model.export(...)
```

## Demo
```
python app.py
# Please visit http://127.0.0.1:7860
```


## Acknowledgement

The code base is based on [ultralytics](https://github.com/ultralytics/ultralytics). Thanks for their excellent work!

## Citation

```BibTeX
@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.xxxxx},
  year={2025}
}
```


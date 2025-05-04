# CrowdFlow: Real-Time Crowd Monitoring with YOLOv8

Welcome to **CrowdFlow**, our innovative solution for real-time crowd detection and monitoring, developed for the IIT Mandi Hackathon. This project leverages advanced computer vision techniques to detect, track, and analyze crowd dynamics using the YOLOv8 model. Our solution processes video inputs (uploaded or live via webcam) and generates four insightful outputs: original video, person detection, tracking, and a heatmap of crowd density.

The challenge was to accurately detect individuals in crowded environments, such as the Kumbh Mela, and our solution delivers robust performance with real-time processing capabilities.

---

## 🚀 Features
- **Real-Time Detection**: Identifies individuals in crowded scenes using YOLOv8.
- **Multi-Output Visualization**:
  - Original video feed.
  - Person detection with bounding boxes.
  - Tracking of individuals across frames.
  - Heatmap illustrating crowd density.
- **Flexible Input Options**:
  - Upload pre-recorded videos.
  - Live feed via DroidCam webcam for real-time simulation.
- **User-Friendly Interface**: Built with Gradio for easy interaction on localhost.
- **Tested on Real-World Data**: Evaluated on the Kumbh Mela dataset for crowd analysis.

---

## 📊 Project Overview
CrowdFlow is built on the [Ultralytics YOLOv8 framework](https://github.com/ultralytics/ultralytics), utilizing a pre-trained YOLOv8 model fine-tuned on the **Stud Head Dataset** from the internet. We further adapted the model using a pre-trained checkpoint from the [GitHub repo](https://github.com/Abcfsa/YOLOv8_head_detector). The system is designed to handle complex crowd scenarios, such as those encountered in large-scale events like the Kumbh Mela.

The application is powered by a Gradio interface, launched via `app.py`, which allows users to upload videos or connect a DroidCam webcam for live processing. The output provides comprehensive insights into crowd behavior, making it a valuable tool for event management and safety monitoring.

---

## 🛠️ Installation
Follow these steps to set up and run CrowdFlow locally:

### Prerequisites
- Python 3.8+
- Conda (for environment management)
- DroidCam (optional, for live webcam feed)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone (https://github.com/codesensei-tushar/CS671-HACKATHON)
   cd CrowdFlow
   ```

2. **Set Up the Environment**:
   Use the provided `environment.yml` to create a Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate crowdflow
   ```

3. **Install Ultralytics YOLOv8**:
   Ensure the Ultralytics package is installed:
   ```bash
   pip install ultralytics
   ```

4. **Download Pre-Trained Model**:
   Download the pre-trained YOLOv8 model from the [GitHub Repo](https://github.com/Abcfsa/YOLOv8_head_detector) and place it in the project directory.

5. **Run the Application**:
   Launch the Gradio interface:
   ```bash
   python app.py
   ```
   This will start a local server (e.g., `http://localhost:7860`) where you can interact with the application.

---

## 📂 Dataset
- **Training**: The model was fine-tuned on the [Stud Head Dataset](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release) sourced from the internet, tailored for head detection in crowded scenarios.
- **Testing**: We evaluated the model on the [Kumbh Mela Dataset](https://drive.google.com/drive/folders/1aT3KRRgx2T6xcJJlazcnLtTZpuzJjuGE), which provides real-world crowd footage for robust performance validation.

---

## 🎥 Usage
1. **Access the Gradio Interface**:
   After running `app.py`, open the provided localhost URL in your browser.

2. **Input Options**:
   - **Upload a Video**: Select a video file to process.
   - **Live Webcam**: Connect a DroidCam webcam for real-time crowd monitoring.

3. **View Outputs**:
   The interface displays four outputs:
   - **Original Video**: The raw input feed.
   - **Person Detection**: Bounding boxes around detected individuals.
   - **Tracking**: Persistent tracking of individuals across frames.
   - **Heatmap**: A visual representation of crowd density.

---

## 🧑‍💻 Technologies Used
- **YOLOv8**: Core object detection model ([Ultralytics](https://github.com/ultralytics/ultralytics)).
- **Pre-Trained Model**: Sourced from [GitHub repo](https://github.com/Abcfsa/YOLOv8_head_detector)..
- **Gradio**: For the interactive web interface.
- **Conda**: Environment management via `environment.yml`.
- **Python**: Backend logic and video processing.

---

## 📈 Results
CrowdFlow excels at detecting and tracking individuals in dense crowds, with reliable performance on the Kumbh Mela dataset. The heatmap output provides actionable insights into crowd density, enabling better event planning and safety measures. Live webcam integration ensures real-time applicability for dynamic environments.

---

## 🙌 Acknowledgements
- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 framework.
- [GitHub repo](https://github.com/Abcfsa/YOLOv8_head_detector). for the pre-trained model and challenge inspiration.
- The creators of the Kumbh Mela dataset for providing valuable testing data.

---

## 📬 Contact
For questions or feedback, please reach out to us via opening an issue on the repository.

Thank you for exploring CrowdFlow! We hope this tool inspires innovative solutions for crowd management and safety.

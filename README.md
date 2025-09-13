# 🕵️ Things Detector

A simple **real-time object detection app** using **OpenCV's DNN module** and **SSD (Single Shot Multibox Detector with MobileNet)**.  
It detects everyday objects like **person, car, dog, cat, etc.** directly from your webcam.  

---

## 📂 Project Structure
    thingsDetector/
    │── main.py # Main script to run real-time detection
    │── ssd_deploy.prototxt # Model architecture (Caffe prototxt)
    │── ssd_weights.caffemodel # Pre-trained SSD weights
---

## ⚙️ Requirements

- Python 3.x
- OpenCV
- NumPy

### Install with pip
    ```bash
    pip install opencv-python numpy

### Install with conda
    conda create -n detector python=3.9 -y
    conda activate detector
    conda install -c conda-forge opencv numpy -y
---

## ▶️ Usage

Run the script with:
    ```bash
    python main.py

- The webcam feed will open.
- Press q to quit the window.
---

## 🧠 Model Info
This project uses SSD MobileNet (trained on VOC dataset).
It can detect 20 object classes:
["background", "aeroplane", "bicycle", "bird", "boat",
 "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
 "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
 "sofa", "train", "tvmonitor"]

---

## ✨ Features

- Real-time detection from webcam
- Lightweight SSD MobileNet model
- Bounding box with class labels
- Confidence thresholding
---

## 🔮 Future Improvements

- Add FPS counter
- Save processed video
- Extend support to custom datasets
- Upgrade to YOLOv8 or other modern object detectors

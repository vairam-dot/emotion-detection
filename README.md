# Real-Time Human and Emotion Detection

A professional Python application for real-time human detection and facial emotion estimation using OpenCV, NumPy, and Tkinter.

## Features

- Detects humans in camera video and uploaded images.
- Identifies face regions and draws bounding boxes.
- Estimates facial emotion states: happy, sad, or crying.
- Simple GUI for camera streaming, image upload, processing, and result display.

## Installation

1. Create a Python virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install required packages:

```powershell
pip install -r requirements.txt
```

## Usage

Run the application:

```powershell
python app.py
```

Use the GUI buttons to:

- Start Camera: Begin real-time face and human detection from your webcam.
- Stop Camera: Stop video processing.
- Upload Image: Choose a static image to analyze.
- Process Image: Analyze the selected image and display detections.

## Project Structure

- `app.py` - Main GUI application.
- `detector.py` - Detection and emotion estimation logic.
- `requirements.txt` - Project dependencies.

## Notes

- The emotion detection is rule-based and uses OpenCV Haar cascades for face, eye, and smile detection.
- For production-grade emotion recognition, replace the heuristic logic with a trained classification model.

# Saffron Flower Detection with FastAPI & YOLOv8

This project provides a web interface for detecting saffron flowers in images using a YOLOv8 model, powered by FastAPI. It supports both image upload and live detection using an Intel RealSense camera.

## Features

- Upload an image and detect saffron flowers.
- Live preview and capture from a RealSense camera.
- Displays detection results, bounding boxes, centers, and confidences.
- Results shown on the web interface with annotated images.
- Easily extensible for sending results to another server.

## Project Structure

```
.
├── app.py
├── requirements.txt
├── static/
│   ├── main.js
│   ├── styles1.css
│   ├── styles_all_format.css
│   ├── capture_image/
│   ├── detected_image/
│   └── logo/
├── templates/
│   ├── index.html
│   └── index_Extra_All_Versions.html
├── weights/
│   ├── best.pt
│   └── custom_loss_py_yolov8_best_100.pt
└── .gitignore
```

## Setup

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd <project-directory>
```

### 2. Create and Activate Virtual Environment

```sh
python -m venv fastapienv
# On Windows
fastapienv\Scripts\activate
# On Unix/Mac
source fastapienv/bin/activate
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

### 4. Download Model Weights

Place your YOLOv8 weights (`best.pt` or `custom_loss_py_yolov8_best_100.pt`) in the `weights/` directory.

### 5. Install Intel RealSense SDK (for live camera)

Follow instructions at [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense) to install `pyrealsense2`.

## Running the App

```sh
uvicorn app:app --reload
```

- Visit [http://localhost:8000](http://localhost:8000) in your browser.

## Usage

- **Upload Image**: Use the form to upload an image and set a confidence threshold. Results will be displayed below.
- **Live Camera**: Click "Open Camera (Live Preview)" to start the RealSense camera. Use "Capture & Detect" to capture a frame and run detection.
- **Clear Results**: Use the "Clear Results" button to reset the output.

## File Descriptions

- [`app.py`](app.py): Main FastAPI application with all endpoints and detection logic.
- [`static/`](static/): Static files (JS, CSS, images).
- [`templates/index.html`](templates/index.html): Main HTML template for the web interface.
- [`weights/`](weights/): YOLOv8 model weights.
- [`requirements.txt`](requirements.txt): Python dependencies.

## Notes

- The RealSense camera must be connected for live preview and capture.
- To send detection results to another server, edit the `send_results_to_other_server` function in [`app.py`](app.py).

## License

MIT License (add your license here).

---

**Developed for saffron flower detection using YOLOv8 and FastAPI.**
# import os
# import io
# import time
# from fastapi import FastAPI, UploadFile, Form, Request
# from fastapi.responses import JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from ultralytics import YOLO
# import cv2
# import numpy as np

# app = FastAPI()

# # Load YOLO model once
# model = YOLO("weights/best.pt")

# # Serve static directory for CSS and detected images
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Template directory
# templates = Jinja2Templates(directory="templates")


# def detection(frame, model, threshold):
#     results = model(frame)
#     boxes = results[0].boxes.xyxy.tolist()
#     confidences = results[0].boxes.conf.tolist()
#     classes = results[0].boxes.cls.tolist()
#     names = results[0].names

#     detected_boxes = []
#     detected_centers = []
#     detected_confidences = []

#     annotated_frame = frame.copy()

#     for box, confidence in zip(boxes, confidences):
#         if confidence >= threshold:
#             start_point = (round(box[0]), round(box[1]))
#             end_point = (round(box[2]), round(box[3]))
#             # Draw bounding box
#             cv2.rectangle(annotated_frame, start_point, end_point, (0, 0, 255), 3)
#             # Compute center
#             center_x = round((box[0] + box[2]) / 2)
#             center_y = round((box[1] + box[3]) / 2)
#             cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
#             detected_boxes.append(box)
#             detected_centers.append([center_x, center_y])
#             detected_confidences.append(confidence)

#     return annotated_frame, detected_boxes, detected_centers, detected_confidences


# @app.get("/")
# async def main(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})


# @app.post("/detect/")
# async def detect_image(
#     file: UploadFile,
#     threshold: float = Form(0.5)
# ):
#     if not (0.1 <= threshold <= 1.0):
#         return JSONResponse(
#             status_code=400,
#             content={"error": "Threshold must be between 0.1 and 1.0"}
#         )

#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     start_time = time.time()
#     annotated_image, boxes, centers, confidences = detection(img, model, threshold)
#     end_time = time.time()

#     # If no detections, return message
#     if not boxes:
#         return JSONResponse(
#             content = {
#                 "message": "No Object Detected",
#                 "Processing_Time":"round(end_time-start_time, 3)"
#             }
#         )
#     # Save annotated image
#     save_path = "static/detected.jpg"
#     cv2.imwrite(save_path, annotated_image)

#     response = {
#         "num_detections": len(boxes),
#         "bounding_boxes": boxes,
#         "centers": centers,
#         "confidences": confidences,
#         "processing_time_sec": round(end_time - start_time, 3),
#         "detected_image_url": "/static/detected.jpg"
#     }
#     return JSONResponse(content=response)

######### 2nd version

# import os
# import io
# import time
# from fastapi import FastAPI, UploadFile, Form, Request
# from fastapi.responses import JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from ultralytics import YOLO
# import cv2
# import numpy as np

# app = FastAPI()

# # Load YOLO model once
# model = YOLO("weights/best.pt")

# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")


# def detection(frame, model, threshold):
#     results = model(frame)
#     boxes = results[0].boxes.xyxy.tolist()
#     confidences = results[0].boxes.conf.tolist()
#     detected_boxes = []
#     detected_centers = []
#     detected_confidences = []
#     annotated_frame = frame.copy()

#     for box, confidence in zip(boxes, confidences):
#         if confidence >= threshold:
#             start_point = (round(box[0]), round(box[1]))
#             end_point = (round(box[2]), round(box[3]))
#             cv2.rectangle(annotated_frame, start_point, end_point, (0, 0, 255), 3)
#             center_x = round((box[0] + box[2]) / 2)
#             center_y = round((box[1] + box[3]) / 2)
#             cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
#             detected_boxes.append(box)
#             detected_centers.append([center_x, center_y])
#             detected_confidences.append(confidence)

#     return annotated_frame, detected_boxes, detected_centers, detected_confidences


# @app.get("/")
# async def main(request: Request):
#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "message": None,
#         "num_detections": None,
#         "bounding_boxes": [],
#         "centers": [],
#         "confidences": [],
#         "processing_time_sec": None,
#         "image_url": None
#     })


# @app.post("/detect/")
# async def detect_html(request: Request, file: UploadFile, threshold: float = Form(0.5)):
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     start_time = time.time()
#     annotated_image, boxes, centers, confidences = detection(img, model, threshold)
#     end_time = time.time()

#     if boxes:
#         cv2.imwrite("static/detected.jpg", annotated_image)
#         message = "Objects detected"
#         image_url = "/static/detected.jpg"
#     else:
#         cv2.imwrite("static/detected.jpg", img)
#         message = "No object detected"
#         image_url = "/static/detected.jpg"

#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "message": message,
#         "num_detections": len(boxes),
#         "bounding_boxes": boxes,
#         "centers": centers,
#         "confidences": confidences,
#         "processing_time_sec": round(end_time - start_time, 3),
#         "image_url": image_url
#     })


# @app.post("/detect-json/")
# async def detect_json(file: UploadFile, threshold: float = Form(0.5)):
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     start_time = time.time()
#     annotated_image, boxes, centers, confidences = detection(img, model, threshold)
#     end_time = time.time()

#     if boxes:
#         cv2.imwrite("static/detected.jpg", annotated_image)
#         message = "Objects detected"
#         image_url = "/static/detected.jpg"
#     else:
#         cv2.imwrite("static/detected.jpg", img)
#         message = "No object detected"
#         image_url = "/static/detected.jpg"

#     return JSONResponse({
#         "message": message,
#         "num_detections": len(boxes),
#         "bounding_boxes": boxes,
#         "centers": centers,
#         "confidences": confidences,
#         "processing_time_sec": round(end_time - start_time, 3),
#         "detected_image_url": image_url
#     })

############  2nd version end

############ 3rd version (picture taking using realsense)

import threading
import time
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import pyrealsense2 as rs
import requests

app = FastAPI()

# Load the YOLOv8 model
# model = YOLO("weights/best.pt")
model = YOLO("weights/custom_loss_py_yolov8_best_100.pt")

# Static and template config
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global RealSense camera objects
pipeline = None
camera_thread = None
stop_event = threading.Event()

# Function to show live RealSense camera feed
def live_camera_feed():
    global pipeline, stop_event
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow("RealSense Live Feed - Press Q to close", color_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# Function to capture a frame from RealSense
def capture_realsense_image(save_path):
    global pipeline
    if pipeline is None:
        return None
    for _ in range(5):  # Warm-up
        pipeline.wait_for_frames()
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imwrite(save_path, color_image)
    return color_image

# Object detection function
def detect_objects(image, threshold):
    results = model(image)
    boxes = results[0].boxes.xyxy.tolist()
    confidences = results[0].boxes.conf.tolist()
    centers = []
    filtered_boxes = []
    filtered_confidences = []

    annotated = image.copy()

    for box, conf in zip(boxes, confidences):
        if conf >= threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
            centers.append([cx, cy])
            filtered_boxes.append(box)
            filtered_confidences.append(conf)

    return annotated, filtered_boxes, centers, filtered_confidences

####################################
## Send data to another server function
def send_results_to_other_server(data: dict):
    url = "https://your-other-server.com/receive-data"   # Replace with your target server URL
    try:
        response = requests.post(url, json=data, timeout=5)
        response.raise_for_status()
        print("✅ Data sent successfully to other server.")
    except requests.RequestException as e:
        print(f"❌ Error sending data: {e}")

#########################################################

# Home page
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": None,
        "num_detections": None,
        "bounding_boxes": [],
        "centers": [],
        "confidences": [],
        "processing_time_sec": None,
        "image_url": None
    })

# Start live camera in background
@app.post("/start-camera/")
async def start_camera():
    global camera_thread, stop_event
    if camera_thread is None or not camera_thread.is_alive():
        stop_event.clear()
        camera_thread = threading.Thread(target=live_camera_feed, daemon=True)
        camera_thread.start()
        return JSONResponse({"status": "Camera started"})
    else:
        return JSONResponse({"status": "Camera already running"})

# Capture image from RealSense and detect
@app.post("/capture-detect/")
async def capture_and_detect(request: Request, threshold: float = Form(0.5)):
    save_path = "static/capture_image/frame.jpg"
    captured_image = capture_realsense_image(save_path)
    if captured_image is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "❌ Failed to capture image. Start camera first.",
            "num_detections": 0,
            "bounding_boxes": [],
            "centers": [],
            "confidences": [],
            "processing_time_sec": 0,
            "image_url": None
        })

    start = time.time()
    annotated, boxes, centers, confs = detect_objects(captured_image, threshold)
    end = time.time()

    output_path = "static/detected_image/detected.jpg"
    cv2.imwrite(output_path, annotated if boxes else captured_image)

    #######################################
    # ## Send data to another server
    # # Prepare JSON payload
    # payload = {
    #     "num_detections": len(boxes),
    #     "bounding_boxes": boxes,
    #     "centers": centers,
    #     "confidences": confs,
    #     "processing_time_sec": round(end - start, 3)
    # }

    # # Send to other server
    # send_results_to_other_server(payload)

    #############################################

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": "✅ Objects detected" if boxes else "⚠️ No object detected",
        "num_detections": len(boxes),
        "bounding_boxes": boxes,
        "centers": centers,
        "confidences": confs,
        "processing_time_sec": round(end - start, 3),
        "image_url": "/" + output_path
    })

# Upload an image and detect
@app.post("/detect/")
async def detect_uploaded_image(request: Request, file: UploadFile, threshold: float = Form(0.5)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    start = time.time()
    annotated, boxes, centers, confs = detect_objects(img, threshold)
    end = time.time()

    save_path = "static/detected_image/detected.jpg"
    cv2.imwrite(save_path, annotated if boxes else img)

    ###################
    # ## Send data from another server
    # # Prepare JSON payload
    # payload = {
    #     "num_detections": len(boxes),
    #     "bounding_boxes": boxes,
    #     "centers": centers,
    #     "confidences": confs,
    #     "processing_time_sec": round(end - start, 3)
    # }
    # # Send to other server
    # send_results_to_other_server(payload)

    #####################################
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": "✅ Objects detected" if boxes else "⚠️ No object detected",
        "num_detections": len(boxes),
        "bounding_boxes": boxes,
        "centers": centers,
        "confidences": confs,
        "processing_time_sec": round(end - start, 3),
        "image_url": "/" + save_path
    })


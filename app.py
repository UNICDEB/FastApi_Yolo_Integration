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

# import threading
# import time
# import cv2
# import numpy as np
# from fastapi import FastAPI, UploadFile, Form, Request
# from fastapi.responses import JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from ultralytics import YOLO
# import pyrealsense2 as rs
# import requests

# app = FastAPI()

# # Load the YOLOv8 model
# # model = YOLO("weights/best.pt")
# model = YOLO("weights/custom_loss_py_yolov8_best_100.pt")

# # Static and template config
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# # Global RealSense camera objects
# pipeline = None
# camera_thread = None
# stop_event = threading.Event()

# # Function to show live RealSense camera feed
# def live_camera_feed():
#     global pipeline, stop_event
#     pipeline = rs.pipeline()
#     config = rs.config()
#     config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
#     pipeline.start(config)

#     try:
#         while not stop_event.is_set():
#             frames = pipeline.wait_for_frames()
#             color_frame = frames.get_color_frame()
#             if not color_frame:
#                 continue
#             color_image = np.asanyarray(color_frame.get_data())
#             cv2.imshow("RealSense Live Feed - Press Q to close", color_image)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 stop_event.set()
#                 break
#     finally:
#         pipeline.stop()
#         cv2.destroyAllWindows()

# # Function to capture a frame from RealSense
# def capture_realsense_image(save_path):
#     global pipeline
#     if pipeline is None:
#         return None
#     for _ in range(5):  # Warm-up
#         pipeline.wait_for_frames()
#     frames = pipeline.wait_for_frames()
#     color_frame = frames.get_color_frame()
#     if not color_frame:
#         return None
#     color_image = np.asanyarray(color_frame.get_data())
#     cv2.imwrite(save_path, color_image)
#     return color_image

# # Object detection function
# def detect_objects(image, threshold):
#     results = model(image)
#     boxes = results[0].boxes.xyxy.tolist()
#     confidences = results[0].boxes.conf.tolist()
#     centers = []
#     filtered_boxes = []
#     filtered_confidences = []

#     annotated = image.copy()

#     for box, conf in zip(boxes, confidences):
#         if conf >= threshold:
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#             cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
#             centers.append([cx, cy])
#             filtered_boxes.append(box)
#             filtered_confidences.append(conf)

#     return annotated, filtered_boxes, centers, filtered_confidences

# ####################################
# ## Send data to another server function
# def send_results_to_other_server(data: dict):
#     url = "https://your-other-server.com/receive-data"   # Replace with your target server URL
#     try:
#         response = requests.post(url, json=data, timeout=5)
#         response.raise_for_status()
#         print("✅ Data sent successfully to other server.")
#     except requests.RequestException as e:
#         print(f"❌ Error sending data: {e}")

# #########################################################

# # Home page
# @app.get("/")
# async def home(request: Request):
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

# # Start live camera in background
# @app.post("/start-camera/")
# async def start_camera():
#     global camera_thread, stop_event
#     if camera_thread is None or not camera_thread.is_alive():
#         stop_event.clear()
#         camera_thread = threading.Thread(target=live_camera_feed, daemon=True)
#         camera_thread.start()
#         return JSONResponse({"status": "Camera started"})
#     else:
#         return JSONResponse({"status": "Camera already running"})

# # Capture image from RealSense and detect
# @app.post("/capture-detect/")
# async def capture_and_detect(request: Request, threshold: float = Form(0.5)):
#     save_path = "static/capture_image/frame.jpg"
#     captured_image = capture_realsense_image(save_path)
#     if captured_image is None:
#         return templates.TemplateResponse("index.html", {
#             "request": request,
#             "message": "❌ Failed to capture image. Start camera first.",
#             "num_detections": 0,
#             "bounding_boxes": [],
#             "centers": [],
#             "confidences": [],
#             "processing_time_sec": 0,
#             "image_url": None
#         })

#     start = time.time()
#     annotated, boxes, centers, confs = detect_objects(captured_image, threshold)
#     end = time.time()

#     output_path = "static/detected_image/detected.jpg"
#     cv2.imwrite(output_path, annotated if boxes else captured_image)

#     #######################################
#     # ## Send data to another server
#     # # Prepare JSON payload
#     # payload = {
#     #     "num_detections": len(boxes),
#     #     "bounding_boxes": boxes,
#     #     "centers": centers,
#     #     "confidences": confs,
#     #     "processing_time_sec": round(end - start, 3)
#     # }

#     # # Send to other server
#     # send_results_to_other_server(payload)

#     #############################################

#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "message": "✅ Objects detected" if boxes else "⚠️ No object detected",
#         "num_detections": len(boxes),
#         "bounding_boxes": boxes,
#         "centers": centers,
#         "confidences": confs,
#         "processing_time_sec": round(end - start, 3),
#         "image_url": "/" + output_path
#     })

# # Upload an image and detect
# @app.post("/detect/")
# async def detect_uploaded_image(request: Request, file: UploadFile, threshold: float = Form(0.5)):
#     contents = await file.read()
#     npimg = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#     start = time.time()
#     annotated, boxes, centers, confs = detect_objects(img, threshold)
#     end = time.time()

#     save_path = "static/detected_image/detected.jpg"
#     cv2.imwrite(save_path, annotated if boxes else img)

#     ###################
#     # ## Send data from another server
#     # # Prepare JSON payload
#     # payload = {
#     #     "num_detections": len(boxes),
#     #     "bounding_boxes": boxes,
#     #     "centers": centers,
#     #     "confidences": confs,
#     #     "processing_time_sec": round(end - start, 3)
#     # }
#     # # Send to other server
#     # send_results_to_other_server(payload)

#     #####################################
    
#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "message": "✅ Objects detected" if boxes else "⚠️ No object detected",
#         "num_detections": len(boxes),
#         "bounding_boxes": boxes,
#         "centers": centers,
#         "confidences": confs,
#         "processing_time_sec": round(end - start, 3),
#         "image_url": "/" + save_path
#     })


###############################
# Date: 03_10_2025

"""
app.py - FastAPI + Intel RealSense integration with depth capture and pixel->3D conversion

Features:
- Starts a background RealSense reader thread that continuously grabs color + depth frames.
- Provides an MJPEG endpoint (/video_feed) so the camera stream can be embedded into the webpage.
- /start-camera/ starts the reader thread, /stop-camera/ stops it.
- /capture-detect/ saves the latest RGB and depth frames, runs YOLO detection on RGB,
  computes center pixel for each detection, converts center pixel -> real-world (X,Y,Z) using
  RealSense intrinsics and depth, and returns detection + 3D points to the template.
- /detect/ works for uploaded images (no depth available, returns None for 3D points).

Notes:
- Ensure directories static/capture_image and static/detected_image exist and are writable.
- Requires: ultralytics, pyrealsense2, fastapi, uvicorn, opencv-python, numpy, requests
"""

import threading
import time
import os
import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import pyrealsense2 as rs
import requests

# ------------- Configuration / Setup -------------
app = FastAPI()

# Load YOLO model (change path if needed)
MODEL_PATH = "weights/custom_loss_py_yolov8_best_100.pt"
model = YOLO(MODEL_PATH)

# Create required directories if not present
os.makedirs("static/capture_image", exist_ok=True)
os.makedirs("static/detected_image", exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global camera/pipeline state
pipeline = None
reader_thread = None
stop_event = threading.Event()
frame_lock = threading.Lock()

# latest frames stored by the reader thread
latest_frame_bytes = None      # JPEG bytes for streaming
latest_color_frame = None      # numpy array BGR color
latest_depth_frame = None      # pyrealsense2 depth frame object

# ------------- RealSense Utilities -------------
def start_realsense_pipeline():
    """
    Configure and start the RealSense pipeline. Returns the profile object.
    Modify stream resolution/format if your camera is different.
    """
    global pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    # match resolution & fps to your camera & requirements
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)
    return profile

def stop_realsense_pipeline():
    """Stop the RealSense pipeline if running."""
    global pipeline
    try:
        if pipeline is not None:
            pipeline.stop()
    except Exception:
        pass
    pipeline = None

def realsense_reader():
    """
    Background thread function:
    Continuously grabs frames and stores them in global variables for streaming/capture.
    """
    global latest_frame_bytes, latest_color_frame, latest_depth_frame, stop_event, pipeline

    try:
        profile = start_realsense_pipeline()
    except Exception as e:
        print("Failed to start RealSense pipeline:", e)
        return

    try:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                # if missing frames, skip this iteration
                time.sleep(0.01)
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Encode color as JPEG for MJPEG streaming
            success, jpeg = cv2.imencode(".jpg", color_image)
            if not success:
                time.sleep(0.01)
                continue

            with frame_lock:
                latest_frame_bytes = jpeg.tobytes()
                latest_color_frame = color_image.copy()
                latest_depth_frame = depth_frame

            # small sleep to reduce CPU usage; adjust as needed
            time.sleep(0.01)
    finally:
        # Ensure pipeline is stopped on exit
        stop_realsense_pipeline()

# ------------- Streaming generator -------------
def mjpeg_generator():
    """
    Yields MJPEG frames for StreamingResponse.
    """
    global latest_frame_bytes, stop_event
    boundary = b"--frame"
    while not stop_event.is_set():
        with frame_lock:
            frame = latest_frame_bytes
        if frame is None:
            # if no frame yet, wait briefly
            time.sleep(0.05)
            continue
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        # control stream rate here
        time.sleep(0.03)

# ------------- Detection helpers -------------
def detect_objects(image: np.ndarray, threshold: float):
    """
    Run YOLO model on the BGR image and return annotated image, integer boxes, centers, and confidences.
    """
    # ultralytics model accepts numpy BGR images
    results = model(image)
    # results[0].boxes.xyxy, conf etc.
    boxes_raw = results[0].boxes.xyxy.tolist()
    confs_raw = results[0].boxes.conf.tolist()

    annotated = image.copy()
    filtered_boxes = []
    centers = []
    filtered_confs = []

    for box, conf in zip(boxes_raw, confs_raw):
        if conf >= threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
            filtered_boxes.append([x1, y1, x2, y2])
            centers.append([cx, cy])
            filtered_confs.append(round(float(conf), 3))

    return annotated, filtered_boxes, centers, filtered_confs

# ------------- Optional external server sender -------------
def send_results_to_other_server(data: dict):
    url = "https://your-other-server.com/receive-data"   # change accordingly
    try:
        r = requests.post(url, json=data, timeout=5)
        r.raise_for_status()
        print("Data sent successfully.")
    except requests.RequestException as e:
        print("Error sending data:", e)

# ------------- FastAPI routes -------------
@app.get("/")
async def home(request: Request):
    """Render home page with empty context initially."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": None,
        "num_detections": None,
        "bounding_boxes": [],
        "centers": [],
        "confidences": [],
        "processing_time_sec": None,
        "image_url": None,
        "real_points": []
    })

@app.post("/start-camera/")
async def start_camera():
    """Start the background RealSense reader thread and return status JSON."""
    global reader_thread, stop_event
    if reader_thread is None or not reader_thread.is_alive():
        stop_event.clear()
        reader_thread = threading.Thread(target=realsense_reader, daemon=True)
        reader_thread.start()
        return JSONResponse({"status": "Camera started"})
    else:
        return JSONResponse({"status": "Camera already running"})

@app.post("/stop-camera/")
async def stop_camera():
    """Stop the background reader thread and RealSense pipeline."""
    global stop_event
    stop_event.set()
    # small wait to let thread stop
    time.sleep(0.1)
    return JSONResponse({"status": "Camera stopped"})

@app.get("/video_feed")
async def video_feed():
    """MJPEG streaming endpoint for embedding the live camera into HTML."""
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/capture-detect/")
async def capture_and_detect(request: Request, threshold: float = Form(0.5)):
    """
    Capture latest frames (color + depth), save them, run detection on color,
    compute real-world points for centers, save annotated image, and render template.
    """
    global latest_color_frame, latest_depth_frame, pipeline

    # Obtain snapshot from latest frames protected by lock
    with frame_lock:
        color_snapshot = latest_color_frame.copy() if latest_color_frame is not None else None
        depth_frame = latest_depth_frame

    # If latest frames aren't available, try synchronous capture
    if color_snapshot is None or depth_frame is None:
        if pipeline is None:
            # camera not running
            return templates.TemplateResponse("index.html", {
                "request": request,
                "message": "❌ Failed to capture image. Start camera first.",
                "num_detections": 0,
                "bounding_boxes": [],
                "centers": [],
                "confidences": [],
                "processing_time_sec": 0,
                "image_url": None,
                "real_points": []
            })
        # synchronous capture
        frames = pipeline.wait_for_frames()
        color_frame_sync = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame_sync or not depth_frame:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "message": "❌ Failed to capture image frames.",
                "num_detections": 0,
                "bounding_boxes": [],
                "centers": [],
                "confidences": [],
                "processing_time_sec": 0,
                "image_url": None,
                "real_points": []
            })
        color_snapshot = np.asanyarray(color_frame_sync.get_data())

    # Save RGB image
    save_color_path = "static/capture_image/frame.jpg"
    cv2.imwrite(save_color_path, color_snapshot)

    # Save raw depth as numpy and a visualized depth image
    try:
        depth_array = np.asanyarray(depth_frame.get_data())
        save_depth_raw = "static/capture_image/depth_raw.npy"
        np.save(save_depth_raw, depth_array)

        # Normalize for visualization and apply colormap
        depth_vis = depth_array.astype(np.float32)
        # Normalize to 0-255 for viewing (avoid divide-by-zero)
        if np.max(depth_vis) - np.min(depth_vis) > 0:
            depth_norm = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
        else:
            depth_norm = depth_vis
        depth_norm = np.uint8(depth_norm)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        save_depth_vis = "static/capture_image/depth_vis.png"
        cv2.imwrite(save_depth_vis, depth_color)
    except Exception as e:
        # If depth saving fails, continue but mark depth_array as None
        print("Warning: failed to save depth frames:", e)
        depth_array = None

    # Run detection
    t0 = time.time()
    annotated, boxes, centers, confs = detect_objects(color_snapshot, threshold)
    t1 = time.time()

    # Save annotated image (or original if no boxes)
    output_path = "static/detected_image/detected.jpg"
    cv2.imwrite(output_path, annotated if boxes else color_snapshot)

    # Convert center pixels -> 3D coordinates (meters) using intrinsics & depth
    real_points = []
    try:
        # obtain depth intrinsics and scale
        depth_intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
        depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
    except Exception as e:
        depth_intrinsics = None
        depth_scale = None
        print("Warning: couldn't get depth intrinsics/scale:", e)

    if depth_intrinsics is not None:
        for (cx, cy) in centers:
            try:
                # get_distance returns meters directly (rs uses depth_scale internally)
                distance_m = depth_frame.get_distance(int(cx), int(cy))
                # Deproject pixel to 3D point (meters)
                point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [int(cx), int(cy)], distance_m)
                real_points.append([round(float(p), 4) for p in point_3d])  # round for neatness
            except Exception as e:
                # If depth missing at pixel (common for reflective or distant surfaces), append None
                real_points.append([None, None, None])
    else:
        # if no intrinsics -> return placeholders
        real_points = [[None, None, None] for _ in centers]

    # (Optional) send payload to other server
    # payload = {
    #     "num_detections": len(boxes),
    #     "bounding_boxes": boxes,
    #     "centers": centers,
    #     "confidences": confs,
    #     "real_points": real_points,
    #     "processing_time_sec": round(t1 - t0, 3)
    # }
    # send_results_to_other_server(payload)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": "✅ Objects detected" if boxes else "⚠️ No object detected",
        "num_detections": len(boxes),
        "bounding_boxes": boxes,
        "centers": centers,
        "confidences": confs,
        "processing_time_sec": round(t1 - t0, 3),
        "image_url": "/" + output_path,
        "real_points": real_points
    })


@app.post("/detect/")
async def detect_uploaded_image(request: Request, file: UploadFile = None, threshold: float = Form(0.5)):
    """
    Detect on an uploaded image (no depth available). Returns real_points as None placeholders.
    """
    if file is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "⚠️ No file uploaded.",
            "num_detections": 0,
            "bounding_boxes": [],
            "centers": [],
            "confidences": [],
            "processing_time_sec": 0,
            "image_url": None,
            "real_points": []
        })

    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    t0 = time.time()
    annotated, boxes, centers, confs = detect_objects(img, threshold)
    t1 = time.time()

    save_path = "static/detected_image/detected.jpg"
    cv2.imwrite(save_path, annotated if boxes else img)

    # No depth available -> placeholders
    real_points = [[None, None, None] for _ in centers]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": "✅ Objects detected" if boxes else "⚠️ No object detected",
        "num_detections": len(boxes),
        "bounding_boxes": boxes,
        "centers": centers,
        "confidences": confs,
        "processing_time_sec": round(t1 - t0, 3),
        "image_url": "/" + save_path,
        "real_points": real_points
    })


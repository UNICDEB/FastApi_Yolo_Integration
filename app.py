
# #############################
# # Date: 06_10_2025
# import threading
# import time
# import os
# import cv2
# import numpy as np
# import requests
# import httpx
# import pyrealsense2 as rs
# from fastapi import FastAPI, UploadFile, Form, Request
# from fastapi.responses import JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from ultralytics import YOLO

# # -------------------- Global Config --------------------
# MODEL_PATH = "weights/custom_loss_py_yolov8_best_100.pt"
# RECEIVER_URL = "http://10.240.9.18:5000"

# app = FastAPI()
# model = YOLO(MODEL_PATH)

# os.makedirs("static/capture_image", exist_ok=True)
# os.makedirs("static/detected_image", exist_ok=True)

# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# pipeline = None
# reader_thread = None
# stop_event = threading.Event()
# frame_lock = threading.Lock()

# latest_color_frame = None
# latest_depth_frame = None


# # -------------------- RealSense Helpers --------------------
# def start_realsense_pipeline():
#     global pipeline
#     pipeline = rs.pipeline()
#     config = rs.config()
#     config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
#     config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
#     return pipeline.start(config)


# def stop_realsense_pipeline():
#     global pipeline
#     if pipeline is not None:
#         try:
#             pipeline.stop()
#         except Exception:
#             pass
#     pipeline = None


# def realsense_opencv_view():
#     """OpenCV live preview loop that runs in a background thread."""
#     global latest_color_frame, latest_depth_frame

#     try:
#         start_realsense_pipeline()
#     except Exception as e:
#         print("❌ Failed to start RealSense pipeline:", e)
#         return

#     try:
#         while not stop_event.is_set():
#             frames = pipeline.wait_for_frames()
#             color_frame = frames.get_color_frame()
#             depth_frame = frames.get_depth_frame()
#             if not color_frame or not depth_frame:
#                 continue

#             color_image = np.asanyarray(color_frame.get_data())

#             with frame_lock:
#                 latest_color_frame = color_image.copy()
#                 latest_depth_frame = depth_frame

#             # Show in OpenCV window
#             cv2.imshow("Live Camera Feed", color_image)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#         cv2.destroyAllWindows()
#     finally:
#         stop_realsense_pipeline()


# # -------------------- Detection Helpers --------------------
# def detect_objects(image: np.ndarray, threshold: float):
#     results = model(image)
#     boxes_raw = results[0].boxes.xyxy.tolist()
#     confs_raw = results[0].boxes.conf.tolist()

#     annotated = image.copy()
#     boxes, centers, confidences = [], [], []

#     for box, conf in zip(boxes_raw, confs_raw):
#         if conf >= threshold:
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#             cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
#             boxes.append([x1, y1, x2, y2])
#             centers.append([cx, cy])
#             confidences.append(round(float(conf), 3))

#     return annotated, boxes, centers, confidences


# def compute_real_points(centers, depth_frame, intrinsics):
#     flat_points = []
#     valid_count = 0

#     for (cx, cy) in centers:
#         try:
#             dist_m = depth_frame.get_distance(int(cx), int(cy))
#             if np.isnan(dist_m) or dist_m <= 0 or dist_m==0:   # Skip NaN ,invalid & zero distance
#                 continue

#             point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [int(cx), int(cy)], dist_m)

#             # Convert to millimeters (int)
#             px, py, pz = [int(p * 1000) for p in point_3d]
#             flat_points.extend([px, py, pz])
#             valid_count += 1

#         except Exception:
#             continue  # skip errors silently

#     # First element = number of valid detections
#     return [valid_count] + flat_points if valid_count > 0 else []



# async def send_to_receiver(payload: dict):
#     try:
#         async with httpx.AsyncClient(timeout=5) as client:
#             r = await client.post(RECEIVER_URL, json=payload)
#             print("✅ Sent to receiver:", payload)
#             print("📩 Receiver response:", r.text)
#     except Exception as e:
#         print("❌ Send error:", e)
#         try:
#             r = requests.post(RECEIVER_URL, json=payload, timeout=5)
#             print("✅ Sent (fallback):", payload)
#             print("📩 Receiver response:", r.text)
#         except Exception as e2:
#             print("❌ Fallback send error:", e2)


# # -------------------- Routes --------------------
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
#         "image_url": None,
#         "real_points": []
#     })


# @app.post("/start-camera/")
# async def start_camera():
#     global reader_thread, stop_event
#     if reader_thread is None or not reader_thread.is_alive():
#         stop_event.clear()
#         reader_thread = threading.Thread(target=realsense_opencv_view, daemon=True)
#         reader_thread.start()
#         return JSONResponse({"status": "Camera started (OpenCV window opened)"})
#     return JSONResponse({"status": "Camera already running"})


# @app.post("/stop-camera/")
# async def stop_camera():
#     global stop_event
#     stop_event.set()
#     time.sleep(0.1)
#     cv2.destroyAllWindows()
#     return JSONResponse({"status": "Camera stopped"})


# @app.post("/capture-detect/")
# async def capture_and_detect(request: Request, threshold: float = Form(0.5)):
#     global latest_color_frame, latest_depth_frame, pipeline

#     with frame_lock:
#         color_snapshot = latest_color_frame.copy() if latest_color_frame is not None else None
#         depth_frame = latest_depth_frame

#     if color_snapshot is None or depth_frame is None:
#         return templates.TemplateResponse("index.html", {
#             "request": request,
#             "message": "❌ Failed to capture image. Start camera first.",
#             "num_detections": 0,
#             "bounding_boxes": [],
#             "centers": [],
#             "confidences": [],
#             "processing_time_sec": 0,
#             "image_url": None,
#             "real_points": []
#         })

#     cv2.imwrite("static/capture_image/frame.jpg", color_snapshot)

#     t0 = time.time()
#     annotated, boxes, centers, confs = detect_objects(color_snapshot, threshold)
#     t1 = time.time()

#     cv2.imwrite("static/detected_image/detected.jpg", annotated if boxes else color_snapshot)

#     real_points = []
#     try:
#         intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
#         real_points = compute_real_points(centers, depth_frame, intrinsics)
#     except Exception as e:
#         print("⚠️ Intrinsics error:", e)

#     if len(centers) > 0:
#         payload = {"message": "Objects detected", "centers": centers, "real_points": real_points}
#     else:
#         payload = {"message": "No Object Detected, Try Again"}

#     await send_to_receiver(payload)

#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "message": "✅ Objects detected" if boxes else "⚠️ No object detected",
#         "num_detections": len(boxes),
#         "bounding_boxes": boxes,
#         "centers": centers,
#         "confidences": confs,
#         "processing_time_sec": round(t1 - t0, 3),
#         "image_url": "/static/detected_image/detected.jpg",
#         "real_points": real_points
#     })


# @app.post("/detect/")
# async def detect_uploaded_image(request: Request, file: UploadFile = None, threshold: float = Form(0.5)):
#     if file is None:
#         return templates.TemplateResponse("index.html", {
#             "request": request,
#             "message": "⚠️ No file uploaded.",
#             "num_detections": 0,
#             "bounding_boxes": [],
#             "centers": [],
#             "confidences": [],
#             "processing_time_sec": 0,
#             "image_url": None,
#             "real_points": []
#         })

#     contents = await file.read()
#     npimg = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#     t0 = time.time()
#     annotated, boxes, centers, confs = detect_objects(img, threshold)
#     t1 = time.time()

#     cv2.imwrite("static/detected_image/detected.jpg", annotated if boxes else img)

#     real_points = [[None, None, None] for _ in centers]

#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "message": "✅ Objects detected" if boxes else "⚠️ No object detected",
#         "num_detections": len(boxes),
#         "bounding_boxes": boxes,
#         "centers": centers,
#         "confidences": confs,
#         "processing_time_sec": round(t1 - t0, 3),
#         "image_url": "/static/detected_image/detected.jpg",
#         "real_points": real_points
#     })


# # -------------------- Main --------------------
# def main():
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# if __name__ == "__main__":
#     main()

#################

# import threading
# import time
# import math
# import os
# import cv2
# import numpy as np
# import requests
# import httpx
# import pyrealsense2 as rs
# from fastapi import FastAPI, UploadFile, Form, Request
# from fastapi.responses import JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from ultralytics import YOLO
# from filterpy.kalman import KalmanFilter
# from collections import deque

# # -------------------- Global Config --------------------
# MODEL_PATH = "weights/custom_loss_py_yolov8_best_100.pt"
# RECEIVER_URL = "http://10.240.9.18:5000"

# app = FastAPI()
# model = YOLO(MODEL_PATH)

# os.makedirs("static/capture_image", exist_ok=True)
# os.makedirs("static/detected_image", exist_ok=True)

# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# pipeline = None
# reader_thread = None
# stop_event = threading.Event()
# frame_lock = threading.Lock()

# latest_color_frame = None
# latest_depth_frame = None

# # -------------------- RealSense Helpers --------------------
# def start_realsense_pipeline():
#     global pipeline
#     pipeline = rs.pipeline()
#     config = rs.config()
#     config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
#     config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
#     return pipeline.start(config)


# def stop_realsense_pipeline():
#     global pipeline
#     if pipeline is not None:
#         try:
#             pipeline.stop()
#         except Exception:
#             pass
#     pipeline = None


# def realsense_opencv_view():
#     global latest_color_frame, latest_depth_frame
#     try:
#         start_realsense_pipeline()
#     except Exception as e:
#         print("❌ Failed to start RealSense pipeline:", e)
#         return

#     try:
#         while not stop_event.is_set():
#             frames = pipeline.wait_for_frames()
#             color_frame = frames.get_color_frame()
#             depth_frame = frames.get_depth_frame()
#             if not color_frame or not depth_frame:
#                 continue

#             color_image = np.asanyarray(color_frame.get_data())

#             with frame_lock:
#                 latest_color_frame = color_image.copy()
#                 latest_depth_frame = depth_frame

#             cv2.imshow("Live Camera Feed", color_image)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#         cv2.destroyAllWindows()
#     finally:
#         stop_realsense_pipeline()


# # -------------------- Detection Helpers --------------------
# def detect_objects(image: np.ndarray, threshold: float):
#     results = model(image)
#     boxes_raw = results[0].boxes.xyxy.tolist()
#     confs_raw = results[0].boxes.conf.tolist()

#     annotated = image.copy()
#     boxes, centers, confidences = [], [], []

#     for box, conf in zip(boxes_raw, confs_raw):
#         if conf >= threshold:
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#             cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
#             boxes.append([x1, y1, x2, y2])
#             centers.append([cx, cy])
#             confidences.append(round(float(conf), 3))

#     return annotated, boxes, centers, confidences



# # -------------------- Filtering Helpers --------------------
# def init_kalman_filter():
#     """Initialize a simple 1D Kalman Filter for depth smoothing."""
#     kf = KalmanFilter(dim_x=2, dim_z=1)
#     kf.x = np.array([[0.], [0.]])  # initial state: pos, velocity
#     kf.F = np.array([[1., 1.],
#                      [0., 1.]])
#     kf.H = np.array([[1., 0.]])
#     kf.P *= 1000.
#     kf.R = 0.1
#     kf.Q = 1e-5 * np.eye(2)
#     return kf


# def filter_depth_kalman(depth_value, kf: KalmanFilter):
#     """Apply Kalman filter to a single depth measurement."""
#     kf.predict()
#     kf.update(np.array([depth_value]))
#     return kf.x[0, 0]


# # Maintain last N depth values for median filtering
# depth_history = {}  # key = point index, value = deque of last N depth readings
# MEDIAN_HISTORY = 5  # number of frames for median

# def compute_real_points_filtered(centers, depth_frame, intrinsics):
#     """
#     Compute 3D coordinates using both Kalman + Median filters.
#     Returns list: [valid_count, x1, y1, z1, x2, y2, z2, ...]
#     """
#     flat_points = []
#     valid_count = 0

#     # Initialize Kalman filter per detected point
#     kfs = [init_kalman_filter() for _ in centers]

#     for i, (cx, cy) in enumerate(centers):
#         try:
#             dist_m = depth_frame.get_distance(int(cx), int(cy))
#             if np.isnan(dist_m) or dist_m <= 0:
#                 continue

#             # --- Median Filtering ---
#             if i not in depth_history:
#                 depth_history[i] = deque(maxlen=MEDIAN_HISTORY)
#             depth_history[i].append(dist_m)
#             dist_m_median = np.median(depth_history[i])

#             # --- Kalman Filtering ---
#             dist_m_filtered = filter_depth_kalman(dist_m_median, kfs[i])

#             # Compute 3D point in meters
#             point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [int(cx), int(cy)], dist_m_filtered)

#             # Convert to millimeters (int)
#             px, py, pz = [int(p * 1000) for p in point_3d]
#             flat_points.extend([px, py, pz])
#             valid_count += 1

#         except Exception:
#             continue

#     return [valid_count] + flat_points if valid_count > 0 else []



# # -------------------- Sending Helpers --------------------
# async def send_to_receiver(payload: dict):
#     try:
#         async with httpx.AsyncClient(timeout=5) as client:
#             r = await client.post(RECEIVER_URL, json=payload)
#             print("✅ Sent to receiver:", payload)
#             print("📩 Receiver response:", r.text)
#     except Exception as e:
#         print("❌ Send error:", e)
#         try:
#             r = requests.post(RECEIVER_URL, json=payload, timeout=5)
#             print("✅ Sent (fallback):", payload)
#             print("📩 Receiver response:", r.text)
#         except Exception as e2:
#             print("❌ Fallback send error:", e2)


# # -------------------- Routes --------------------
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
#         "image_url": None,
#         "real_points": []
#     })


# @app.post("/start-camera/")
# async def start_camera():
#     global reader_thread, stop_event
#     if reader_thread is None or not reader_thread.is_alive():
#         stop_event.clear()
#         reader_thread = threading.Thread(target=realsense_opencv_view, daemon=True)
#         reader_thread.start()
#         return JSONResponse({"status": "Camera started (OpenCV window opened)"})
#     return JSONResponse({"status": "Camera already running"})


# @app.post("/stop-camera/")
# async def stop_camera():
#     global stop_event
#     stop_event.set()
#     time.sleep(0.1)
#     cv2.destroyAllWindows()
#     return JSONResponse({"status": "Camera stopped"})


# @app.post("/capture-detect/")
# async def capture_and_detect(request: Request, threshold: float = Form(0.5)):
#     global latest_color_frame, latest_depth_frame, pipeline

#     with frame_lock:
#         color_snapshot = latest_color_frame.copy() if latest_color_frame is not None else None
#         depth_frame = latest_depth_frame

#     if color_snapshot is None or depth_frame is None:
#         return templates.TemplateResponse("index.html", {
#             "request": request,
#             "message": "❌ Failed to capture image. Start camera first.",
#             "num_detections": 0,
#             "bounding_boxes": [],
#             "centers": [],
#             "confidences": [],
#             "processing_time_sec": 0,
#             "image_url": None,
#             "real_points": []
#         })

#     cv2.imwrite("static/capture_image/frame.jpg", color_snapshot)

#     t0 = time.time()
#     annotated, boxes, centers, confs = detect_objects(color_snapshot, threshold)
#     t1 = time.time()

#     cv2.imwrite("static/detected_image/detected.jpg", annotated if boxes else color_snapshot)

#     real_points = []
#     try:
#         intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
#         real_points = compute_real_points_filtered(centers, depth_frame, intrinsics)

#     except Exception as e:
#         print("⚠️ Intrinsics error:", e)

#     if len(centers) > 0:
#         payload = {"message": "Objects detected", "centers": centers, "real_points": real_points}
#     else:
#         payload = {"message": "No Object Detected, Try Again"}

#     await send_to_receiver(payload)

#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "message": "✅ Objects detected" if boxes else "⚠️ No object detected",
#         "num_detections": len(boxes),
#         "bounding_boxes": boxes,
#         "centers": centers,
#         "confidences": confs,
#         "processing_time_sec": round(t1 - t0, 3),
#         "image_url": "/static/detected_image/detected.jpg",
#         "real_points": real_points
#     })


# @app.post("/detect/")
# async def detect_uploaded_image(request: Request, file: UploadFile = None, threshold: float = Form(0.5)):
#     if file is None:
#         return templates.TemplateResponse("index.html", {
#             "request": request,
#             "message": "⚠️ No file uploaded.",
#             "num_detections": 0,
#             "bounding_boxes": [],
#             "centers": [],
#             "confidences": [],
#             "processing_time_sec": 0,
#             "image_url": None,
#             "real_points": []
#         })

#     contents = await file.read()
#     npimg = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#     t0 = time.time()
#     annotated, boxes, centers, confs = detect_objects(img, threshold)
#     t1 = time.time()

#     cv2.imwrite("static/detected_image/detected.jpg", annotated if boxes else img)

#     real_points = [[None, None, None] for _ in centers]

#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "message": "✅ Objects detected" if boxes else "⚠️ No object detected",
#         "num_detections": len(boxes),
#         "bounding_boxes": boxes,
#         "centers": centers,
#         "confidences": confs,
#         "processing_time_sec": round(t1 - t0, 3),
#         "image_url": "/static/detected_image/detected.jpg",
#         "real_points": real_points
#     })


# # -------------------- Main --------------------
# def main():
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# if __name__ == "__main__":
#     main()

##############
# Date: 09_10_2025 include all filter

import threading
import time
import math
import os
import cv2
import numpy as np
import requests
import httpx
import pyrealsense2 as rs
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from collections import deque

# -------------------- Global Config --------------------
MODEL_PATH = "weights/custom_loss_py_yolov8_best_100.pt"
RECEIVER_URL = "http://10.240.9.18:5000"

# Filters / smoothing params
MEDIAN_HISTORY = 5       # history length for median temporal filter per detection key
REGION_SIZE = 5          # sample region size (odd number preferred, e.g. 5x5)
KALMAN_R = 0.05          # measurement noise for Kalman
KALMAN_Q = 1e-5          # process noise scale
CENTER_QUANT = 4         # pixels: quantize center to this grid to maintain kf/history across frames

# Camera tilt angle (degrees). Set to actual mount: e.g. 90.0 for straight-down,
# 45.0 for 45° from horizontal, 110.0 for backward tilt etc.
CAMERA_TILT_ANGLE = 110.0

app = FastAPI()
model = YOLO(MODEL_PATH)

os.makedirs("static/capture_image", exist_ok=True)
os.makedirs("static/detected_image", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

pipeline = None
reader_thread = None
stop_event = threading.Event()
frame_lock = threading.Lock()

latest_color_frame = None
latest_depth_frame = None  # will store filtered depth frame

# For tracking Kalman and median history across frames keyed by quantized center
kf_dict = {}         # key -> KalmanFilter instance
depth_history = {}   # key -> deque of recent raw depth medians

# Precompute rotation convenience (not used directly here; rotation computed on demand)
# We treat 90° as the "no rotation" reference: rotation angle = (tilt - 90) degrees.


# -------------------- RealSense Helpers --------------------
def start_realsense_pipeline():
    """Start RealSense pipeline with recommended streams."""
    global pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    return pipeline.start(config)


def stop_realsense_pipeline():
    global pipeline
    if pipeline is not None:
        try:
            pipeline.stop()
        except Exception:
            pass
    pipeline = None


# RealSense post-processing filters: decimation, spatial, temporal, hole filling
decimation = rs.decimation_filter()
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()


def apply_realsense_filters(depth_frame):
    """
    Apply a standard filter chain to a depth frame to reduce noise and holes.
    Returns a filtered depth frame object.
    """
    try:
        f = decimation.process(depth_frame)
        f = spatial.process(f)
        f = temporal.process(f)
        f = hole_filling.process(f)
        return f
    except Exception:
        return depth_frame


def realsense_opencv_view():
    """OpenCV live preview loop that runs in a background thread."""
    global latest_color_frame, latest_depth_frame

    try:
        start_realsense_pipeline()
    except Exception as e:
        print("❌ Failed to start RealSense pipeline:", e)
        return

    try:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Apply filters to depth_frame for improved depth quality
            filtered_depth = apply_realsense_filters(depth_frame)

            color_image = np.asanyarray(color_frame.get_data())

            with frame_lock:
                latest_color_frame = color_image.copy()
                latest_depth_frame = filtered_depth  # store processed frame object

            cv2.imshow("Live Camera Feed", color_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
    finally:
        stop_realsense_pipeline()


# -------------------- Detection Helpers --------------------
def detect_objects(image: np.ndarray, threshold: float):
    results = model(image)
    boxes_raw = results[0].boxes.xyxy.tolist()
    confs_raw = results[0].boxes.conf.tolist()

    annotated = image.copy()
    boxes, centers, confidences = [], [], []

    for box, conf in zip(boxes_raw, confs_raw):
        if conf >= threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
            boxes.append([x1, y1, x2, y2])
            centers.append([cx, cy])
            confidences.append(round(float(conf), 3))

    return annotated, boxes, centers, confidences


# -------------------- Filtering Helpers --------------------
def init_kalman_filter(initial_depth=0.0):
    """Initialize a simple constant-velocity 1D Kalman Filter for depth smoothing."""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[initial_depth], [0.]])  # initial state: pos (depth), velocity
    kf.F = np.array([[1., 1.], [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 1000.0
    kf.R = KALMAN_R
    kf.Q = KALMAN_Q * np.eye(2)
    return kf


def filter_depth_kalman(depth_value, kf: KalmanFilter):
    """Apply Kalman filter to a single depth measurement (meters)."""
    try:
        kf.predict()
        kf.update(np.array([depth_value]))
        return float(kf.x[0, 0])
    except Exception:
        return depth_value


def quantize_center(cx, cy):
    """Quantize center to a coarse grid so that the same object maps to same key across frames."""
    qx = int(round(cx / CENTER_QUANT) * CENTER_QUANT)
    qy = int(round(cy / CENTER_QUANT) * CENTER_QUANT)
    return f"{qx}_{qy}"


def sample_region_median(depth_frame, cx, cy, intrinsics, region_size=REGION_SIZE):
    """
    Sample a small square region around (cx,cy) from the depth frame.
    Returns median distance in meters of valid pixels, or None if none valid.
    """
    half = region_size // 2
    vals = []
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            px = int(cx + dx)
            py = int(cy + dy)
            # bounds
            if px < 0 or py < 0 or px >= intrinsics.width or py >= intrinsics.height:
                continue
            try:
                d = depth_frame.get_distance(px, py)  # in meters
                if d is None:
                    continue
                if math.isnan(d) or d <= 0:
                    continue
                vals.append(d)
            except Exception:
                continue

    if len(vals) == 0:
        return None
    return float(np.median(vals))


def rotate_point_camera_to_world(px_mm, py_mm, pz_mm, tilt_deg=CAMERA_TILT_ANGLE):
    """
    Rotate point from camera coordinates to world coordinates.
    We treat CAMERA_TILT_ANGLE == 90.0 as the 'no rotation' reference.
    Rotation axis: X-axis (pitch). Rotation angle = (tilt_deg - 90) degrees.

    px_mm: camera X (left/right) in mm
    py_mm: camera Y (up/down) in mm
    pz_mm: camera Z (forward along optical axis) in mm

    Returns world (x, y, z) in mm (floats).
    """
    # If tilt is exactly 90 (or very close), do not rotate.
    angle_deg = tilt_deg - 90.0
    if abs(angle_deg) < 1e-3:
        return px_mm, py_mm, pz_mm

    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)

    # Rotation matrix Rx(theta) applied to (x_cam, y_cam, z_cam)
    # [1  0   0]   [x]
    # [0  c  -s] * [y]
    # [0  s   c]   [z]
    x_w = px_mm
    y_w = c * py_mm - s * pz_mm
    z_w = s * py_mm + c * pz_mm
    return x_w, y_w, z_w


def compute_real_points_filtered(centers, depth_frame, intrinsics):
    """
    Compute 3D coordinates using region sampling + Median history + Kalman filtering + conditional rotation correction.
    Returns list: [valid_count, x1, y1, z1, x2, y2, z2, ...] where coords are integers in millimeters.
    """
    flat_points = []
    valid_count = 0

    if depth_frame is None:
        return []

    for (cx, cy) in centers:
        try:
            key = quantize_center(cx, cy)

            # 1) Region median sampling
            region_median = sample_region_median(depth_frame, cx, cy, intrinsics, region_size=REGION_SIZE)
            if region_median is None or math.isnan(region_median) or region_median <= 0:
                # no valid depth in region
                continue

            # 2) temporal median history
            if key not in depth_history:
                depth_history[key] = deque(maxlen=MEDIAN_HISTORY)
            depth_history[key].append(region_median)
            temporal_median = float(np.median(depth_history[key]))

            # 3) Kalman filter per key
            if key not in kf_dict:
                kf_dict[key] = init_kalman_filter(initial_depth=temporal_median)
            depth_filtered_m = filter_depth_kalman(temporal_median, kf_dict[key])  # meters

            # 4) Deproject to camera-space (meters)
            point_3d_m = rs.rs2_deproject_pixel_to_point(intrinsics, [int(cx), int(cy)], depth_filtered_m)
            px_m, py_m, pz_m = float(point_3d_m[0]), float(point_3d_m[1]), float(point_3d_m[2])

            # 5) Convert to millimeters (float)
            px_mm = px_m * 1000.0
            py_mm = py_m * 1000.0
            pz_mm = pz_m * 1000.0

            # 6) Conditional rotation:
            # - If CAMERA_TILT_ANGLE == 90: no rotation (camera downwards)
            # - Else: rotate by (tilt - 90) about X axis (works for <90 and >90)
            x_w_mm, y_w_mm, z_w_mm = rotate_point_camera_to_world(px_mm, py_mm, pz_mm, tilt_deg=CAMERA_TILT_ANGLE)

            # 7) Store integer mm values (rounded)
            flat_points.extend([int(round(x_w_mm)), int(round(y_w_mm)), int(round(z_w_mm))])
            valid_count += 1

        except Exception as e:
            # Log but continue
            print("⚠️ compute_real_points_filtered exception for center", (cx, cy), ":", e)
            continue

    return [valid_count] + flat_points if valid_count > 0 else []


# -------------------- Sending Helpers --------------------
async def send_to_receiver(payload: dict):
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post(RECEIVER_URL, json=payload)
            print("✅ Sent to receiver:", payload)
            print("📩 Receiver response:", r.text)
    except Exception as e:
        print("❌ Send error:", e)
        try:
            r = requests.post(RECEIVER_URL, json=payload, timeout=5)
            print("✅ Sent (fallback):", payload)
            print("📩 Receiver response:", r.text)
        except Exception as e2:
            print("❌ Fallback send error:", e2)


# -------------------- Routes --------------------
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
        "image_url": None,
        "real_points": []
    })


@app.post("/start-camera/")
async def start_camera():
    global reader_thread, stop_event
    if reader_thread is None or not reader_thread.is_alive():
        stop_event.clear()
        reader_thread = threading.Thread(target=realsense_opencv_view, daemon=True)
        reader_thread.start()
        return JSONResponse({"status": "Camera started (OpenCV window opened)"})
    return JSONResponse({"status": "Camera already running"})


@app.post("/stop-camera/")
async def stop_camera():
    global stop_event
    stop_event.set()
    time.sleep(0.1)
    cv2.destroyAllWindows()
    return JSONResponse({"status": "Camera stopped"})


@app.post("/capture-detect/")
async def capture_and_detect(request: Request, threshold: float = Form(0.5)):
    global latest_color_frame, latest_depth_frame, pipeline

    with frame_lock:
        color_snapshot = latest_color_frame.copy() if latest_color_frame is not None else None
        depth_frame = latest_depth_frame

    if color_snapshot is None or depth_frame is None:
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

    cv2.imwrite("static/capture_image/frame.jpg", color_snapshot)

    t0 = time.time()
    annotated, boxes, centers, confs = detect_objects(color_snapshot, threshold)
    t1 = time.time()

    cv2.imwrite("static/detected_image/detected.jpg", annotated if boxes else color_snapshot)

    real_points = []
    try:
        intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
        real_points = compute_real_points_filtered(centers, depth_frame, intrinsics)
    except Exception as e:
        print("⚠️ Intrinsics error:", e)

    if len(centers) > 0:
        payload = {"message": "Objects detected", "centers": centers, "real_points": real_points}
    else:
        payload = {"message": "No Object Detected, Try Again"}

    await send_to_receiver(payload)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": "✅ Objects detected" if boxes else "⚠️ No object detected",
        "num_detections": len(boxes),
        "bounding_boxes": boxes,
        "centers": centers,
        "confidences": confs,
        "processing_time_sec": round(t1 - t0, 3),
        "image_url": "/static/detected_image/detected.jpg",
        "real_points": real_points
    })


@app.post("/detect/")
async def detect_uploaded_image(request: Request, file: UploadFile = None, threshold: float = Form(0.5)):
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

    cv2.imwrite("static/detected_image/detected.jpg", annotated if boxes else img)

    real_points = [[None, None, None] for _ in centers]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": "✅ Objects detected" if boxes else "⚠️ No object detected",
        "num_detections": len(boxes),
        "bounding_boxes": boxes,
        "centers": centers,
        "confidences": confs,
        "processing_time_sec": round(t1 - t0, 3),
        "image_url": "/static/detected_image/detected.jpg",
        "real_points": real_points
    })


# -------------------- Main --------------------
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

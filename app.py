
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

import threading
import time
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

# -------------------- Global Config --------------------
MODEL_PATH = "weights/custom_loss_py_yolov8_best_100.pt"
RECEIVER_URL = "http://10.240.9.18:5000"

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
latest_depth_frame = None

# -------------------- RealSense Helpers --------------------
def start_realsense_pipeline():
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


def realsense_opencv_view():
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

            color_image = np.asanyarray(color_frame.get_data())

            with frame_lock:
                latest_color_frame = color_image.copy()
                latest_depth_frame = depth_frame

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
def init_kalman_filter():
    """Initialize a simple 1D Kalman Filter for distance smoothing."""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.], [0.]])  # initial state: pos, velocity
    kf.F = np.array([[1., 1.],
                     [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 1000.
    kf.R = 0.1
    kf.Q = 1e-5 * np.eye(2)
    return kf


def filter_depth_kalman(depth_value, kf: KalmanFilter):
    """Apply Kalman filter to a single depth measurement."""
    kf.predict()
    kf.update(np.array([depth_value]))
    return kf.x[0, 0]


def compute_real_points_filtered(centers, depth_frame, intrinsics):
    """Compute 3D points using filtered depth (Kalman + median)."""
    flat_points = []
    valid_count = 0
    kfs = [init_kalman_filter() for _ in centers]  # one Kalman per detected point

    for i, (cx, cy) in enumerate(centers):
        try:
            dist_m = depth_frame.get_distance(int(cx), int(cy))
            if np.isnan(dist_m) or dist_m <= 0:
                continue

            # Apply Kalman filter
            dist_m_filtered = filter_depth_kalman(dist_m, kfs[i])

            point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [int(cx), int(cy)], dist_m_filtered)

            # Convert to mm
            px, py, pz = [int(p * 1000) for p in point_3d]
            flat_points.extend([px, py, pz])
            valid_count += 1
        except Exception:
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

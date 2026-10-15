# Reflex Vision — Adhyayan Tiwari
# Recorded-video baseline with GPU timing and CSV logging.

import csv
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from ultralytics import YOLO


MODEL = "yolo26n_fp32.engine"
IMAGE_SIZE = 640
PRECISION = 32
WARMUP_FRAMES = 30

PROJECT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = PROJECT_DIR / "test_video.mp4"
RESULTS_DIR = PROJECT_DIR / "results"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected. Check your environment.")

if not VIDEO_PATH.is_file():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

model = YOLO(str(PROJECT_DIR / MODEL), task="detect")
video = cv2.VideoCapture(str(VIDEO_PATH))

if not video.isOpened():
    video.release()
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

RESULTS_DIR.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output = RESULTS_DIR / f"video_{Path(MODEL).stem}_{IMAGE_SIZE}_fp{PRECISION}_{stamp}.csv"

frame_number = 0
timings = []
stage_timings = []

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Configuration: {IMAGE_SIZE} x {IMAGE_SIZE}, FP{PRECISION}")
print(f"First {WARMUP_FRAMES} frames are excluded as warm-up.")
print("Let the video finish, or press Q in the video window to stop.")

try:
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame",
            "prediction_ms",
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
            "detections",
        ])

        while True:
            success, frame = video.read()

            if not success:
                print("Video ended or a frame could not be decoded.")
                break

            # Measure the prediction call, waiting for GPU completion.
            torch.cuda.synchronize()
            start = perf_counter()

            result = model.predict(
                source=frame,
                device=0,
                imgsz=IMAGE_SIZE,
                rect=False,
                quantize=PRECISION,
                conf=0.25,
                verbose=False,
            )[0]

            torch.cuda.synchronize()
            prediction_ms = (perf_counter() - start) * 1000
            frame_number += 1

            preprocess_ms = result.speed["preprocess"]
            inference_ms = result.speed["inference"]
            postprocess_ms = result.speed["postprocess"]

            # Save measurements only after warm-up.
            if frame_number > WARMUP_FRAMES:
                timings.append(prediction_ms)
                stage_timings.append([
                    preprocess_ms,
                    inference_ms,
                    postprocess_ms,
                ])

                writer.writerow([
                    frame_number,
                    round(prediction_ms, 3),
                    round(preprocess_ms, 3),
                    round(inference_ms, 3),
                    round(postprocess_ms, 3),
                    len(result.boxes),
                ])

                label = f"Prediction: {prediction_ms:.1f} ms"
            else:
                label = f"Warming up: {frame_number}/{WARMUP_FRAMES}"

            # Drawing and display are outside the prediction timer.
            display = result.plot()
            cv2.putText(
                display,
                label,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Reflex Vision | Q to quit", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break

finally:
    video.release()
    cv2.destroyAllWindows()

if timings:
    stages = np.median(stage_timings, axis=0)

    print(f"\nMeasured frames: {len(timings)}")
    print(f"Median prediction: {np.median(timings):.2f} ms")
    print(f"95th percentile: {np.percentile(timings, 95):.2f} ms")
    print(f"Median preprocessing: {stages[0]:.2f} ms")
    print(f"Median inference: {stages[1]:.2f} ms")
    print(f"Median postprocessing: {stages[2]:.2f} ms")
    print(f"Results saved: {output.resolve()}")
else:
    print("No measurements collected after warm-up.")
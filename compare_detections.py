# Reflex Vision — Adhyayan Tiwari
# Compare PyTorch and TensorRT detections across the entire video.

import csv
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT = ROOT / "results" / f"agreement_{stamp}"
OUTPUT.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected.")

pt_model = YOLO(str(ROOT / "yolo26n.pt"))
trt_model = YOLO(str(ROOT / "yolo26n_fp32.engine"), task="detect")

video = cv2.VideoCapture(str(ROOT / "test_video.mp4"))
if not video.isOpened():
    video.release()
    raise RuntimeError("Could not open test_video.mp4.")


def predict(model, frame):
    return model.predict(
        frame, device=0, imgsz=640, rect=False,
        quantize=32, conf=0.25, iou=0.7,
        max_det=300, verbose=False,
    )[0]


def overlap(a, b):
    # Intersection over Union: shared box area / combined box area.
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(a, b):
    # Greedy one-to-one matching: same class, highest IoU first.
    candidates = []
    for i, box_a in enumerate(a):
        for j, box_b in enumerate(b):
            if int(box_a[5]) == int(box_b[5]):
                score = overlap(box_a[:4], box_b[:4])
                if score >= 0.5:
                    candidates.append((score, i, j))

    used_a, used_b, matches = set(), set(), []
    for score, i, j in sorted(candidates, reverse=True):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            difference = abs(float(a[i, 4]) - float(b[j, 4]))
            matches.append((score, difference))

    return matches, len(a) - len(used_a), len(b) - len(used_b)


def panel(result, title):
    picture = result.plot()
    height, width = picture.shape[:2]
    picture = cv2.resize(picture, (640, round(height * 640 / width)))
    picture = cv2.copyMakeBorder(
        picture, 50, 0, 0, 0,
        cv2.BORDER_CONSTANT, value=(25, 25, 25),
    )
    cv2.putText(
        picture, title, (12, 33),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2,
    )
    return picture


frames = 0
flagged_frames = 0
saved_images = 0
total_pt = 0
total_trt = 0
unmatched_pt = 0
unmatched_trt = 0
all_ious = []
all_conf_differences = []

try:
    with (OUTPUT / "frame_summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame", "pytorch_count", "tensorrt_count",
            "matched", "unmatched_pytorch", "unmatched_tensorrt",
            "minimum_iou", "maximum_confidence_difference", "flagged",
        ])

        while True:
            success, frame = video.read()
            if not success:
                break

            pt_result = predict(pt_model, frame)
            trt_result = predict(trt_model, frame)
            a = pt_result.boxes.data.cpu().numpy()
            b = trt_result.boxes.data.cpu().numpy()

            matches, missing_a, missing_b = match_boxes(a, b)
            ious = [pair[0] for pair in matches]
            differences = [pair[1] for pair in matches]

            flagged = (
                missing_a > 0
                or missing_b > 0
                or any(value < 0.90 for value in ious)
                or any(value > 0.02 for value in differences)
            )

            writer.writerow([
                frames, len(a), len(b), len(matches),
                missing_a, missing_b,
                min(ious) if ious else "",
                max(differences) if differences else "",
                flagged,
            ])

            if flagged:
                flagged_frames += 1
                if saved_images < 20:
                    comparison = cv2.hconcat([
                        panel(pt_result, f"PyTorch | Frame {frames}"),
                        panel(trt_result, f"TensorRT | Frame {frames}"),
                    ])
                    path = OUTPUT / f"difference_{frames:06d}.jpg"
                    if not cv2.imwrite(str(path), comparison):
                        raise RuntimeError(f"Could not save {path}")
                    saved_images += 1

            total_pt += len(a)
            total_trt += len(b)
            unmatched_pt += missing_a
            unmatched_trt += missing_b
            all_ious.extend(ious)
            all_conf_differences.extend(differences)
            frames += 1

            if frames % 100 == 0:
                print(f"Checked {frames} frames...")

finally:
    video.release()

print(f"\nFrames checked: {frames}")
print(f"Flagged frames: {flagged_frames}")
print(f"Total PyTorch detections: {total_pt}")
print(f"Total TensorRT detections: {total_trt}")
print(f"Matched detection pairs: {len(all_ious)}")
print(f"Unmatched PyTorch detections: {unmatched_pt}")
print(f"Unmatched TensorRT detections: {unmatched_trt}")

if all_ious:
    print(f"Median matched-box IoU: {np.median(all_ious):.4f}")
    print(f"Lowest matched-box IoU: {min(all_ious):.4f}")
    print(
        "Largest confidence difference: "
        f"{100 * max(all_conf_differences):.3f} percentage points"
    )

print(f"Difference images saved: {saved_images}")
print(f"Results folder: {OUTPUT}")
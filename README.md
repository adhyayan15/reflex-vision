# Reflex Vision
GPU object-detection benchmarking with PyTorch and NVIDIA TensorRT.

**Author:** Adhyayan Tiwari

## Goal
Investigate how inference backend, input resolution, and numerical
precision affect object-detection latency on an RTX 5060 Laptop GPU.

This project uses pretrained YOLO26n. My work focuses on the video
pipeline, timing instrumentation, backend comparisons, detection
agreement checks, and analysis—not training the detector.

## Environment
- Windows
- NVIDIA GeForce RTX 5060 Laptop GPU, approximately 8 GB VRAM
- NVIDIA driver 576.76
- Python 3.12.10
- PyTorch 2.7.1+cu128
- Ultralytics 8.4.177
- TensorRT 10.13.0.35

Exact installed package versions are recorded in requirements-lock.txt.

## Benchmark method
- Same recorded video for every configuration.
- Batch size 1.
- Fixed square model input using rect=False.
- Confidence threshold: 0.25.
- First 30 video frames excluded as warm-up.
- 604 subsequent frames measured per run.
- CUDA synchronization before and after each timed prediction call.
- Median and 95th-percentile prediction latency reported.
- Preprocessing, inference, and postprocessing timings also recorded.

Prediction timing excludes video decoding, annotation, CSV writing,
and display. Those operations still occur between predictions and
can affect execution conditions.

These measurements are not camera-to-display latency or application FPS.

## Main results
640 × 640 input, FP32 configuration/export:

| Backend | Median prediction | P95 prediction | Median inference |
|---|---:|---:|---:|
| PyTorch | 9.90 ms | 10.78 ms | 6.36 ms |
| TensorRT, run 1 | 5.57 ms | 7.42 ms | 1.63 ms |
| TensorRT, run 2 | 5.55 ms | 7.48 ms | 1.63 ms |

Relative to the listed PyTorch run, TensorRT reduced median prediction
latency by approximately 44%. Median model inference was approximately
3.9× faster.

These results describe this video, device, and software environment.
They are initial measurements, not a comprehensive statistical study.

FP32 labels describe the selected configuration/export. Internal kernel
arithmetic, including possible TF32 use, was not independently audited.

## Other experiments
Initial PyTorch runs:

| Configuration | Median prediction |
|---|---:|
| 640 × 640, FP32 | Approximately 10–10.5 ms across runs |
| 480 × 480, FP32 | 9.62 ms |
| 320 × 320, FP32 | 9.42 ms |
| 640 × 640, FP16, without deprecation warnings | 11.13 ms |

Smaller inputs provided modest latency improvements. Possible missed
objects were noticed visually but were not quantitatively evaluated.
FP16 did not improve PyTorch prediction latency in the observed runs.

## Detection agreement
Compared PyTorch and TensorRT FP32 on all 634 readable video frames.

Boxes were matched greedily, one-to-one, within the same class,
prioritizing highest Intersection over Union (IoU), with IoU >= 0.5.

Frames were flagged for:
- Unmatched detections.
- Matched-box IoU below 0.90.
- Confidence difference above 0.02.

Results:
- PyTorch detections: 959.
- TensorRT detections: 961.
- Matched detection pairs: 959.
- Unmatched PyTorch detections: 0.
- Unmatched TensorRT detections: 2.
- Median matched-box IoU: 0.9998.
- Lowest matched-box IoU: 0.9763.
- Largest matched confidence difference: 0.840 percentage points.

Manual inspection of the two flagged frames:
- Frame 70: TensorRT incorrectly labeled a wall as a refrigerator.
- Frame 113: TensorRT detected a real chair that PyTorch missed.

Both additional detections displayed confidence of 0.25, near the
configured cutoff. Small numerical differences are a possible
explanation, but the exact cause was not established.

These are detection-agreement measurements, not ground-truth accuracy.
Matching PyTorch does not prove a detection is correct.

## Files
- reflex_vision.py — video detection and timing measurements.
- compare_detections.py — full-video backend agreement checks.
- plot_results.py — charts and summary generated from selected CSVs.
- requirements-lock.txt — installed package versions.
- results/ — timing CSVs, comparisons, and charts.
- test_video.mp4 — local benchmark input.
- yolo26n.pt — pretrained checkpoint.
- yolo26n_fp32.engine — preserved TensorRT FP32 engine.

TensorRT engines depend on the build hardware and runtime.
Rebuild for the intended deployment environment.

## Setup
Create and activate a Python 3.12 environment in Command Prompt:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-lock.txt
```

Place test_video.mp4 in the project folder.

The lock file records the original environment. Recreating it on a
different machine may require compatible GPU drivers and runtime setup.

## Run the benchmark
For PyTorch, set in reflex_vision.py:

```python
MODEL = "yolo26n.pt"
IMAGE_SIZE = 640
PRECISION = 32
```

For the preserved TensorRT FP32 engine:

```python
MODEL = "yolo26n_fp32.engine"
IMAGE_SIZE = 640
PRECISION = 32
```

CSV filenames automatically include the model name and selected precision.
Run:

```bat
python reflex_vision.py
```

Allow the full video to finish.

## Build a TensorRT FP32 engine
The original successful export used:

```bat
yolo export model=yolo26n.pt format=engine device=0 imgsz=640 batch=1 dynamic=False quantize=32 workspace=2
copy yolo26n.engine yolo26n_fp32.engine
```

Export writes yolo26n.engine. Preserve existing engines before exporting
another configuration.

## Compare detections
In compare_detections.py, point the TensorRT model at the preserved
FP32 engine:

```python
trt_model = YOLO(str(ROOT / "yolo26n_fp32.engine"), task="detect")
```

Then run:

```bat
python compare_detections.py
```

The script writes a per-frame summary and up to 20 flagged images.

## Generate charts
```bat
python plot_results.py
```

The script reads three explicitly named CSV files, verifies matching
frame sequences, and saves charts and summary.csv to results/charts.

For new experiments, update its FILES mapping to the intended CSVs.
Stage medians are independent statistics and are not stacked as totals.

## FP16 TensorRT status
An FP16 engine build was attempted but interrupted by a Windows crash.

Windows recorded BugcheckCode 278 (0x116, VIDEO_TDR_FAILURE):
display-driver timeout recovery failed. The underlying cause was not
established. The GPU responded after reboot.

No completed TensorRT FP16 benchmark is reported. Further FP16 builds
are paused pending investigation.

## Limitations and next steps
- Evaluate both backends on a labeled dataset.
- Test additional videos and more diverse scenes.
- Automate repeated, interleaved benchmark runs.
- Record power mode, temperature, and clock behavior.
- Measure GPU memory and full application throughput.
- Investigate the crash before resuming TensorRT FP16 export.

## Attribution
Pretrained YOLO26 and its Python interface are provided by Ultralytics.
GPU execution uses PyTorch/CUDA and NVIDIA TensorRT.
Consult the respective projects for licensing and usage terms.

## Charts

![Prediction latency](results/charts/prediction_latency.png)

![Processing stages](results/charts/processing_stages.png)



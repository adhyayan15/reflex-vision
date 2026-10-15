# Reflex Vision — Adhyayan Tiwari
# Create comparison charts from saved measurements. No GPU required.

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"
CHARTS.mkdir(exist_ok=True)

FILES = {
    "PyTorch": "video_640_fp32_20261015_011603.csv",
    "TensorRT run 1": "video_tensorrt_640_fp32_20261015_013305.csv",
    "TensorRT run 2": "video_tensorrt_640_fp32_20261015_013453.csv",
}

COLUMNS = [
    "prediction_ms",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
]

data = {}
frame_ids = {}

for label, filename in FILES.items():
    with (RESULTS / filename).open(newline="") as file:
        reader = csv.DictReader(file)
        required = set(COLUMNS + ["frame"])
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing timing columns in {filename}")

        rows = list(reader)

    if not rows:
        raise ValueError(f"No measurements in {filename}")

    data[label] = np.array([
        [float(row[column]) for column in COLUMNS]
        for row in rows
    ])
    frame_ids[label] = [int(row["frame"]) for row in rows]

    if not np.isfinite(data[label]).all():
        raise ValueError(f"Invalid measurements in {filename}")

labels = list(data)
reference_frames = frame_ids[labels[0]]

if any(frame_ids[label] != reference_frames for label in labels):
    raise ValueError("Runs contain different frame sequences.")

medians = np.array([
    np.median(data[label], axis=0) for label in labels
])
p95 = np.array([
    np.percentile(data[label][:, 0], 95) for label in labels
])

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Chart 1: overall prediction latency.
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(
    x - 0.18, medians[:, 0], 0.36,
    label="Median", color="#2563eb",
)
bars2 = ax.bar(
    x + 0.18, p95, 0.36,
    label="95th percentile", color="#f59e0b",
)
ax.bar_label(bars1, fmt="%.2f", padding=3)
ax.bar_label(bars2, fmt="%.2f", padding=3)
ax.set_xticks(x, labels)
ax.set_ylabel("Prediction time (ms) — lower is better")
ax.set_title("Reflex Vision | Prediction latency")
ax.set_ylim(0, max(p95) * 1.25)
ax.legend()
fig.text(
    0.5, 0.02,
    "RTX 5060 Laptop | 640 × 640 | FP32 export/configuration\n"
    f"{len(reference_frames)} measured frames per run | "
    "Capture, drawing and display excluded",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=(0, 0.10, 1, 1))
fig.savefig(CHARTS / "prediction_latency.png", dpi=200)
plt.close(fig)

# Chart 2: separate stage medians, not stacked totals.
stages = ["Preprocessing", "Inference", "Postprocessing"]
x = np.arange(len(stages))
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#2563eb", "#16a34a", "#0d9488"]

for i, label in enumerate(labels):
    bars = ax.bar(
        x + (i - 1) * 0.25,
        medians[i, 1:],
        width=0.25,
        label=label,
        color=colors[i],
    )
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

ax.set_xticks(x, stages)
ax.set_ylabel("Median time (ms) — lower is better")
ax.set_title("Reflex Vision | Processing stages")
ax.set_ylim(0, medians[:, 1:].max() * 1.3)
ax.legend()
fig.text(
    0.5, 0.02,
    "Stage medians are calculated independently; "
    "their sum is not the median total.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(CHARTS / "processing_stages.png", dpi=200)
plt.close(fig)

# Preserve the calculated summary and source filenames.
with (CHARTS / "summary.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "run", "source_file", "frames", "median_prediction_ms",
        "p95_prediction_ms", "median_preprocess_ms",
        "median_inference_ms", "median_postprocess_ms",
    ])
    for i, label in enumerate(labels):
        writer.writerow([
            label, FILES[label], len(data[label]),
            medians[i, 0], p95[i], *medians[i, 1:],
        ])
        print(
            f"{label}: median={medians[i, 0]:.2f} ms, "
            f"p95={p95[i]:.2f} ms"
        )

print(f"\nCharts and summary saved to: {CHARTS}")
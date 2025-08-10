# train_yolo.py
# Optimized for Apple Silicon (M1/M2/M3/M4) + MPS with verbose device info.

from ultralytics import YOLO
import torch
import platform

# ---- Hardware/device detection ----
def pick_device(verbose=True):
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        if verbose:
            print("✅ MPS (Apple GPU) is available and will be used for training.")
        return "mps"
    else:
        if verbose:
            print("⚠️  MPS not available — training will run on CPU.")
            if not torch.backends.mps.is_built():
                print("   (PyTorch not built with MPS support)")
            elif not torch.backends.mps.is_available():
                print("   (MPS driver/hardware not accessible)")
        return "cpu"

device = pick_device(verbose=True)
is_mac = (platform.system() == "Darwin")
workers = 2 if (device == "mps" and is_mac) else 4

print(f"[Config] Device: {device} | Dataloader workers: {workers} | macOS: {is_mac}")
print(f"[Config] PyTorch: {torch.__version__} | Python: {platform.python_version()} ({platform.machine()})")

# ---- Load YOLO model ----
model = YOLO("yolov8n.pt")   # change to 'yolov8s.pt' for more accuracy

# ---- Train ----
model.train(
    data="yolo_logo.yaml",
    epochs=40,
    imgsz=1024,
    batch=16,
    device=device,
    workers=workers,
    optimizer="AdamW",
    lr0=0.002,
    lrf=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    patience=10,       # early stop if no val mAP improvement for 10 epochs
    fliplr=0.0,        # avoid flipping text/logos
    mosaic=0.7,
    mixup=0.1,
    cache=True,        # speeds up macOS I/O
    verbose=True       # YOLO prints per-batch info
)
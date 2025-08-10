import os, glob
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

CONF_THRES = 0.25
IOU_THRES  = 0.45
PAD        = 8

def exif_fix(pil):
    return ImageOps.exif_transpose(pil)

def crop_pad(bgr, box, pad=8):
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1-pad); y1 = max(0, y1-pad)
    x2 = min(w, x2+pad); y2 = min(h, y2+pad)
    return bgr[y1:y2, x1:x2]

def load_gallery(logos_dir):
    """Load clean logos and precompute ORB descriptors."""
    orb = cv2.ORB_create(nfeatures=1500)
    gallery = []
    for p in sorted(Path(logos_dir).glob("*")):
        img = Image.open(p).convert("RGB")
        img = exif_fix(img)
        g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        ggray = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        kps, des = orb.detectAndCompute(ggray, None)
        if des is not None:
            gallery.append((p.name, g, des))
    if not gallery:
        raise RuntimeError("No valid logos in gallery")
    return gallery, orb

def identify_logo(crop_bgr, gallery, orb):
    """Return best-matching gallery filename and a score (higher is better)."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    kps, des = orb.detectAndCompute(gray, None)
    if des is None or len(des) < 10:
        return None, 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_name, best_score = None, -1
    for name, gimg, gdes in gallery:
        matches = bf.match(des, gdes)
        if not matches: 
            continue
        # quality = high # matches and low distance
        score = sum(1.0/(m.distance+1e-6) for m in sorted(matches, key=lambda m: m.distance)[:60])
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, float(best_score)

def main(img_path, weights="runs/detect/train/weights/best.pt", logos_dir="logos", out_dir="out"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    gallery, orb = load_gallery(logos_dir)

    pil = Image.open(img_path).convert("RGB")
    pil = exif_fix(pil)
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    res = model(bgr, conf=CONF_THRES, iou=IOU_THRES, imgsz=1280, verbose=False)[0]
    if len(res.boxes) == 0:
        print("No logo detected.")
        return

    for i, box in enumerate(res.boxes.xyxy.cpu().numpy()):
        crop = crop_pad(bgr, box, PAD)
        name, score = identify_logo(crop, gallery, orb)
        tag = name if name else f"candidate_{i+1}"
        outp = Path(out_dir) / f"{Path(img_path).stem}_{tag}.png"
        cv2.imwrite(str(outp), crop)
        print(f"Saved {outp}  (match: {name}, score: {score:.2f})")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
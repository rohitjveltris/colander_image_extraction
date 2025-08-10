import os, random, uuid
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from tqdm import trange

ROOT = Path(__file__).resolve().parent
LOGO_DIR = ROOT / "logos"
BKG_DIR  = ROOT / "invoices_raw"
OUT_DIR  = ROOT / "data"

# how many to make
N_TRAIN = 2000
N_VAL   = 300
IM_SIZE = 1600     # synth canvas size (square simplifies scaling)

random.seed(1337)

def load_rgba(folder: Path):
    out = []
    for p in folder.glob("*"):
        try:
            img = Image.open(p)
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            out.append(img)
        except Exception:
            pass
    if not out:
        raise RuntimeError(f"No readable images in {folder} — check file types/permissions")
    return out

def random_logo_transform(logo: Image.Image, cw, ch):
    # scale w.r.t canvas width; logos typically small
    if logo.mode != "RGBA":
        logo = logo.convert("RGBA")

    w_target = int(cw * random.uniform(0.10, 0.22))
    r = w_target / logo.width
    h_target = max(1, int(logo.height * r))
    L = logo.resize((w_target, h_target), Image.LANCZOS)

    # --- apply ops on RGB only, keep alpha ---
    rgb = L.convert("RGB")
    a   = L.split()[3]  # alpha

    if random.random() < 0.5:
        rgb = ImageOps.autocontrast(rgb)            # safe now
    if random.random() < 0.25:
        rgb = rgb.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))

    # recombine RGB + original alpha
    L = Image.merge("RGBA", (*rgb.split(), a))
    return L

def place_logo(canvas: Image.Image, L: Image.Image):
    cw, ch = canvas.size
    # mostly top-left; sometimes top-center/right
    x = int(random.uniform(0.01, 0.22) * cw)
    y = int(random.uniform(0.01, 0.18) * ch)
    if random.random() < 0.15:
        x = int(random.uniform(0.35, 0.75) * cw)
    canvas.alpha_composite(L, (x, y))
    # YOLO bbox (normalized)
    cx = (x + L.width/2) / cw
    cy = (y + L.height/2) / ch
    return cx, cy, L.width/cw, L.height/ch

def make_split(split, n):
    (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    logos = load_rgba(LOGO_DIR)
    bkgs  = load_rgba(BKG_DIR)

    for _ in trange(n, desc=f"gen {split}"):
        bg = random.choice(bkgs).copy()
        # square canvas; fit background
        canvas = Image.new("RGBA", (IM_SIZE, IM_SIZE), (255,255,255,255))
        bg_r = bg.resize((IM_SIZE, IM_SIZE), Image.BICUBIC)
        canvas.alpha_composite(bg_r)

        # paste 1–2 logos
        boxes = []
        for __ in range(1 if random.random()<0.85 else 2):
            L = random_logo_transform(random.choice(logos), IM_SIZE, IM_SIZE)
            boxes.append(place_logo(canvas, L))

        # light page noise
        if random.random() < 0.6:
            canvas = canvas.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.8)))

        img = canvas.convert("RGB")
        uid = uuid.uuid4().hex[:12]
        img_path = OUT_DIR / "images" / split / f"{uid}.jpg"
        lbl_path = OUT_DIR / "labels" / split / f"{uid}.txt"

        # ✅ save the image properly
        img.save(img_path, quality=92)

        with open(lbl_path, "w") as f:
            for (cx, cy, w, h) in boxes:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def main():
    OUT_DIR.mkdir(exist_ok=True)
    make_split("train", N_TRAIN)
    make_split("val",   N_VAL)

if __name__ == "__main__":
    main()
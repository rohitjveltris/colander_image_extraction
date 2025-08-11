import os, random
from pathlib import Path
from xml.etree import ElementTree as ET
from multiprocessing import Pool, cpu_count

# ===== CONFIG =====
LOGODET_ROOT = Path("/Users/rohitjavvadi/Documents/colander_image_extraction/datasets/LogoDet-3K")  # <-- change this to where LogoDet-3K lives
OUT_DIR = Path("data_logodet_yolo")                    # output under your project
VAL_FRACTION = 0.10
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RANDOM_SEED = 1337
# ==================

random.seed(RANDOM_SEED)

def bbox_xyxy_to_yolo(x1,y1,x2,y2,w,h):
    cx = (x1+x2)/2.0 / w
    cy = (y1+y2)/2.0 / h
    bw = (x2-x1)/float(w)
    bh = (y2-y1)/float(h)
    return cx,cy,bw,bh

def parse_xml(xml_path: Path):
    # Returns (image_path, label_txt_lines) or (None, None) if fail
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        fn = root.findtext("filename")
    except Exception:
        return None, None

    # 1) prefer <filename> in same dir
    img = None
    if fn:
        cand = (xml_path.parent / Path(fn).name)
        if cand.exists():
            img = cand

    # 2) fallback: same-stem any known image ext
    if img is None:
        for ext in IMG_EXTS:
            cand = xml_path.with_suffix(ext)
            if cand.exists():
                img = cand
                break

    if img is None or not img.exists():
        return None, None

    # Single-class "logo" => class id 0
    lines = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        try:
            x1 = float(bb.find("xmin").text)
            y1 = float(bb.find("ymin").text)
            x2 = float(bb.find("xmax").text)
            y2 = float(bb.find("ymax").text)
        except Exception:
            continue
        cx, cy, bw, bh = bbox_xyxy_to_yolo(x1, y1, x2, y2, w, h)
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if not lines:
        return None, None

    return img.resolve().as_posix(), lines

def main():
    xmls = list(LOGODET_ROOT.rglob("*.xml"))
    if not xmls:
        raise SystemExit(f"No XML found under {LOGODET_ROOT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_dir = OUT_DIR / "labels_all"  # flat label store mirroring unique names
    labels_dir.mkdir(parents=True, exist_ok=True)

    # parallel parse
    n_workers = min(32, max(4, cpu_count() - 1))
    with Pool(processes=n_workers) as pool:
        results = list(pool.imap_unordered(parse_xml, xmls, chunksize=64))

    # collect valid pairs
    pairs = [(p, l) for (p, l) in results if p is not None]
    if not pairs:
        raise SystemExit("No valid (image,label) pairs parsed. Check paths/permissions.")

    # dedupe (some mirrors may have dup paths)
    seen = set()
    unique = []
    for p, l in pairs:
        if p not in seen:
            unique.append((p, l))
            seen.add(p)

    # split
    random.shuffle(unique)
    n_total = len(unique)
    n_val = max(1, int(n_total * VAL_FRACTION))
    val_set = unique[:n_val]
    train_set = unique[n_val:]

    # write label files and path lists (NO image copies)
    def save_split(split_name, split_pairs):
        list_file = OUT_DIR / f"{split_name}.txt"
        with open(list_file, "w") as lf:
            for img_path, lines in split_pairs:
                # create a deterministic label filename from absolute path
                # replace path separators to avoid collisions
                safe_stem = img_path.replace(":", "_").replace("/", "__")
                lbl_path = labels_dir / f"{safe_stem}.txt"
                if not lbl_path.exists():
                    with open(lbl_path, "w") as f:
                        f.write("\n".join(lines) + "\n")
                lf.write(img_path + "\n")

    save_split("train", train_set)
    save_split("val", val_set)

    # write YAML that points to list files
    yaml_path = OUT_DIR / "logodet.yaml"
    yaml_text = f"""# YOLO single-class dataset (no-copy lists)
names:
  0: logo
train: {OUT_DIR.resolve().as_posix()}/train.txt
val: {OUT_DIR.resolve().as_posix()}/val.txt
"""
    yaml_path.write_text(yaml_text)

    print(f"[DONE] Total images: {n_total} | Train: {len(train_set)} | Val: {len(val_set)}")
    print(f"[DONE] Lists: {OUT_DIR/'train.txt'} , {OUT_DIR/'val.txt'}")
    print(f"[DONE] Labels dir: {labels_dir}")
    print(f"[DONE] YAML: {yaml_path}")

if __name__ == "__main__":
    main()
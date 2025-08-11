#!/usr/bin/env python3
import os, random, argparse
from pathlib import Path
from xml.etree import ElementTree as ET
from multiprocessing import Pool, cpu_count

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def bbox_xyxy_to_yolo(x1,y1,x2,y2,w,h):
    cx = (x1+x2)/2.0 / w
    cy = (y1+y2)/2.0 / h
    bw = (x2-x1)/float(w)
    bh = (y2-y1)/float(h)
    return cx,cy,bw,bh

def parse_xml(xml_path: Path):
    # returns (abs_img_path, [yolo_lines]) or (None, None)
    try:
        root = ET.parse(xml_path).getroot()
        w = int(root.find("size/width").text)
        h = int(root.find("size/height").text)
        fn = root.findtext("filename")
    except Exception:
        return None, None

    # 1) image via <filename> in same dir
    img = (xml_path.parent / Path(fn).name) if fn else None
    if not (img and img.exists()):
        # 2) fallback: same stem with known ext
        img = None
        for ext in IMG_EXTS:
            cand = xml_path.with_suffix(ext)
            if cand.exists():
                img = cand; break
    if not (img and img.exists()):
        return None, None

    # single-class "logo" = 0
    lines = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None: continue
        try:
            x1 = float(bb.find("xmin").text); y1 = float(bb.find("ymin").text)
            x2 = float(bb.find("xmax").text); y2 = float(bb.find("ymax").text)
        except Exception:
            continue
        cx, cy, bw, bh = bbox_xyxy_to_yolo(x1,y1,x2,y2,w,h)
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    if not lines:
        return None, None
    return str(img.resolve()), lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="Root folder that contains LogoDet-3K categories (e.g. datasets/LogoDet-3K)")
    ap.add_argument("--out", type=Path, default=Path("data_logodet_yolo"))
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    random.seed(args.seed)

    xmls = list(args.source.rglob("*.xml"))
    if not xmls:
        raise SystemExit(f"No XML files under {args.source}")

    args.out.mkdir(parents=True, exist_ok=True)
    labels_dir = args.out / "labels_all"
    labels_dir.mkdir(parents=True, exist_ok=True)

    n_workers = min(32, max(4, cpu_count() - 1))
    with Pool(processes=n_workers) as pool:
        results = list(pool.imap_unordered(parse_xml, xmls, chunksize=64))

    pairs = [(p,l) for (p,l) in results if p]
    # dedupe by absolute image path
    seen, unique = set(), []
    for p,l in pairs:
        if p not in seen:
            seen.add(p); unique.append((p,l))

    random.shuffle(unique)
    n_total = len(unique)
    n_val = max(1, int(n_total * args.val))
    val_set, train_set = unique[:n_val], unique[n_val:]

    def save_split(name, items):
        list_file = args.out / f"{name}.txt"
        with open(list_file, "w") as lf:
            for img_path, lines in items:
                safe = img_path.replace(":", "_").replace("/", "__")
                lbl_path = labels_dir / f"{safe}.txt"
                if not lbl_path.exists():
                    with open(lbl_path, "w") as f:
                        f.write("\n".join(lines) + "\n")
                lf.write(img_path + "\n")

    save_split("train", train_set)
    save_split("val",   val_set)

    yaml_text = f"""# YOLO single-class dataset (no-copy lists)
names:
  0: logo
train: {args.out.resolve().as_posix()}/train.txt
val: {args.out.resolve().as_posix()}/val.txt
"""
    (args.out / "logodet.yaml").write_text(yaml_text)

    print(f"[DONE] Total: {n_total} | Train: {len(train_set)} | Val: {len(val_set)}")
    print(f"[DONE] YAML: {(args.out/'logodet.yaml')}")
    print(f"[DONE] Example image path: {train_set[0][0] if train_set else 'N/A'}")

if __name__ == "__main__":
    main()

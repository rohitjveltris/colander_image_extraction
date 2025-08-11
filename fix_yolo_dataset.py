#!/usr/bin/env python3
"""
Fix YOLO dataset structure for LogoDet-3K
"""

import os
import shutil
from pathlib import Path

def main():
    base_dir = Path("/root/colander_image_extraction/data_logodet_yolo")
    
    # Read train and val splits
    with open(base_dir / "train.txt", "r") as f:
        train_images = [line.strip() for line in f if line.strip()]
    
    with open(base_dir / "val.txt", "r") as f:
        val_images = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(train_images)} training images and {len(val_images)} validation images...")
    
    # Process training set
    print("Processing training set...")
    new_train_paths = []
    for i, img_path in enumerate(train_images):
        if i % 1000 == 0:
            print(f"  Train: {i}/{len(train_images)}")
            
        # Get original image path
        img_path_obj = Path(img_path)
        if not img_path_obj.exists():
            print(f"Warning: Image not found: {img_path}")
            continue
            
        # Create new filename (using index to avoid conflicts)
        new_img_name = f"train_{i:06d}.jpg"
        new_img_path = base_dir / "images" / "train" / new_img_name
        
        # Copy image
        shutil.copy2(img_path, new_img_path)
        new_train_paths.append(str(new_img_path))
        
        # Find corresponding label
        # Convert original path to mangled label filename
        mangled_name = img_path.replace("/", "__") + ".txt"
        label_path = base_dir / "labels_all" / mangled_name
        
        if label_path.exists():
            # Copy label with matching name
            new_label_name = f"train_{i:06d}.txt"
            new_label_path = base_dir / "labels" / "train" / new_label_name
            shutil.copy2(label_path, new_label_path)
        else:
            print(f"Warning: Label not found for {img_path}")
    
    # Process validation set
    print("Processing validation set...")
    new_val_paths = []
    for i, img_path in enumerate(val_images):
        if i % 1000 == 0:
            print(f"  Val: {i}/{len(val_images)}")
            
        # Get original image path
        img_path_obj = Path(img_path)
        if not img_path_obj.exists():
            print(f"Warning: Image not found: {img_path}")
            continue
            
        # Create new filename
        new_img_name = f"val_{i:06d}.jpg"
        new_img_path = base_dir / "images" / "val" / new_img_name
        
        # Copy image
        shutil.copy2(img_path, new_img_path)
        new_val_paths.append(str(new_img_path))
        
        # Find corresponding label
        mangled_name = img_path.replace("/", "__") + ".txt"
        label_path = base_dir / "labels_all" / mangled_name
        
        if label_path.exists():
            # Copy label with matching name
            new_label_name = f"val_{i:06d}.txt"
            new_label_path = base_dir / "labels" / "val" / new_label_name
            shutil.copy2(label_path, new_label_path)
        else:
            print(f"Warning: Label not found for {img_path}")
    
    # Update train.txt and val.txt
    print("Updating train.txt and val.txt...")
    with open(base_dir / "train.txt", "w") as f:
        for path in new_train_paths:
            f.write(path + "\n")
    
    with open(base_dir / "val.txt", "w") as f:
        for path in new_val_paths:
            f.write(path + "\n")
    
    print(f"Dataset reorganization complete!")
    print(f"Training images: {len(new_train_paths)}")
    print(f"Validation images: {len(new_val_paths)}")

if __name__ == "__main__":
    main()
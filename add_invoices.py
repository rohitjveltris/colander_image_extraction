#!/usr/bin/env python3
"""
Add invoice images to existing YOLO dataset (single-class logo detection)
"""

import os
from pathlib import Path

def main():
    base_dir = Path("/root/colander_image_extraction")
    invoices_dir = base_dir / "invoices_raw"
    data_dir = base_dir / "data_logodet_yolo"
    labels_all_dir = data_dir / "labels_all"
    train_file = data_dir / "train.txt"
    
    # Create labels_all directory if it doesn't exist
    labels_all_dir.mkdir(exist_ok=True)
    
    print("Adding invoice images to YOLO dataset...")
    
    # Find invoice images
    invoice_images = sorted(invoices_dir.glob("*.jpg"))
    if not invoice_images:
        print("No .jpg files found in invoices_raw/")
        return
    
    print(f"Found {len(invoice_images)} invoice images")
    
    # Process each invoice
    added_count = 0
    for img_path in invoice_images:
        print(f"Processing {img_path.name}...")
        
        # Check if corresponding label file exists
        label_path = invoices_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"  ⚠️  No label file found: {label_path.name}")
            print(f"  👉 You need to annotate this image first!")
            continue
        
        # Read label file and ensure all classes are 0
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Fix any non-zero class IDs to 0
        fixed_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:  # class x y w h
                parts[0] = '0'  # Force class to 0 (logo)
                fixed_lines.append(' '.join(parts) + '\n')
        
        if not fixed_lines:
            print(f"  ⚠️  Empty or invalid label file: {label_path.name}")
            continue
        
        # Create mangled filename for labels_all directory
        # Convert absolute path to mangled name
        abs_path = str(img_path.resolve())
        mangled_name = abs_path.replace("/", "__") + ".txt"
        target_label = labels_all_dir / mangled_name
        
        # Write fixed labels
        with open(target_label, 'w') as f:
            f.writelines(fixed_lines)
        
        print(f"  ✅ Label saved: {mangled_name}")
        added_count += 1
    
    if added_count == 0:
        print("\n❌ No images were added. Make sure you have:")
        print("   1. Invoice images (.jpg) in invoices_raw/")
        print("   2. Corresponding label files (.txt) with YOLO format annotations")
        print("   3. Run annotation tool first if needed")
        return
    
    # Append invoice image paths to train.txt
    print(f"\nAppending {added_count} invoice paths to train.txt...")
    with open(train_file, 'a') as f:
        for img_path in invoice_images:
            label_path = invoices_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                f.write(str(img_path.resolve()) + '\n')
    
    # Verification
    print("\n✅ Invoice merge complete!")
    print(f"   📁 {added_count} labels added to {labels_all_dir}")
    print(f"   📝 {added_count} image paths added to train.txt")
    
    # Quick sanity check
    print("\n🔍 Sanity check:")
    with open(train_file, 'r') as f:
        lines = f.readlines()
    
    invoice_lines = [line for line in lines if 'invoices_raw' in line]
    print(f"   📊 Total images in train.txt: {len(lines)}")
    print(f"   🧾 Invoice images in train.txt: {len(invoice_lines)}")
    
    if invoice_lines:
        print("   📋 Sample invoice entries:")
        for line in invoice_lines[:3]:
            print(f"     {line.strip()}")

if __name__ == "__main__":
    main()
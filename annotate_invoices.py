#!/usr/bin/env python3
"""
Helper script for annotating invoice logos using labelImg
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch labelImg for invoice annotation"""
    
    # Paths
    project_root = Path("/root/colander_image_extraction")
    images_dir = project_root / "logos_annotation" 
    predefined_classes_file = project_root / "logo_classes.txt"
    
    # Create predefined classes file for labelImg (single class)
    logo_classes = [
        "logo"
    ]
    
    with open(predefined_classes_file, "w") as f:
        for logo_class in logo_classes:
            f.write(f"{logo_class}\n")
    
    print("=" * 60)
    print("INVOICE LOGO ANNOTATION SETUP")
    print("=" * 60)
    print(f"Images directory: {images_dir}")
    print(f"Predefined classes: {', '.join(logo_classes)}")
    print(f"Classes file: {predefined_classes_file}")
    print()
    print("INSTRUCTIONS:")
    print("1. LabelImg will open with your invoice images")
    print("2. For each image:")
    print("   - Click 'Create RectBox' (or press 'w')")
    print("   - Draw bounding box around each logo")
    print("   - Select appropriate class from dropdown")
    print("   - Click 'Save' (or press Ctrl+S)")
    print("3. Make sure to set format to 'YOLO' (not PascalVOC)")
    print("4. Save annotations as .txt files in same directory")
    print()
    print("ANNOTATION GOAL:")
    print("- Detect ANY logo/brand mark for extraction")
    print("- Don't worry about identifying specific brands")
    print("- Include: company logos, watermarks, signatures, certification marks")
    print("- Use single class 'logo' for everything")
    print("- Focus on good bounding boxes around logo areas")
    print("=" * 60)
    
    # Check if display is available (for GUI)
    if "DISPLAY" not in os.environ:
        print("WARNING: No display available. You'll need to:")
        print("1. Set up X11 forwarding: ssh -X user@host")
        print("2. Or use a local machine with display")
        print("3. Or use an alternative annotation tool like Roboflow")
        print()
        print("Alternative: Use Roboflow (https://roboflow.com)")
        print("- Upload images to Roboflow")
        print("- Annotate online")
        print("- Export in YOLO format")
        return
    
    # Launch labelImg
    cmd = [
        "labelImg",
        str(images_dir),           # images directory 
        str(predefined_classes_file)  # predefined classes
    ]
    
    print(f"Launching: {' '.join(cmd)}")
    print("Close labelImg when done annotating all images.")
    
    try:
        subprocess.run(cmd)
    except FileNotFoundError:
        print("ERROR: labelImg not found. Install with: pip install labelImg")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
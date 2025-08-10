from pathlib import Path
from PIL import Image

def check_folder(folder):
    print(f"\nChecking: {folder}")
    for p in sorted(Path(folder).glob("*")):
        try:
            img = Image.open(p)
            exif = img.getexif()
            orientation = exif.get(274)  # 274 is the EXIF Orientation tag
            if orientation and orientation != 1:
                print(f"  {p.name} -> EXIF orientation: {orientation}  ⚠ Needs rotation")
            else:
                print(f"  {p.name} -> OK (no rotation)")
        except Exception as e:
            print(f"  {p.name} -> ERROR: {e}")

if __name__ == "__main__":
    check_folder("logos")
    check_folder("invoices_raw")
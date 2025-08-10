import streamlit as st
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2

# Path to your trained model
MODEL_PATH = Path("runs/detect/train2/weights/best.pt")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

model = load_model()

# --- Functions ---
def exif_upright(pil_img: Image.Image) -> Image.Image:
    """Correct orientation based on EXIF."""
    return ImageOps.exif_transpose(pil_img)

def crop_with_pad(image_np, xyxy, pad=8):
    """Crop logo with padding."""
    h, w = image_np.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return image_np[y1:y2, x1:x2]

# --- Streamlit UI ---
st.title("📄 Logo Extractor from Invoice")
st.write("Upload an invoice and the model will detect and extract the logo.")

uploaded_file = st.file_uploader("Upload invoice image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file)
    pil_img = exif_upright(pil_img)
    img_np = np.array(pil_img)

    # Run YOLO detection
    results = model.predict(
        source=img_np,
        conf=0.25,
        iou=0.45,
        imgsz=1024,
        verbose=False
    )

    # Draw results and crop logo
    if len(results) > 0 and len(results[0].boxes) > 0:
        st.subheader("Original Invoice:")
        st.image(pil_img, caption="Uploaded Invoice", use_column_width=True)

        for i, box in enumerate(results[0].boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            crop_np = crop_with_pad(img_np, xyxy, pad=8)
            crop_pil = Image.fromarray(crop_np)

            st.subheader(f"Extracted Logo #{i+1}")
            st.image(crop_pil, use_column_width=False)

    else:
        st.warning("No logo detected in the image.")
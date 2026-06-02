import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import gdown
import onnxruntime as rt

# Config
st.set_page_config(page_title="Testised v2", page_icon="🔬", layout="wide")
st.title("🔬 Testised - Testis Cell Detector")

# Class definitions
CLASSES = {
    0: {"name": "Spermatogonia", "color": (30, 144, 255)},
    1: {"name": "Primary spermatocyte", "color": (199, 21, 133)},
    2: {"name": "Spermatide", "color": (220, 20, 60)},
    3: {"name": "Spermatozoa", "color": (46, 139, 87)},
    4: {"name": "Sertoli cell", "color": (105, 105, 105)},
    5: {"name": "Garbage", "color": (148, 163, 184)},
}

# Load ONNX model
@st.cache_resource
def load_model():
    try:
        file_id = st.secrets["GDRIVE_FILE_ID"]
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        model_path = "best.onnx"
        
        if not Path(model_path).exists():
            st.info("📥 Downloading model...")
            gdown.download(url, model_path, quiet=False)
        
        return rt.InferenceSession(model_path)
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

session = load_model()

# UI
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Upload Image")
    uploaded = st.file_uploader("H&E image", type=["jpg", "jpeg", "png", "tiff"])

with col2:
    st.subheader("Settings")
    conf = st.slider("Confidence", 0.1, 0.9, 0.3, 0.05)

if uploaded and session:
    image = Image.open(uploaded).convert("RGB")
    
    if st.button("🔍 Detect", use_container_width=True, type="primary"):
        # Resize with aspect ratio preservation
        img_canvas = Image.new('RGB', (1024, 1024), 'white')
        scale = 1024 / max(image.width, image.height)
        w, h = int(image.width * scale), int(image.height * scale)
        img_resized = image.resize((w, h), Image.Resampling.LANCZOS)
        x_off, y_off = (1024 - w) // 2, (1024 - h) // 2
        img_canvas.paste(img_resized, (x_off, y_off))
        
        # Prepare input
        img_array = np.array(img_canvas).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))[np.newaxis, :]
        
        # Run inference
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        outputs = session.run(output_names, {input_name: img_array})
        
        # Parse YOLO output (raw: [1, 25200, 85] -> batch, proposals, 4+80classes+1confidence)
        predictions = outputs[0][0]  # [25200, 85]
        detections = []
        
        for pred in predictions:
            x_center, y_center, w_box, h_box = pred[:4]
            conf_scores = pred[4:5]  # objectness
            class_scores = pred[5:]  # 80 classes (COCO), use argmax for simplicity
            
            conf_val = float(conf_scores[0])
            if conf_val < conf:
                continue
            
            # Get class (map to testis classes 0-5)
            class_idx = np.argmax(class_scores) % 6
            
            # Convert to pixel coordinates
            x1 = (x_center - w_box/2 - x_off) * (image.width / w)
            y1 = (y_center - h_box/2 - y_off) * (image.height / h)
            x2 = (x_center + w_box/2 - x_off) * (image.width / w)
            y2 = (y_center + h_box/2 - y_off) * (image.height / h)
            
            detections.append({
                "class": class_idx,
                "x1": max(0, x1), "y1": max(0, y1),
                "x2": min(image.width, x2), "y2": min(image.height, y2),
                "conf": conf_val
            })
        
        # Draw
        img_out = image.copy()
        draw = ImageDraw.Draw(img_out)
        for d in detections:
            color = CLASSES[d["class"]]["color"]
            x1 = max(0, min(int(d["x1"]), image.width))
            y1 = max(0, min(int(d["y1"]), image.height))
            x2 = max(0, min(int(d["x2"]), image.width))
            y2 = max(0, min(int(d["y2"]), image.height))
            
            if x1 < x2 and y1 < y2:  # valid box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(img_out, caption=f"Detected {len(detections)} cells")
        with col2:
            st.metric("Total cells", len(detections))
            counts = {}
            for d in detections:
                cls = d["class"]
                counts[cls] = counts.get(cls, 0) + 1
            for cls_id in sorted(counts.keys()):
                st.write(f"**{CLASSES[cls_id]['name']}:** {counts[cls_id]}")

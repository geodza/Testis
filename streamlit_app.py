import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import gdown
import onnxruntime as rt

st.set_page_config(page_title="Testised v2", page_icon="🔬", layout="wide")
st.title("🔬 Testised - Testis Cell Detector")

CLASSES = {
    0: {"name": "Spermatogonia", "color": (30, 144, 255)},
    1: {"name": "Primary spermatocyte", "color": (199, 21, 133)},
    2: {"name": "Spermatide", "color": (220, 20, 60)},
    3: {"name": "Spermatozoa", "color": (46, 139, 87)},
    4: {"name": "Sertoli cell", "color": (105, 105, 105)},
    5: {"name": "Garbage", "color": (148, 163, 184)},
}

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
        
        # Parse YOLO output
        predictions = outputs[0][0]  # [25200, 85]
        detections = []
        
        for pred in predictions:
            conf_val = float(pred[4])
            if conf_val < conf:
                continue
            
            # Get class
            class_scores = pred[5:]
            class_idx = int(np.argmax(class_scores)) % 6
            
            # YOLO format: x_center, y_center, width, height (normalized 0-1)
            x_center, y_center, box_w, box_h = pred[:4]
            
            # Convert to pixel coords in 1024 image
            x1_img = (x_center - box_w/2) * 1024
            y1_img = (y_center - box_h/2) * 1024
            x2_img = (x_center + box_w/2) * 1024
            y2_img = (y_center + box_h/2) * 1024
            
            # Remove offset and scale to original image
            if w > 0 and h > 0:
                x1 = max(0, (x1_img - x_off) * (image.width / w))
                y1 = max(0, (y1_img - y_off) * (image.height / h))
                x2 = min(image.width, (x2_img - x_off) * (image.width / w))
                y2 = min(image.height, (y2_img - y_off) * (image.height / h))
                
                # Ensure x1 < x2, y1 < y2
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                if x2 - x1 > 5 and y2 - y1 > 5:  # minimum box size
                    detections.append({
                        "class": class_idx,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": conf_val
                    })
        
        # Draw
        img_out = image.copy()
        draw = ImageDraw.Draw(img_out)
        for d in detections:
            color = CLASSES[d["class"]]["color"]
            x1, y1 = int(d["x1"]), int(d["y1"])
            x2, y2 = int(d["x2"]), int(d["y2"])
            
            if x1 < x2 and y1 < y2:
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
                if cls_id in counts:
                    st.write(f"**{CLASSES[cls_id]['name']}:** {counts[cls_id]}")

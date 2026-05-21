import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io
import json
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import gdown

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Testised v1.0",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Testised v1.0")
st.markdown("**Testis cell detector** — Upload H&E images and detect cells")

# ============ CLASS INFO ============
CLASS_INFO = {
    0: {"name": "Spermatogonia", "color": (30, 144, 255)},
    1: {"name": "Primary spermatocyte", "color": (199, 21, 133)},
    2: {"name": "Spermatide", "color": (220, 20, 60)},
    3: {"name": "Spermatozoa", "color": (46, 139, 87)},
    4: {"name": "Sertoli cell", "color": (105, 105, 105)},
    5: {"name": "Garbage", "color": (148, 163, 184)},
}

# ============ SESSION STATE ============
if "detections" not in st.session_state:
    st.session_state.detections = []
if "image" not in st.session_state:
    st.session_state.image = None
if "selected_classes" not in st.session_state:
    st.session_state.selected_classes = {0, 1, 2, 3, 4}

# ============ LOAD MODEL ============
@st.cache_resource
def load_model():
    try:
        if Path("best.pt").exists():
            return YOLO("best.pt")
        
        st.info("📥 Downloading model from Google Drive (first load)...")
        
        try:
            file_id = st.secrets["GDRIVE_FILE_ID"]
        except:
            st.error("""
            ❌ Google Drive file ID not configured.
            
            **To fix:**
            1. Upload best.pt to Google Drive
            2. Share → Copy link → Extract file ID (between /d/ and /)
            3. Go to Streamlit Cloud → App settings → Secrets
            4. Add: `GDRIVE_FILE_ID = "YOUR_FILE_ID"`
            """)
            return None
        
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        gdown.download(url, "best.pt", quiet=False)
        return YOLO("best.pt")
        
    except Exception as e:
        st.error(f"❌ Model error: {e}")
        return None

model = load_model()

# ============ HELPER FUNCTIONS ============
def assess_quality(image):
    """Assess tissue quality"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        gray = np.mean(img_array[:,:,:3], axis=2)
    else:
        gray = img_array
    
    edges = np.sum(np.abs(np.diff(gray))) / gray.size
    contrast = np.std(gray) / 128
    score = (edges + contrast) / 2
    
    if score > 0.6:
        return "Excellent", "#d4edda"
    elif score > 0.4:
        return "Good", "#cce5ff"
    elif score > 0.2:
        return "Fair", "#fff3cd"
    else:
        return "Poor", "#f8d7da"

def detect_cells(image, confidence):
    """Run detection with proper aspect ratio"""
    if model is None:
        return []
    
    # Preserve aspect ratio with white padding
    img_canvas = Image.new('RGB', (1024, 1024), 'white')
    scale = 1024 / max(image.width, image.height)
    new_w = int(image.width * scale)
    new_h = int(image.height * scale)
    img_resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x_offset = (1024 - new_w) // 2
    y_offset = (1024 - new_h) // 2
    img_canvas.paste(img_resized, (x_offset, y_offset))
    
    # Run inference
    results = model.predict(img_canvas, conf=confidence, iou=0.5, verbose=False)
    
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            # Scale back to original image
            scale_x = image.width / new_w
            scale_y = image.height / new_h
            x1 = (x1 - x_offset) * scale_x
            y1 = (y1 - y_offset) * scale_y
            x2 = (x2 - x_offset) * scale_x
            y2 = (y2 - y_offset) * scale_y
            
            if cls in st.session_state.selected_classes:
                detections.append({
                    "class_id": cls,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": conf,
                })
    
    return detections

def draw_boxes(image, detections):
    """Draw detection boxes on image"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for det in detections:
        cls_id = det["class_id"]
        if cls_id not in st.session_state.selected_classes:
            continue
        
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        color = CLASS_INFO[cls_id]["color"]
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label
        label = CLASS_INFO[cls_id]["name"].split()[0]
        text_bbox = draw.textbbox((x1, y1), label)
        draw.rectangle([x1, y1 - 16, text_bbox[2] + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - 14), label, fill="white")
    
    return img_copy

# ============ UI ============
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📁 Upload Image")
    uploaded_file = st.file_uploader("Choose H&E image", type=["jpg", "jpeg", "png", "tiff"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.image = image
        
        quality, color = assess_quality(image)
        st.markdown(f"**Quality:** <span style='background: {color}; padding: 4px 8px; border-radius: 4px; font-weight: 500;'>{quality}</span>", unsafe_allow_html=True)
        st.caption(f"Size: {image.width}×{image.height} px")

with col2:
    st.subheader("⚙️ Settings")
    
    confidence = st.slider("Confidence", 0.1, 0.9, 0.3, 0.05)
    st.caption(f"Current: {confidence:.2f}")
    
    st.markdown("**Cell types:**")
    for cls_id, info in CLASS_INFO.items():
        default = cls_id in st.session_state.selected_classes
        checked = st.checkbox(info["name"], value=default, key=f"cls_{cls_id}")
        
        if checked:
            st.session_state.selected_classes.add(cls_id)
        else:
            st.session_state.selected_classes.discard(cls_id)

# ============ DETECTION ============
if st.session_state.image:
    if st.button("🔍 Detect cells", use_container_width=True, type="primary"):
        with st.spinner("Running detection..."):
            st.session_state.detections = detect_cells(st.session_state.image, confidence)
    
    if st.session_state.detections:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Results")
            img_result = draw_boxes(st.session_state.image, st.session_state.detections)
            st.image(img_result, use_column_width=True)
        
        with col2:
            st.subheader("📊 Stats")
            
            # Count cells
            counts = {i: 0 for i in range(6)}
            for det in st.session_state.detections:
                if det["class_id"] in st.session_state.selected_classes:
                    counts[det["class_id"]] += 1
            
            total = sum(counts.values())
            st.metric("Total", total)
            
            st.write("")
            for cls_id in range(6):
                if cls_id in st.session_state.selected_classes:
                    color_hex = "#{:02x}{:02x}{:02x}".format(*CLASS_INFO[cls_id]["color"])
                    st.write(f"<span style='color: {color_hex};'>●</span> {CLASS_INFO[cls_id]['name']}: **{counts[cls_id]}**", 
                            unsafe_allow_html=True)
        
        # Export buttons
        st.write("")
        col1, col2 = st.columns(2)
        
        with col1:
            df = pd.DataFrame([
                {
                    "Class": CLASS_INFO[d["class_id"]]["name"],
                    "X1": int(d["x1"]),
                    "Y1": int(d["y1"]),
                    "X2": int(d["x2"]),
                    "Y2": int(d["y2"]),
                    "Confidence": f"{d['conf']:.3f}",
                }
                for d in st.session_state.detections
                if d["class_id"] in st.session_state.selected_classes
            ])
            
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, sheet_name="Detections", index=False)
            excel_buffer.seek(0)
            
            st.download_button(
                "📊 Excel",
                excel_buffer,
                "detections.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col2:
            features = []
            for d in st.session_state.detections:
                if d["class_id"] in st.session_state.selected_classes:
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [float(d["x1"]), float(d["y1"])],
                                [float(d["x1"]), float(d["y2"])],
                                [float(d["x2"]), float(d["y2"])],
                                [float(d["x2"]), float(d["y1"])],
                                [float(d["x1"]), float(d["y1"])]
                            ]]
                        },
                        "properties": {
                            "classification": {"name": CLASS_INFO[d["class_id"]]["name"]},
                            "confidence": float(d["conf"]),
                        }
                    })
            
            geojson_str = json.dumps({"type": "FeatureCollection", "features": features}, indent=2)
            
            st.download_button(
                "📍 GeoJSON",
                geojson_str,
                "detections.geojson",
                "application/json",
                use_container_width=True
            )

else:
    st.info("👈 Upload an image to get started")

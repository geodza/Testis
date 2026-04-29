import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io
import json
from ultralytics import YOLO
import pandas as pd
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Testised v1.0",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    # 🔬 Testised v1.0
    **Advanced testis cell detector** — Detect and quantify cells from H&E histology
    """)

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

# ============ LOAD MODEL FROM GOOGLE DRIVE ============
@st.cache_resource
def load_model():
    try:
        # Check if model exists locally
        if Path("best.pt").exists():
            model = YOLO("best.pt")
            return model
        
        # Download from Google Drive
        st.info("📥 First load: downloading model from Google Drive...")
        
        try:
            file_id = st.secrets["GDRIVE_FILE_ID"]
        except:
            st.error("""
            ❌ Model file ID not configured.
            
            **To deploy on Streamlit Cloud:**
            1. Upload best.pt to Google Drive
            2. Right-click → Share → Copy link
            3. Extract file ID from link (between /d/ and /)
            4. Go to Streamlit Cloud app settings → Secrets
            5. Add: `GDRIVE_FILE_ID = "YOUR_FILE_ID"`
            """)
            return None
        
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        gdown.download(url, "best.pt", quiet=False)
        
        model = YOLO("best.pt")
        return model
        
    except Exception as e:
        st.error(f"❌ Model error: {e}")
        return None

model = load_model()

# ============ ASSESS TISSUE QUALITY ============
def assess_quality(image):
    """Simple quality assessment"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        gray = np.mean(img_array[:,:,:3], axis=2)
    else:
        gray = img_array
    
    edges = np.sum(np.abs(np.diff(gray))) / gray.size
    contrast = np.std(gray) / 128
    score = (edges + contrast) / 2
    
    if score > 0.6:
        return "Excellent ✓", "#d4edda"
    elif score > 0.4:
        return "Good ✓", "#cce5ff"
    elif score > 0.2:
        return "Fair", "#fff3cd"
    else:
        return "Poor", "#f8d7da"

# ============ DETECT CELLS ============
def detect_cells(image, magnification, confidence):
    """Run YOLOv8 detection with proper aspect ratio preservation"""
    if model is None:
        st.error("Model not loaded")
        return []
    
    # Preserve aspect ratio with white padding
    img_copy = Image.new('RGB', (1024, 1024), 'white')
    scale = 1024 / max(image.width, image.height)
    new_w = int(image.width * scale)
    new_h = int(image.height * scale)
    img_resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x_offset = (1024 - new_w) // 2
    y_offset = (1024 - new_h) // 2
    img_copy.paste(img_resized, (x_offset, y_offset))
    
    results = model.predict(img_copy, conf=confidence, iou=0.5, verbose=False)
    
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            # Scale coordinates back to original image size
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

# ============ DRAW BOXES ============
def draw_detections(image, detections):
    """Draw bounding boxes"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for det in detections:
        cls_id = det["class_id"]
        if cls_id not in st.session_state.selected_classes:
            continue
        
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        color = CLASS_INFO[cls_id]["color"]
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = CLASS_INFO[cls_id]["name"].split()[0]
        text_bbox = draw.textbbox((x1, y1), label)
        draw.rectangle([x1, y1 - 16, text_bbox[2] + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - 14), label, fill="white")
    
    return img_copy

# ============ MAIN UI ============
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📁 Upload Image")
    uploaded_file = st.file_uploader("Choose an H&E image", type=["jpg", "jpeg", "png", "tiff"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.image = image
        
        quality, color = assess_quality(image)
        st.markdown(f"**Tissue Quality:** <span style='background: {color}; padding: 4px 8px; border-radius: 4px;'>{quality}</span>", unsafe_allow_html=True)
        st.write(f"Image size: {image.width}×{image.height} px")

with col2:
    st.subheader("⚙️ Settings")
    
    magnification = st.radio("Input magnification", [20, 40], format_func=lambda x: f"{x}x")
    
    st.markdown("**Confidence threshold**")
    confidence = st.slider("Detection sensitivity", 0.1, 0.9, 0.3, 0.05, 
                          help="Lower = more boxes (loose), Higher = fewer boxes (strict)")
    st.caption(f"Current: {confidence:.2f}")
    
    st.markdown("**Cell types:**")
    for cls_id, info in CLASS_INFO.items():
        default = cls_id in st.session_state.selected_classes
        checked = st.checkbox(info["name"], value=default, key=f"class_{cls_id}")
        
        if checked:
            st.session_state.selected_classes.add(cls_id)
        else:
            st.session_state.selected_classes.discard(cls_id)

# ============ DETECTION ============
if st.session_state.image:
    if st.button("🔍 Detect cells", use_container_width=True, type="primary"):
        with st.spinner("Running detection..."):
            st.session_state.detections = detect_cells(st.session_state.image, magnification, confidence)
    
    if st.session_state.detections:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Detection Results")
            img_with_boxes = draw_detections(st.session_state.image, st.session_state.detections)
            st.image(img_with_boxes, use_column_width=True)
        
        with col2:
            st.subheader("📊 Statistics")
            
            counts = {cls_id: 0 for cls_id in CLASS_INFO}
            for det in st.session_state.detections:
                if det["class_id"] in st.session_state.selected_classes:
                    counts[det["class_id"]] += 1
            
            total = sum(counts.values())
            st.metric("Total cells", total)
            st.write("")
            
            for cls_id, count in counts.items():
                if cls_id in st.session_state.selected_classes:
                    color_hex = "#{:02x}{:02x}{:02x}".format(*CLASS_INFO[cls_id]["color"])
                    st.write(f"<span style='color: {color_hex};'>●</span> **{CLASS_INFO[cls_id]['name']}:** {count}", unsafe_allow_html=True)
        
        st.write("")
        col1, col2 = st.columns(2)
        
        with col1:
            df = pd.DataFrame([
                {
                    "Class": CLASS_INFO[det["class_id"]]["name"],
                    "X1": int(det["x1"]),
                    "Y1": int(det["y1"]),
                    "X2": int(det["x2"]),
                    "Y2": int(det["y2"]),
                    "Confidence": f"{det['conf']:.3f}",
                }
                for det in st.session_state.detections
                if det["class_id"] in st.session_state.selected_classes
            ])
            
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, sheet_name="Detections", index=False)
            excel_buffer.seek(0)
            
            st.download_button(
                label="📊 Download Excel",
                data=excel_buffer,
                file_name="testised_detections.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            features = []
            for det in st.session_state.detections:
                if det["class_id"] in st.session_state.selected_classes:
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [float(det["x1"]), float(det["y1"])],
                                [float(det["x1"]), float(det["y2"])],
                                [float(det["x2"]), float(det["y2"])],
                                [float(det["x2"]), float(det["y1"])],
                                [float(det["x1"]), float(det["y1"])]
                            ]]
                        },
                        "properties": {
                            "classification": {"name": CLASS_INFO[det["class_id"]]["name"]},
                            "confidence": float(det["conf"]),
                        }
                    })
            
            geojson = {"type": "FeatureCollection", "features": features}
            geojson_str = json.dumps(geojson, indent=2)
            
            st.download_button(
                label="📍 Download GeoJSON",
                data=geojson_str,
                file_name="testised_detections.geojson",
                mime="application/json"
            )
    
    elif st.session_state.image:
        st.info("👆 Click 'Detect cells' to start")

else:
    st.info("👈 Upload an H&E image")

import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path
from PIL import Image

import sys
# Add parent dir to path so we can import src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config
from src.data.preprocessing import SonarPreprocessor
from src.models.model import ModelWrapper
from src.models.inference import TiledInferenceEngine

st.title("Sonar Detection")

# Setup config
try:
    config = load_config()
except Exception as e:
    st.error(f"Error loading config: {e}")
    st.stop()

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Sonar Image", type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'])

# Preprocessing Toggles
st.sidebar.subheader("Preprocessing")
use_median = st.sidebar.checkbox("Median Filter", value=config.preprocessing.median_filter)
use_bilateral = st.sidebar.checkbox("Bilateral Filter", value=config.preprocessing.bilateral_filter)
use_clahe = st.sidebar.checkbox("CLAHE", value=config.preprocessing.clahe)
use_norm = st.sidebar.checkbox("Normalize", value=config.preprocessing.normalize)

# Update config dynamically for demo
config.preprocessing.median_filter = use_median
config.preprocessing.bilateral_filter = use_bilateral
config.preprocessing.clahe = use_clahe
config.preprocessing.normalize = use_norm

conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, config.postprocessing.confidence_thresholds.default)

if uploaded_file is not None:
    # Save temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
        tf.write(uploaded_file.getbuffer())
        temp_path = tf.name
        
    img = cv2.imread(temp_path)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    preprocessor = SonarPreprocessor(config)
    prep_img = preprocessor.preprocess(img)
    
    with col2:
        st.subheader("Preprocessed")
        st.image(cv2.cvtColor(prep_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    if st.button("Run Detection"):
        with st.spinner("Running inference..."):
            try:
                model = ModelWrapper(config)
                inference_engine = TiledInferenceEngine(model, config)
                preds, inf_time = inference_engine.run_inference(prep_img)
                
                # Filter by confidence
                preds = [p for p in preds if p.confidence >= conf_thresh]
                
                # Draw boxes
                disp_img = prep_img.copy()
                class_names = model.get_class_names()
                
                for p in preds:
                    x1, y1, x2, y2 = [int(v) for v in p.bbox]
                    cv2.rectangle(disp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cls_name = class_names.get(p.class_id, str(p.class_id))
                    cv2.putText(disp_img, f"{cls_name} {p.confidence:.2f}", (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                st.subheader("Detections")
                st.image(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.success(f"Found {len(preds)} objects in {inf_time:.1f} ms.")
                
            except Exception as e:
                st.error(f"Inference failed: {e}")
                
    os.unlink(temp_path)
else:
    st.info("Upload an image on the sidebar to begin.")

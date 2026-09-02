import streamlit as st
import json
from pathlib import Path
from PIL import Image

st.title("Dataset Explorer")

stats_file = Path("outputs/dataset_stats.json")
if stats_file.exists():
    with open(stats_file, 'r') as f:
        stats = json.load(f)
        
    st.write("### Statistics")
    st.json(stats)
    
    sample_grid = Path("outputs/visualizations/sample_grid.png")
    if sample_grid.exists():
        st.write("### Sample Images")
        img = Image.open(sample_grid)
        st.image(img, use_container_width=True)
else:
    st.warning("Dataset statistics not found. Please run inspect_dataset.py.")

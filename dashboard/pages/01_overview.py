import streamlit as st
import json
import pandas as pd
from pathlib import Path
import plotly.express as px

st.title("Overview")

stats_file = Path("outputs/dataset_stats.json")
if stats_file.exists():
    with open(stats_file, 'r') as f:
        stats = json.load(f)
        
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Images Processed", stats.get("total_images", 0))
    
    # Try to load detection stats
    report_file = Path("outputs/reports/detections.csv")
    if report_file.exists():
        df = pd.read_csv(report_file)
        col2.metric("Total Detections", len(df))
        col3.metric("Average Confidence", f"{df['confidence'].mean():.2f}" if not df.empty else "0.00")
        
        # Simple bar chart
        if not df.empty:
            class_counts = df['class_name'].value_counts().reset_index()
            class_counts.columns = ['class', 'count']
            fig = px.bar(class_counts, x='class', y='count', title="Detection Distribution", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = px.histogram(df, x="confidence", title="Confidence Distribution", template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Dataset statistics not found. Please run the dataset inspection step.")
    
st.write("### Model Status")
weights_path = Path("models/yolov8s-seg.pt") # fallback path
best_path = Path("outputs/training_run/weights/best.pt")

if best_path.exists():
    st.success(f"Trained model found: {best_path.name}")
else:
    st.error("No trained model found. Please run training or place weights in the models directory.")

import streamlit as st
import json
from pathlib import Path

st.title("Model Performance")

metrics_file = Path("outputs/evaluation/metrics.json")

if metrics_file.exists():
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
        
    st.write("### Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("mAP@50", f"{metrics.get('mAP50', 0.0):.4f}")
    col2.metric("mAP@50-95", f"{metrics.get('mAP50-95', 0.0):.4f}")
    col3.metric("Precision", f"{metrics.get('precision', 0.0):.4f}")
    col4.metric("Recall", f"{metrics.get('recall', 0.0):.4f}")
    
    st.json(metrics)
    
else:
    st.info("Model performance metrics unavailable. Run: `python training/evaluate.py`")
    
export_file = Path("outputs/metrics/export_benchmark.json")
if export_file.exists():
    st.write("### Export Benchmark")
    with open(export_file, 'r') as f:
        bench = json.load(f)
    st.json(bench)

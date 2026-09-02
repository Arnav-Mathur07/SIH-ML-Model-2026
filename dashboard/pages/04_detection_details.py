import streamlit as st
import pandas as pd
from pathlib import Path

st.title("Detection Details")

report_file = Path("outputs/reports/detections.csv")
if report_file.exists():
    df = pd.read_csv(report_file)
    if not df.empty:
        st.write("### All Detections")
        st.dataframe(df)
        
        st.write("### Analyze specific detection")
        det_id = st.selectbox("Select Detection ID", df["detection_id"].tolist())
        
        det_row = df[df["detection_id"] == det_id].iloc[0]
        st.write(f"**Class**: {det_row['class_name']}")
        st.write(f"**Confidence**: {det_row['confidence']:.2f}")
        st.write(f"**Coordinate Type**: {det_row['coordinate_type']}")
        
        if "latitude" in det_row and pd.notna(det_row["latitude"]):
            st.write(f"**Location**: {det_row['latitude']}, {det_row['longitude']}")
            
        if "width_m" in det_row and pd.notna(det_row["width_m"]):
            st.write(f"**Estimated Size**: {det_row['width_m']:.2f}m x {det_row['height_m']:.2f}m")
    else:
        st.info("No detections found in report.")
else:
    st.warning("No detection reports found. Run inference on some images first.")

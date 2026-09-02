import streamlit as st
import pandas as pd
from pathlib import Path

st.title("Reports")

report_dir = Path("outputs/reports")
csv_file = report_dir / "detections.csv"
json_file = report_dir / "detections.json"

if csv_file.exists():
    df = pd.read_csv(csv_file)
    st.write("### Detection Data")
    st.dataframe(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with open(csv_file, "rb") as f:
            st.download_button(
                label="Download CSV",
                data=f,
                file_name="detections.csv",
                mime="text/csv",
            )
            
    with col2:
        if json_file.exists():
            with open(json_file, "rb") as f:
                st.download_button(
                    label="Download JSON",
                    data=f,
                    file_name="detections.json",
                    mime="application/json",
                )
else:
    st.warning("No reports available. Run inference first.")

import streamlit as st
import pandas as pd
from pathlib import Path
import folium
from streamlit_folium import st_folium

st.title("Geospatial Map")

report_file = Path("outputs/reports/detections.csv")
if report_file.exists():
    df = pd.read_csv(report_file)
    if not df.empty:
        if "latitude" in df.columns and "longitude" in df.columns:
            # Filter rows that actually have GPS
            geo_df = df[df["latitude"].notna() & df["longitude"].notna()]
            
            if not geo_df.empty:
                # Center map
                center_lat = geo_df["latitude"].mean()
                center_lon = geo_df["longitude"].mean()
                
                m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
                
                for _, row in geo_df.iterrows():
                    color = "red" if row["coordinate_type"] == "GPS_ESTIMATED" else "green"
                    
                    folium.Marker(
                        location=[row["latitude"], row["longitude"]],
                        popup=f"Class: {row['class_name']}<br>Conf: {row['confidence']:.2f}<br>Type: {row['coordinate_type']}",
                        icon=folium.Icon(color=color, icon="info-sign"),
                    ).add_to(m)
                    
                    if row["coordinate_type"] == "GPS_ESTIMATED":
                        folium.Circle(
                            location=[row["latitude"], row["longitude"]],
                            radius=15, # 15 meters uncertainty
                            color="red",
                            fill=True,
                            opacity=0.3
                        ).add_to(m)
                        
                st_data = st_folium(m, width=800, height=600)
            else:
                st.info("Geospatial visualization requires sonar positioning metadata (latitude, longitude, heading, range resolution). Place metadata CSV in data/sonar/metadata/ and re-run inference.")
        else:
            st.info("Geospatial visualization requires sonar positioning metadata (latitude, longitude, heading, range resolution). Place metadata CSV in data/sonar/metadata/ and re-run inference.")
    else:
        st.info("No detections found in report.")
else:
    st.warning("No detection reports found. Run inference on some images first.")

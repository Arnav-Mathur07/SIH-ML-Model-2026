import pandas as pd
import logging
from dataclasses import dataclass
from typing import Optional, Dict
from pathlib import Path

@dataclass
class SonarMetadata:
    filename: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    heading: Optional[float] = None
    depth: Optional[float] = None
    range_m: Optional[float] = None
    across_track_resolution: Optional[float] = None
    along_track_resolution: Optional[float] = None
    timestamp: Optional[str] = None
    ping_id: Optional[str] = None
    sensor_info: Optional[str] = None

class MetadataParser:
    def __init__(self, metadata_dir: Path):
        self.metadata_dir = metadata_dir
        self.metadata_cache: Dict[str, SonarMetadata] = {}
        self._load_all()

    def _load_all(self):
        if not self.metadata_dir.exists():
            return
            
        for csv_file in self.metadata_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                # Standardize column names
                cols = {c.lower().strip(): c for c in df.columns}
                
                # Check for filename column, fallback to index/stem if not found
                filename_col = cols.get("filename", cols.get("image_filename"))
                
                for _, row in df.iterrows():
                    filename = str(row[filename_col]) if filename_col else f"{csv_file.stem}.png"
                    
                    self.metadata_cache[filename] = SonarMetadata(
                        filename=filename,
                        latitude=row.get(cols.get("latitude")),
                        longitude=row.get(cols.get("longitude")),
                        heading=row.get(cols.get("heading")),
                        depth=row.get(cols.get("depth")),
                        range_m=row.get(cols.get("range_m", cols.get("range"))),
                        across_track_resolution=row.get(cols.get("across_track_resolution")),
                        along_track_resolution=row.get(cols.get("along_track_resolution")),
                        timestamp=str(row.get(cols.get("timestamp"))) if cols.get("timestamp") else None,
                        ping_id=str(row.get(cols.get("ping_id"))) if cols.get("ping_id") else None,
                    )
            except Exception as e:
                logging.error(f"Failed to parse metadata CSV {csv_file}: {e}")

    def get_metadata(self, filename: str) -> Optional[SonarMetadata]:
        return self.metadata_cache.get(filename)

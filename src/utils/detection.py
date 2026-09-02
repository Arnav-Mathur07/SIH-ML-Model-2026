from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import numpy as np

class CoordinateType(Enum):
    GPS_ACCURATE = "GPS_ACCURATE"
    GPS_ESTIMATED = "GPS_ESTIMATED"
    IMAGE_SPACE = "IMAGE_SPACE"

@dataclass
class Detection:
    detection_id: str
    image_path: str
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]           # [xmin, ymin, xmax, ymax] absolute pixel coords
    mask: Optional[np.ndarray]  # binary mask if segmentation model
    coordinate_type: CoordinateType
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timestamp: Optional[str] = None
    ping_id: Optional[str] = None
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    source_tile: Optional[str] = None
    inference_time_ms: float = 0.0

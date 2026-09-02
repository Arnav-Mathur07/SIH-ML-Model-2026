import logging
import uuid
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Minimal definitions for TiledInferenceEngine requirements
class RawPrediction:
    def __init__(self, bbox: List[float], conf: float, cls_id: int, mask: Optional[np.ndarray] = None):
        self.bbox = bbox # [xmin, ymin, xmax, ymax] in absolute coords
        self.confidence = conf
        self.class_id = cls_id
        self.mask = mask

# Just to ensure utils module is complete, not creating a massive utils file.

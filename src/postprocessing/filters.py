import logging
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod

from src.utils.detection import Detection
from src.utils.config import Config

class FilterLog:
    def __init__(self):
        self.rejected_count = 0
        self.reasons: Dict[str, int] = {}
        
    def add_rejection(self, reason: str):
        self.rejected_count += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

class BaseFilter(ABC):
    @abstractmethod
    def filter(self, detections: List[Detection], filter_log: FilterLog) -> List[Detection]:
        pass

class ConfidenceFilter(BaseFilter):
    def __init__(self, threshold: float):
        self.threshold = threshold
        
    def filter(self, detections: List[Detection], filter_log: FilterLog) -> List[Detection]:
        keep = []
        for det in detections:
            if det.confidence >= self.threshold:
                keep.append(det)
            else:
                filter_log.add_rejection("ConfidenceFilter")
        return keep

class GeometricFilter(BaseFilter):
    def __init__(self, min_area: int, max_area_fraction: float, image_shape: Tuple[int, int]):
        self.min_area = min_area
        self.max_area_fraction = max_area_fraction
        self.image_area = image_shape[0] * image_shape[1]
        
    def filter(self, detections: List[Detection], filter_log: FilterLog) -> List[Detection]:
        keep = []
        for det in detections:
            w = det.bbox[2] - det.bbox[0]
            h = det.bbox[3] - det.bbox[1]
            area = w * h
            
            if area < self.min_area:
                filter_log.add_rejection("GeometricFilter_MinArea")
            elif (area / self.image_area) > self.max_area_fraction:
                filter_log.add_rejection("GeometricFilter_MaxArea")
            else:
                keep.append(det)
        return keep

class ShadowConsistencyFilter(BaseFilter):
    def __init__(self, nadir_position: str):
        self.nadir_position = nadir_position
        
    def filter(self, detections: List[Detection], filter_log: FilterLog) -> List[Detection]:
        # For MVP, just a placeholder structure
        # A real shadow filter would analyze the pixels right next to the bbox based on nadir
        # Here we just pass through and log
        keep = []
        for det in detections:
            # Fake logic for testing/MVP: assume all pass
            keep.append(det)
        return keep

class SpatialDensityFilter(BaseFilter):
    def filter(self, detections: List[Detection], filter_log: FilterLog) -> List[Detection]:
        # Just flags high-density clusters. Doesn't reject them.
        # Could log a warning.
        return detections

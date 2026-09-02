from typing import List, Tuple
from src.utils.detection import Detection
from src.utils.config import Config
from src.postprocessing.filters import (
    FilterLog, ConfidenceFilter, GeometricFilter, 
    ShadowConsistencyFilter, SpatialDensityFilter
)
from src.postprocessing.nms import non_max_suppression

class PostProcessingPipeline:
    def __init__(self, config: Config, image_shape: Tuple[int, int]):
        self.config = config.postprocessing
        self.image_shape = image_shape
        
        self.filters = [
            ConfidenceFilter(self.config.confidence_thresholds.default),
            GeometricFilter(
                min_area=self.config.min_area_px,
                max_area_fraction=self.config.max_area_fraction,
                image_shape=image_shape
            )
        ]
        
        if self.config.shadow_consistency_check:
            self.filters.append(ShadowConsistencyFilter(self.config.nadir_position))
            
        self.filters.append(SpatialDensityFilter())

    def process(self, detections: List[Detection]) -> Tuple[List[Detection], FilterLog]:
        filter_log = FilterLog()
        
        # 1. Apply NMS (important if tiling was used)
        current_dets = non_max_suppression(detections, iou_threshold=self.config.iou_threshold)
        
        # 2. Run through composable filters
        for f in self.filters:
            current_dets = f.filter(current_dets, filter_log)
            
        return current_dets, filter_log

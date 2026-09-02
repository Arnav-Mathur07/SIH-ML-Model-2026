import time
import logging
from typing import List, Dict, Tuple
import numpy as np

from src.utils.inference_utils import RawPrediction
from src.data.preprocessing import TilingEngine
from src.models.model import ModelWrapper
from src.utils.config import Config

class TiledInferenceEngine:
    def __init__(self, model: ModelWrapper, config: Config):
        self.model = model
        self.config = config
        self.tiling_enabled = config.preprocessing.tiling.enabled
        if self.tiling_enabled:
            self.tiling_engine = TilingEngine(
                tile_size=config.preprocessing.tiling.tile_size,
                overlap=config.preprocessing.tiling.overlap
            )

    def run_inference(self, image: np.ndarray) -> Tuple[List[RawPrediction], float]:
        start_time = time.time()
        
        if self.tiling_enabled:
            predictions = self._run_tiled(image)
        else:
            predictions = self.model.predict(image, conf_threshold=0.1) # low conf, filtered later
            
        inference_time_ms = (time.time() - start_time) * 1000
        return predictions, inference_time_ms

    def _run_tiled(self, image: np.ndarray) -> List[RawPrediction]:
        tiles = self.tiling_engine.split(image)
        all_predictions = []
        
        for tile_meta in tiles:
            tile_img = tile_meta["image"]
            x_offset = tile_meta["x_offset"]
            y_offset = tile_meta["y_offset"]
            
            tile_preds = self.model.predict(tile_img, conf_threshold=0.1)
            
            # Map coordinates back to full image
            for p in tile_preds:
                p.bbox[0] += x_offset
                p.bbox[1] += y_offset
                p.bbox[2] += x_offset
                p.bbox[3] += y_offset
                # The mask would also need remapping if segmentation is used, 
                # but for simplicity in MVP we just return it as is or pad it.
                # Cross-tile NMS will handle duplicates
                all_predictions.append(p)
                
        return all_predictions

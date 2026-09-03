import os
import logging
from pathlib import Path
from typing import List, Union
import numpy as np
import torch
from ultralytics import YOLO

from src.utils.inference_utils import RawPrediction
from src.utils.config import Config

class ModelWrapper:
    def __init__(self, config: Config):
        self.config = config.model
        self.device = self._select_device()
        self.model = self._load_model()
        
    def _select_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
            
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_model(self) -> getattr(YOLO, "__class__", YOLO):
        model_path = Path(self.config.weights)
        if not model_path.exists():
            # If not found locally, ultralytics might download it if it's a known model (like yolov8s.pt)
            logging.info(f"Model file {model_path} not found locally. Assuming pretrained default.")
            return YOLO(self.config.weights)
            
        logging.info(f"Loading model from {model_path}")
        size_mb = model_path.stat().st_size / (1024 * 1024)
        logging.info(f"Model size: {size_mb:.2f} MB")
        
        return YOLO(str(model_path))

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25, augment: bool = False) -> List[RawPrediction]:
        # Ultralytics predictor
        results = self.model.predict(
            source=image, 
            device=self.device, 
            imgsz=self.config.input_size,
            conf=conf_threshold,
            augment=augment,
            verbose=False
        )
        
        predictions = []
        for r in results:
            boxes = r.boxes
            masks = r.masks
            
            if boxes is None:
                continue
                
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                mask = None
                if masks is not None:
                    # masks.data is [N, H, W]
                    mask = masks.data[i].cpu().numpy()
                    
                predictions.append(RawPrediction(
                    bbox=xyxy,
                    conf=conf,
                    cls_id=cls_id,
                    mask=mask
                ))
                
        return predictions

    def get_class_names(self):
        return self.model.names if hasattr(self.model, "names") else {}

import json
import csv
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict

from src.utils.detection import Detection, CoordinateType
from src.utils.config import Config

class ReportGenerator:
    def __init__(self, config: Config):
        self.config = config.reporting

    def to_csv(self, detections: List[Detection], path: Path):
        if not detections and not self.config.include_empty_detections:
            return
            
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Filter out keys that have None for ALL detections (e.g. no lat/lon anywhere)
        # to never write empty lat/lon columns if no GPS data
        has_gps = any(d.latitude is not None for d in detections)
        has_dims = any(d.width_m is not None for d in detections)
        
        headers = ["detection_id", "image_path", "class_name", "confidence", "coordinate_type", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]
        if has_gps:
            headers.extend(["latitude", "longitude", "timestamp", "ping_id"])
        if has_dims:
            headers.extend(["width_m", "height_m"])
            
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for d in detections:
                row = [
                    d.detection_id,
                    d.image_path,
                    d.class_name,
                    f"{d.confidence:.4f}",
                    d.coordinate_type.value,
                    int(d.bbox[0]), int(d.bbox[1]), int(d.bbox[2]), int(d.bbox[3])
                ]
                if has_gps:
                    row.extend([d.latitude if d.latitude else "", d.longitude if d.longitude else "", d.timestamp if d.timestamp else "", d.ping_id if d.ping_id else ""])
                if has_dims:
                    row.extend([f"{d.width_m:.2f}" if d.width_m else "", f"{d.height_m:.2f}" if d.height_m else ""])
                
                writer.writerow(row)

    def to_json(self, detections: List[Detection], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            "session_summary": self.session_summary(detections),
            "detections": []
        }
        
        for d in detections:
            d_dict = {
                "detection_id": d.detection_id,
                "image_path": d.image_path,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox": d.bbox,
                "coordinate_type": d.coordinate_type.value
            }
            if d.latitude is not None:
                d_dict["latitude"] = d.latitude
                d_dict["longitude"] = d.longitude
            if d.timestamp is not None:
                d_dict["timestamp"] = d.timestamp
            if d.width_m is not None:
                d_dict["width_m"] = d.width_m
                d_dict["height_m"] = d.height_m
                
            output["detections"].append(d_dict)
            
        with open(path, 'w') as f:
            json.dump(output, f, indent=4)

    def to_annotated_images(self, image_path: str, detections: List[Detection], output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        
        img = cv2.imread(image_path)
        if img is None:
            return
            
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d.bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Mask rendering
            if d.mask is not None:
                color = np.array([0, 255, 0], dtype=np.uint8)
                mask_indices = d.mask > 0.5
                img[mask_indices] = img[mask_indices] * 0.5 + color * 0.5
                
            label = f"{d.class_name} {d.confidence:.2f}"
            cv2.putText(img, label, (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        out_path = output_dir / Path(image_path).name
        cv2.imwrite(str(out_path), img)

    def session_summary(self, detections: List[Detection]) -> dict:
        summary = {
            "total_detections": len(detections),
            "classes": {},
            "avg_confidence": 0.0
        }
        
        if not detections:
            return summary
            
        total_conf = 0.0
        for d in detections:
            total_conf += d.confidence
            summary["classes"][d.class_name] = summary["classes"].get(d.class_name, 0) + 1
            
        summary["avg_confidence"] = total_conf / len(detections)
        return summary

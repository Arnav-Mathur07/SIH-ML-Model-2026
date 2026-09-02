from typing import List
from src.utils.detection import Detection

def compute_iou(box1: List[float], box2: List[float]) -> float:
    # box: [xmin, ymin, xmax, ymax]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def non_max_suppression(detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    """Applies cross-tile NMS to merge duplicate detections across tile boundaries."""
    if len(detections) == 0:
        return []
        
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
    
    keep = []
    
    for det in detections:
        discard = False
        for kept_det in keep:
            if det.class_id == kept_det.class_id:
                if compute_iou(det.bbox, kept_det.bbox) > iou_threshold:
                    discard = True
                    break
        if not discard:
            keep.append(det)
            
    return keep

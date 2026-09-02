from src.utils.detection import Detection, CoordinateType
from src.postprocessing.filters import GeometricFilter, FilterLog

def test_geometric_filter():
    filter_log = FilterLog()
    geo_filter = GeometricFilter(min_area=100, max_area_fraction=0.5, image_shape=(1000, 1000))
    
    # Area = 10x10 = 100 (Passes)
    det_pass = Detection(
        detection_id="1", image_path="test.png", class_id=0, class_name="net", 
        confidence=0.9, bbox=[0, 0, 10, 10], mask=None, coordinate_type=CoordinateType.IMAGE_SPACE
    )
    
    # Area = 5x5 = 25 (Fails min_area)
    det_fail_small = Detection(
        detection_id="2", image_path="test.png", class_id=0, class_name="net", 
        confidence=0.9, bbox=[0, 0, 5, 5], mask=None, coordinate_type=CoordinateType.IMAGE_SPACE
    )
    
    # Area = 800x800 = 640000 (Fails max_area - 0.64 > 0.5)
    det_fail_large = Detection(
        detection_id="3", image_path="test.png", class_id=0, class_name="net", 
        confidence=0.9, bbox=[0, 0, 800, 800], mask=None, coordinate_type=CoordinateType.IMAGE_SPACE
    )
    
    filtered = geo_filter.filter([det_pass, det_fail_small, det_fail_large], filter_log)
    
    assert len(filtered) == 1
    assert filtered[0].detection_id == "1"
    assert filter_log.rejected_count == 2

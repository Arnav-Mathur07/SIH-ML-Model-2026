import pytest
from src.utils.detection import Detection, CoordinateType
from src.geospatial.metadata import SonarMetadata
from src.geospatial.geotagging import GeospatialEngine

def test_geotagging(mock_config):
    # Setup test
    engine = GeospatialEngine(mock_config, image_shape=(1000, 1000))
    
    det = Detection(
        detection_id="1", image_path="test.png", class_id=0, class_name="net", 
        confidence=0.9, bbox=[400, 400, 600, 600], mask=None, coordinate_type=CoordinateType.IMAGE_SPACE
    )
    
    meta = SonarMetadata(
        filename="test.png",
        latitude=0.0,
        longitude=0.0,
        heading=0.0, # North
        across_track_resolution=0.1, # 10 cm per pixel
        along_track_resolution=0.1
    )
    
    # Process
    res = engine.process([det], meta)
    
    # The center of bbox is (500, 500)
    # The nadir is 500 (image center)
    # So distance from nadir is 0
    # The position should be exactly the vessel position
    
    assert res[0].coordinate_type == CoordinateType.GPS_ACCURATE
    assert pytest.approx(res[0].latitude) == 0.0
    assert pytest.approx(res[0].longitude) == 0.0
    assert res[0].width_m == 200 * 0.1 # 20.0 meters
    assert res[0].height_m == 200 * 0.1

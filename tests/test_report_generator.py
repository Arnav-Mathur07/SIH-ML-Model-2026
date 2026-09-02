import json
import csv
from pathlib import Path
from src.utils.detection import Detection, CoordinateType
from src.reporting.report_generator import ReportGenerator

def test_report_generator(tmp_path, mock_config):
    gen = ReportGenerator(mock_config)
    
    det = Detection(
        detection_id="1", image_path="test.png", class_id=0, class_name="net", 
        confidence=0.9, bbox=[0, 0, 10, 10], mask=None, coordinate_type=CoordinateType.IMAGE_SPACE
    )
    
    csv_path = tmp_path / "test.csv"
    json_path = tmp_path / "test.json"
    
    gen.to_csv([det], csv_path)
    gen.to_json([det], json_path)
    
    assert csv_path.exists()
    assert json_path.exists()
    
    # Check CSV for NO lat/lon columns since it's IMAGE_SPACE
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        assert "latitude" not in header
        assert "longitude" not in header
        
    # Check JSON schema
    with open(json_path, 'r') as f:
        data = json.load(f)
        assert "session_summary" in data
        assert "detections" in data
        assert len(data["detections"]) == 1

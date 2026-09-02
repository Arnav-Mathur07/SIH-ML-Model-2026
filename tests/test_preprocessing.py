import numpy as np
from src.data.preprocessing import SonarPreprocessor

def test_preprocessing_shape_and_range(mock_config):
    # Mock image 100x100 grayscale
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    preprocessor = SonarPreprocessor(mock_config)
    output = preprocessor.preprocess(image)
    
    # Check shape: should be 3-channel for YOLO
    assert len(output.shape) == 3
    assert output.shape[2] == 3
    
    # Check no NaNs
    assert not np.isnan(output).any()
    
    # Normalization check
    assert output.min() >= 0
    assert output.max() <= 255

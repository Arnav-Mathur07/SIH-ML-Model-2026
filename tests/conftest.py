import pytest
from src.utils.config import Config, PathsConfig, ModelConfig, PreprocessingConfig, PostprocessingConfig, GeospatialConfig, ReportingConfig, ProjectConfig

@pytest.fixture
def mock_config():
    config = Config(
        project=ProjectConfig(name="test", version="1.0"),
        paths=PathsConfig(),
        model=ModelConfig(),
        preprocessing=PreprocessingConfig(enabled=True),
        postprocessing=PostprocessingConfig(),
        geospatial=GeospatialConfig(),
        reporting=ReportingConfig()
    )
    return config

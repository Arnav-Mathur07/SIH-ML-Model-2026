import yaml
import logging
from typing import List, Dict, Union
from pydantic import BaseModel, Field
from pathlib import Path

class PathsConfig(BaseModel):
    data_root: str = "data/sonar"
    images_dir: str = "data/sonar/images"
    labels_dir: str = "data/sonar/labels"
    metadata_dir: str = "data/sonar/metadata"
    processed_dir: str = "data/processed"
    outputs_dir: str = "outputs"
    models_dir: str = "models"

class ModelConfig(BaseModel):
    architecture: str = "yolov8s-seg"
    weights: str = "yolov8s-seg.pt"
    input_size: int = 640
    device: str = "auto"

class TrainingConfig(BaseModel):
    epochs: int = 100
    patience: int = 20
    batch_size: int = 16
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    val_split: float = 0.2
    test_split: float = 0.1
    mosaic: float = 1.0
    copy_paste: float = 0.3

class TilingConfig(BaseModel):
    enabled: bool = False
    tile_size: int = 640
    overlap: float = 0.2

class PreprocessingConfig(BaseModel):
    enabled: bool = True
    median_filter: bool = True
    median_kernel: int = 3
    bilateral_filter: bool = True
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: List[int] = [8, 8]
    normalize: bool = True
    tiling: TilingConfig = TilingConfig()

class AugmentationConfig(BaseModel):
    flipud: float = 0.5
    fliplr: float = 0.5
    degrees: float = 5.0
    scale: float = 0.3
    translate: float = 0.1
    hsv_v: float = 0.2
    hsv_h: float = 0.0
    hsv_s: float = 0.0
    noise_injection: bool = True
    random_erasing: float = 0.1

class ConfidenceThresholds(BaseModel):
    default: float = 0.35

class PostprocessingConfig(BaseModel):
    confidence_thresholds: ConfidenceThresholds = ConfidenceThresholds()
    iou_threshold: float = 0.45
    min_area_px: int = 64
    max_area_fraction: float = 0.40
    shadow_consistency_check: bool = True
    nadir_position: str = "top"

class GeospatialConfig(BaseModel):
    coordinate_type_required_fields: List[str] = [
        "latitude", "longitude", "heading", "across_track_resolution"
    ]

class ReportingConfig(BaseModel):
    output_formats: List[str] = ["csv", "json"]
    include_empty_detections: bool = False

class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "outputs/run.log"

class ProjectConfig(BaseModel):
    name: str
    version: str
    seed: int = 42

class Config(BaseModel):
    project: ProjectConfig
    paths: PathsConfig = PathsConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    postprocessing: PostprocessingConfig = PostprocessingConfig()
    geospatial: GeospatialConfig = GeospatialConfig()
    reporting: ReportingConfig = ReportingConfig()
    logging: LoggingConfig = LoggingConfig()

def load_config(config_path: Union[str, Path] = "configs/config.yaml") -> Config:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    try:
        config = Config(**config_dict)
        return config
    except Exception as e:
        logging.error(f"Configuration validation failed: {e}")
        raise

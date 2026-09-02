import argparse
import logging
from pathlib import Path
import cv2
import uuid

from src.utils.config import load_config
from src.utils.detection import Detection
from src.models.model import ModelWrapper
from src.models.inference import TiledInferenceEngine
from src.data.preprocessing import SonarPreprocessor
from src.postprocessing.pipeline import PostProcessingPipeline
from src.geospatial.metadata import MetadataParser
from src.geospatial.geotagging import GeospatialEngine
from src.reporting.report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to image or directory")
    parser.add_argument("--model", type=str, default=None, help="Path to weights")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config.model.weights = args.model
        
    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input path not found: {input_path}")
        return
        
    image_paths = [input_path] if input_path.is_file() else list(input_path.rglob("*.png")) + list(input_path.rglob("*.jpg"))
    if not image_paths:
        logging.error("No valid images found.")
        return

    # Initialize components
    model = ModelWrapper(config)
    inference_engine = TiledInferenceEngine(model, config)
    preprocessor = SonarPreprocessor(config)
    metadata_parser = MetadataParser(Path(config.paths.metadata_dir))
    report_gen = ReportGenerator(config)

    all_detections = []
    class_names = model.get_class_names()

    logging.info(f"Starting inference on {len(image_paths)} images...")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            logging.warning(f"Could not read {img_path}")
            continue

        h, w = img.shape[:2]
        
        # 1. Preprocess
        preprocessed = preprocessor.preprocess(img)
        
        # 2. Inference
        raw_preds, inf_time = inference_engine.run_inference(preprocessed)
        
        # Convert to Detection objects
        detections = []
        for rp in raw_preds:
            cls_name = class_names.get(rp.class_id, str(rp.class_id))
            det = Detection(
                detection_id=str(uuid.uuid4())[:8],
                image_path=str(img_path),
                class_id=rp.class_id,
                class_name=cls_name,
                confidence=rp.confidence,
                bbox=rp.bbox,
                mask=rp.mask,
                coordinate_type=None,
                inference_time_ms=inf_time
            )
            detections.append(det)

        # 3. Postprocess
        post_pipeline = PostProcessingPipeline(config, (h, w))
        filtered_dets, filter_log = post_pipeline.process(detections)
        
        # 4. Geospatial
        meta = metadata_parser.get_metadata(img_path.name)
        geo_engine = GeospatialEngine(config, (h, w))
        geo_dets = geo_engine.process(filtered_dets, meta)
        
        all_detections.extend(geo_dets)
        
        # Render visual output
        out_vis_dir = Path(config.paths.outputs_dir) / "predictions"
        report_gen.to_annotated_images(str(img_path), geo_dets, out_vis_dir)

    # Final reports
    report_dir = Path(config.paths.outputs_dir) / "reports"
    report_gen.to_csv(all_detections, report_dir / "detections.csv")
    report_gen.to_json(all_detections, report_dir / "detections.json")
    
    logging.info("Inference complete.")
    logging.info(f"Total detections: {len(all_detections)}")
    logging.info(f"Reports saved to {report_dir}")

if __name__ == "__main__":
    main()

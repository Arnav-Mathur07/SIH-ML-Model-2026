import argparse
import logging
from pathlib import Path
import json
from ultralytics import YOLO

from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = Path(args.model)
    
    if not model_path.exists():
        logging.error(f"Model file not found: {model_path}")
        return
        
    logging.info(f"Loading model {model_path} for evaluation...")
    model = YOLO(str(model_path))
    
    dataset_yaml = Path("configs/dataset.yaml")
    if not dataset_yaml.exists():
        logging.error("configs/dataset.yaml not found. Please run training first or create it manually.")
        return
        
    logging.info("Running evaluation on validation set...")
    try:
        metrics = model.val(data=str(dataset_yaml))
        
        # Extract metrics
        results = {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        
        # Save metrics
        eval_dir = Path(config.paths.outputs_dir) / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        with open(eval_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=4)
            
        logging.info(f"Evaluation complete. Metrics saved to {eval_dir / 'metrics.json'}")
        logging.info(f"mAP50: {results['mAP50']:.4f}")
        logging.info(f"Precision: {results['precision']:.4f}")
        logging.info(f"Recall: {results['recall']:.4f}")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()

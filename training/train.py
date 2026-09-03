import argparse
import sys
import logging
from pathlib import Path
import yaml
from ultralytics import YOLO

from src.utils.config import load_config
from scripts.inspect_dataset import inspect_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    # 1. Load config
    config = load_config(args.config)

    # We now use the statically generated dataset.yaml 
    dataset_yaml = Path("configs/dataset.yaml")
    if not dataset_yaml.exists():
        logging.error("configs/dataset.yaml not found!")
        sys.exit(1)
    
    # 5. Initialize model
    logging.info(f"Initializing model: {config.model.weights}")
    model = YOLO(config.model.weights)
    
    # 6. Train
    logging.info("Starting training...")
    try:
        model.train(
            data=str(dataset_yaml),
            epochs=config.training.epochs,
            patience=config.training.patience,
            batch=config.training.batch_size,
            optimizer=config.training.optimizer,
            lr0=config.training.lr0,
            lrf=config.training.lrf,
            weight_decay=config.training.weight_decay,
            warmup_epochs=config.training.warmup_epochs,
            val=True,
            device="cpu" if config.model.device == "cpu" else "", # Empty uses auto in YOLOv8
            project=config.paths.outputs_dir,
            name="training_run",
            exist_ok=True
        )
        logging.info("Training complete.")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

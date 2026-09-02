import argparse
import sys
import logging
from pathlib import Path
import yaml
from ultralytics import YOLO

from src.utils.config import load_config
from scripts.inspect_dataset import inspect_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def create_dataset_yaml(config, classes):
    dataset_yaml_path = Path("configs/dataset.yaml")
    dataset_dict = {
        "path": str(Path(config.paths.data_root).absolute()),
        "train": "images",
        "val": "images", # Simplification: should use proper split dirs in full implementation
        "names": {i: name for i, name in enumerate(classes)}
    }
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dataset_dict, f)
    return dataset_yaml_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    # 1. Load config
    config = load_config(args.config)
    
    # 2. Inspect dataset
    inspect_dataset()
    
    # 3. Check for classes
    classes_file = Path(config.paths.labels_dir) / "classes.txt"
    if not classes_file.exists():
        logging.error("No classes.txt found. Annotations might be missing or dataset inspection failed.")
        sys.exit(1)
        
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
        
    if not classes:
        logging.error("classes.txt is empty. Cannot train.")
        sys.exit(1)
        
    # 4. Generate dataset.yaml dynamically
    dataset_yaml = create_dataset_yaml(config, classes)
    
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

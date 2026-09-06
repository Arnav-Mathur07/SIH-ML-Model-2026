import os
import shutil
import random
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def build_sonar_dataset():
    """
    Builds a pure Side-Scan Sonar dataset from data/sonar
    Splits into 80/10/10 train/val/test splits.
    """
    base_dir = Path("data")
    sonar_dir = base_dir / "sonar"
    out_dir = base_dir / "dataset"
    
    if out_dir.exists():
        logging.info("Cleaning old dataset directory...")
        shutil.rmtree(out_dir)
        
    for split in ['train', 'val', 'test']:
        (out_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (out_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
    images_dir = sonar_dir / "images"
    labels_dir = sonar_dir / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        logging.error("Sonar images or labels directory not found!")
        return

    # Get all valid image files that have corresponding labels
    all_images = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        all_images.extend(list(images_dir.glob(ext)))
        
    valid_pairs = []
    for img_path in all_images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_pairs.append((img_path, label_path))
            
    total = len(valid_pairs)
    logging.info(f"Found {total} valid Sonar image/label pairs.")
    
    if total == 0:
        return
        
    # Shuffle and split 80/10/10
    random.seed(42)
    random.shuffle(valid_pairs)
    
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    splits = {
        'train': valid_pairs[:train_end],
        'val': valid_pairs[train_end:val_end],
        'test': valid_pairs[val_end:]
    }
    
    # Copy files
    for split_name, pairs in splits.items():
        logging.info(f"Copying {len(pairs)} files to {split_name} split...")
        for img_path, label_path in pairs:
            # We enforce all labels to be class 0 (Debris/Anomaly)
            new_label_path = out_dir / split_name / 'labels' / label_path.name
            with open(label_path, 'r') as f_in, open(new_label_path, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Force class ID to 0
                        f_out.write(f"0 {' '.join(parts[1:])}\n")
            
            # Copy image
            shutil.copy2(img_path, out_dir / split_name / 'images' / img_path.name)
            
    logging.info("Successfully built pure Side-Scan Sonar dataset!")

if __name__ == "__main__":
    build_sonar_dataset()

import os
import shutil
import random
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def get_pairs_from_dir(img_dir, lbl_dir=None):
    pairs = []
    if lbl_dir is None:
        lbl_dir = img_dir # Labels are alongside images
        
    img_dir_path = Path(img_dir)
    lbl_dir_path = Path(lbl_dir)
    
    if not img_dir_path.exists():
        return pairs
        
    for img_path in img_dir_path.glob("*.jpg"):
        lbl_path = lbl_dir_path / f"{img_path.stem}.txt"
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
    return pairs

def main():
    random.seed(42)
    
    # 1. Collect all pairs from all sources
    all_pairs = []
    
    # Sonar Data
    all_pairs.extend(get_pairs_from_dir("data/sonar/images", "data/sonar/labels"))
    
    # Optical Data 1 (TrashCan test folder)
    all_pairs.extend(get_pairs_from_dir("data/archive_extracted/test"))
    
    # Optical Data 2 (Underwater Plastics)
    base_opt = Path("data/archive_1_extracted/underwater_plastics")
    for split in ["train", "valid", "test"]:
        all_pairs.extend(get_pairs_from_dir(base_opt / split / "images", base_opt / split / "labels"))
        
    if not all_pairs:
        logging.error("No image/label pairs found!")
        return
        
    logging.info(f"Total image/label pairs found across all datasets: {len(all_pairs)}")
    
    # 2. Shuffle
    random.shuffle(all_pairs)
    
    # 3. Split 80/10/10
    total = len(all_pairs)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    
    splits = {
        "train": all_pairs[:train_end],
        "val": all_pairs[train_end:val_end],
        "test": all_pairs[val_end:]
    }
    
    dataset_out_dir = Path("data/dataset")
    
    # 4. Copy and Normalize to Class 0 (Debris)
    for split_name, pairs in splits.items():
        split_img_dir = dataset_out_dir / split_name / "images"
        split_lbl_dir = dataset_out_dir / split_name / "labels"
        
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Writing {len(pairs)} pairs to {split_name} split...")
        for img_path, lbl_path in pairs:
            # Copy image
            new_img_path = split_img_dir / f"{img_path.parent.parent.name}_{img_path.name}"
            shutil.copy2(img_path, new_img_path)
            
            # Read label, convert all class IDs to 0, write new label
            new_lbl_path = split_lbl_dir / f"{img_path.parent.parent.name}_{lbl_path.name}"
            
            with open(lbl_path, "r") as f_in, open(new_lbl_path, "w") as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = "0" # Force class ID to 0
                        f_out.write(" ".join(parts) + "\n")
                        
    logging.info("Universal dataset created successfully in data/dataset/")

if __name__ == "__main__":
    main()

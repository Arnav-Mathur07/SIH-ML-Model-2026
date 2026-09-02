import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class SonarDataset:
    def __init__(self, images_dir: Path, labels_dir: Path, metadata_dir: Path):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.metadata_dir = Path(metadata_dir)
        self.valid_exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
        self.items = self._discover_items()

    def _discover_items(self) -> List[Dict]:
        items = []
        if not self.images_dir.exists():
            logging.warning(f"Images directory not found: {self.images_dir}")
            return items

        for img_path in self.images_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in self.valid_exts:
                # Check for label
                label_name = img_path.stem + ".txt"
                label_path = self.labels_dir / label_name
                label = label_path if label_path.exists() else None

                # Check for metadata
                meta_name = img_path.stem + ".csv"
                meta_path = self.metadata_dir / meta_name
                meta = meta_path if meta_path.exists() else None
                
                # Check JSON metadata as alternative
                if not meta:
                    meta_name_json = img_path.stem + ".json"
                    meta_path_json = self.metadata_dir / meta_name_json
                    meta = meta_path_json if meta_path_json.exists() else None

                items.append({
                    "image": img_path,
                    "label": label,
                    "metadata": meta
                })
                
        logging.info(f"Discovered {len(items)} images in dataset.")
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]

import os
import json
import logging
from pathlib import Path
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import numpy as np

# Prevent PIL from crashing on truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def inspect_dataset():
    data_root = Path("data/sonar")
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"
    outputs_dir = Path("outputs")
    docs_dir = Path("docs")
    
    outputs_dir.mkdir(exist_ok=True)
    (outputs_dir / "visualizations").mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)

    if not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        print("No dataset found. Place sonar images into data/sonar/images/ and re-run inspect_dataset.py.")
        return

    # Image analysis
    valid_exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
    image_paths = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]
    
    total_images = len(image_paths)
    if total_images == 0:
        print("No dataset found. Place sonar images into data/sonar/images/ and re-run inspect_dataset.py.")
        return

    formats_present = set()
    widths = []
    heights = []
    channels = {"Grayscale": 0, "RGB": 0, "RGBA": 0, "Other": 0}
    corrupt_files = []
    sizes = []
    sample_images = []

    logging.info(f"Analyzing {total_images} images...")
    
    for img_path in image_paths:
        formats_present.add(img_path.suffix.lower())
        sizes.append(img_path.stat().st_size)
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                mode = img.mode
                widths.append(w)
                heights.append(h)
                
                if mode in ["L", "1", "I", "F"]:
                    channels["Grayscale"] += 1
                elif mode == "RGB":
                    channels["RGB"] += 1
                elif mode == "RGBA":
                    channels["RGBA"] += 1
                else:
                    channels["Other"] += 1
                    
                if len(sample_images) < 9:
                    # Keep thumbnail for grid
                    img.thumbnail((200, 200))
                    sample_images.append(img.copy())
        except Exception as e:
            corrupt_files.append(str(img_path))
            logging.error(f"Corrupt or unreadable file: {img_path} ({e})")

    # Generate sample grid
    if sample_images:
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        for i, ax in enumerate(axes.flat):
            if i < len(sample_images):
                ax.imshow(sample_images[i], cmap='gray' if sample_images[i].mode in ['L', '1', 'I', 'F'] else None)
            ax.axis('off')
        plt.tight_layout()
        grid_path = outputs_dir / "visualizations" / "sample_grid.png"
        fig.savefig(grid_path)
        plt.close(fig)
        logging.info(f"Saved sample grid to {grid_path}")

    # Annotations check
    annotations_found = False
    
    # Check YOLO
    yolo_labels = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
    if yolo_labels:
        annotations_found = True
        logging.info(f"Found {len(yolo_labels)} YOLO annotation files.")
        
    # Check COCO
    coco_json = data_root / "annotations.json"
    if coco_json.exists():
        annotations_found = True
        logging.info("Found COCO JSON. Running converter...")
        from src.data.converters.coco_to_yolo import convert_coco_to_yolo
        convert_coco_to_yolo(coco_json, labels_dir)
        
    # Check VOC
    voc_dir = data_root / "annotations"
    if voc_dir.exists() and list(voc_dir.glob("*.xml")):
        annotations_found = True
        logging.info("Found Pascal VOC XMLs. Running converter...")
        from src.data.converters.voc_to_yolo import convert_voc_to_yolo
        convert_voc_to_yolo(voc_dir, labels_dir)

    # Check CSV
    csv_files = list(data_root.glob("*.csv"))
    if csv_files:
        annotations_found = True
        logging.info("Found CSV annotations. Running converter...")
        from src.data.converters.csv_to_yolo import convert_csv_to_yolo
        convert_csv_to_yolo(csv_files[0], images_dir, labels_dir)

    if not annotations_found:
        print(f"Dataset contains {total_images} images but no supervised training annotations were detected. Supervised training is not possible. See docs/annotation_guide.md for labeling instructions.")

    # Metadata check
    metadata_dir = data_root / "metadata"
    metadata_files = list(metadata_dir.glob("*.*")) if metadata_dir.exists() else []
    metadata_found = len(metadata_files) > 0

    # Save stats
    stats = {
        "total_images": total_images,
        "formats": list(formats_present),
        "resolution_width": {"min": min(widths) if widths else 0, "max": max(widths) if widths else 0, "median": np.median(widths) if widths else 0},
        "resolution_height": {"min": min(heights) if heights else 0, "max": max(heights) if heights else 0, "median": np.median(heights) if heights else 0},
        "channels": channels,
        "corrupt_files_count": len(corrupt_files),
        "avg_file_size_bytes": np.mean(sizes) if sizes else 0,
        "annotations_found": annotations_found,
        "metadata_found": metadata_found
    }
    
    stats_file = outputs_dir / "dataset_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=4)
        
    # Write report
    report_lines = [
        "# Dataset Report",
        f"**Total Images:** {total_images}",
        f"**Formats Present:** {', '.join(formats_present)}",
        f"**Resolution Width:** {stats['resolution_width']['min']} - {stats['resolution_width']['max']} (Median: {stats['resolution_width']['median']})",
        f"**Resolution Height:** {stats['resolution_height']['min']} - {stats['resolution_height']['max']} (Median: {stats['resolution_height']['median']})",
        "**Channels:** " + ", ".join([f"{k}: {v}" for k, v in channels.items()]),
        f"**Corrupt Files:** {len(corrupt_files)}",
        f"**Average File Size:** {stats['avg_file_size_bytes'] / 1024:.2f} KB",
        f"**Annotations Found:** {annotations_found}",
        f"**Metadata Found:** {metadata_found}",
    ]
    
    with open(docs_dir / "dataset_report.md", "w") as f:
        f.write("\n".join(report_lines) + "\n")
        
    logging.info("Inspection complete. Saved reports.")

if __name__ == "__main__":
    inspect_dataset()

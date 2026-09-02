import argparse
import logging
from pathlib import Path
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def convert_csv_to_yolo(csv_path: Path, images_dir: Path, output_dir: Path):
    if not csv_path.exists():
        logging.error(f"File not found: {csv_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Failed to read CSV {csv_path}: {e}")
        return

    required_cols = ['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    if not all(col in df.columns for col in required_cols):
        logging.error(f"CSV must contain columns: {required_cols}")
        return

    classes = sorted(df['class'].unique().tolist())
    
    converted_count = 0
    grouped = df.groupby('filename')
    
    for filename, group in grouped:
        img_path = images_dir / filename
        if not img_path.exists():
            logging.warning(f"Image {filename} not found, skipping annotations.")
            continue
            
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            logging.warning(f"Failed to open image {img_path}")
            continue
            
        yolo_lines = []
        for _, row in group.iterrows():
            cls_id = classes.index(row['class'])
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']
            
            x_center = ((xmin + xmax) / 2.0) / w
            y_center = ((ymin + ymax) / 2.0) / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h
            
            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
            
        if yolo_lines:
            output_file = output_dir / f"{Path(filename).stem}.txt"
            with open(output_file, 'w') as f:
                f.write("\n".join(yolo_lines) + "\n")
            converted_count += 1
            
    logging.info(f"Converted annotations for {converted_count} images to YOLO format in {output_dir}")
    
    with open(output_dir / "classes.txt", 'w') as f:
        for c in classes:
            f.write(f"{c}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV annotations to YOLO format")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing corresponding images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save YOLO txt files")
    args = parser.parse_args()
    convert_csv_to_yolo(Path(args.csv), Path(args.images_dir), Path(args.output_dir))

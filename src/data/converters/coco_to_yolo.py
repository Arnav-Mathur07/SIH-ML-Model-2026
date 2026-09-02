import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def convert_coco_to_yolo(coco_json_path: Path, output_dir: Path):
    if not coco_json_path.exists():
        logging.error(f"File not found: {coco_json_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
        
    categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    images = {img['id']: img for img in data.get('images', [])}
    
    # map old category IDs to 0-indexed YOLO IDs
    cat_id_to_yolo_id = {old_id: idx for idx, old_id in enumerate(sorted(categories.keys()))}
    
    annotations_by_image = {}
    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
        
    converted_count = 0
    for img_id, anns in annotations_by_image.items():
        if img_id not in images:
            continue
            
        img_info = images[img_id]
        img_w = img_info['width']
        img_h = img_info['height']
        file_name = Path(img_info['file_name']).stem
        
        yolo_lines = []
        for ann in anns:
            cat_id = ann['category_id']
            yolo_id = cat_id_to_yolo_id.get(cat_id, 0)
            
            # Check for segmentation
            if 'segmentation' in ann and isinstance(ann['segmentation'], list) and len(ann['segmentation']) > 0:
                # Use the first polygon
                polygon = ann['segmentation'][0]
                # Normalize
                norm_poly = []
                for i in range(0, len(polygon), 2):
                    x = polygon[i] / img_w
                    y = polygon[i+1] / img_h
                    norm_poly.extend([f"{x:.6f}", f"{y:.6f}"])
                
                line = f"{yolo_id} " + " ".join(norm_poly)
                yolo_lines.append(line)
            else:
                # Bounding box fallback
                bbox = ann['bbox'] # [x_min, y_min, width, height]
                x_center = (bbox[0] + bbox[2] / 2) / img_w
                y_center = (bbox[1] + bbox[3] / 2) / img_h
                w = bbox[2] / img_w
                h = bbox[3] / img_h
                yolo_lines.append(f"{yolo_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                
        if yolo_lines:
            output_file = output_dir / f"{file_name}.txt"
            with open(output_file, 'w') as f:
                f.write("\n".join(yolo_lines) + "\n")
            converted_count += 1
            
    logging.info(f"Converted {converted_count} COCO annotations to YOLO format in {output_dir}")
    
    # Save a classes.txt
    with open(output_dir / "classes.txt", 'w') as f:
        for old_id in sorted(categories.keys()):
            f.write(f"{categories[old_id]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO JSON to YOLO format")
    parser.add_argument("--input", type=str, required=True, help="Path to COCO JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save YOLO txt files")
    args = parser.parse_args()
    convert_coco_to_yolo(Path(args.input), Path(args.output_dir))

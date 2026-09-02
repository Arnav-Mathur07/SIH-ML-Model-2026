import argparse
import logging
from pathlib import Path
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def convert_voc_to_yolo(voc_dir: Path, output_dir: Path):
    if not voc_dir.exists():
        logging.error(f"Directory not found: {voc_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    xml_files = list(voc_dir.glob("*.xml"))
    
    if not xml_files:
        logging.warning(f"No XML files found in {voc_dir}")
        return
        
    classes = []
    converted_count = 0
    
    # First pass to find all classes
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall("object"):
                cls_name = obj.find("name").text
                if cls_name not in classes:
                    classes.append(cls_name)
        except Exception:
            continue
            
    classes = sorted(classes)
    
    # Second pass to convert
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            size = root.find("size")
            w = int(size.find("width").text)
            h = int(size.find("height").text)
            
            if w == 0 or h == 0:
                logging.warning(f"Invalid dimensions in {xml_file.name}")
                continue
                
            yolo_lines = []
            for obj in root.findall("object"):
                cls_name = obj.find("name").text
                cls_id = classes.index(cls_name)
                
                xmlbox = obj.find("bndbox")
                xmin = float(xmlbox.find("xmin").text)
                xmax = float(xmlbox.find("xmax").text)
                ymin = float(xmlbox.find("ymin").text)
                ymax = float(xmlbox.find("ymax").text)
                
                # Normalize
                x_center = ((xmin + xmax) / 2.0) / w
                y_center = ((ymin + ymax) / 2.0) / h
                box_w = (xmax - xmin) / w
                box_h = (ymax - ymin) / h
                
                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
                
            if yolo_lines:
                output_file = output_dir / f"{xml_file.stem}.txt"
                with open(output_file, 'w') as f:
                    f.write("\n".join(yolo_lines) + "\n")
                converted_count += 1
                
        except Exception as e:
            logging.error(f"Failed to process {xml_file.name}: {e}")
            
    logging.info(f"Converted {converted_count} VOC annotations to YOLO format in {output_dir}")
    
    with open(output_dir / "classes.txt", 'w') as f:
        for c in classes:
            f.write(f"{c}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML to YOLO format")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing VOC XML files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save YOLO txt files")
    args = parser.parse_args()
    convert_voc_to_yolo(Path(args.input_dir), Path(args.output_dir))

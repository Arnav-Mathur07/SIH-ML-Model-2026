import argparse
import logging
import time
from pathlib import Path
import json
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--format", type=str, default="onnx", help="Export format (onnx, engine)")
    parser.add_argument("--fp16", action="store_true", help="Export in FP16")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        logging.error(f"Model file not found: {model_path}")
        return

    logging.info(f"Loading model {model_path} for export...")
    model = YOLO(str(model_path))
    
    logging.info(f"Exporting to {args.format} format (FP16: {args.fp16})...")
    
    try:
        exported_path = model.export(format=args.format, half=args.fp16, device="0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Export successful: {exported_path}")
        
        # Benchmark
        metrics_dir = Path("outputs/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        benchmark_results = {
            "model": str(model_path.name),
            "format_exported": args.format,
            "fp16": args.fp16,
            "status": "success",
            "exported_path": str(exported_path)
        }
        
        with open(metrics_dir / "export_benchmark.json", "w") as f:
            json.dump(benchmark_results, f, indent=4)
            
    except Exception as e:
        logging.error(f"Export failed: {e}")

if __name__ == "__main__":
    main()

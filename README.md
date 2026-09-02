# AI-Powered Underwater Marine Debris Detection System

An end-to-end Machine Learning pipeline for side-scan sonar image analysis.

## Problem Statement

Detecting marine debris (like ghost nets, lost cargo, and pollution) underwater is a significant challenge due to limited visibility, turbidity, and challenging marine environments. Side-scan sonar offers a robust alternative to optical cameras by capturing high-resolution acoustic imagery across wide swaths of the ocean floor, unhindered by water clarity.

This system provides an autonomous AI solution capable of identifying, segmenting, and geographically plotting marine debris from side-scan sonar imagery. It leverages state-of-the-art deep learning (YOLOv8 instance segmentation) combined with specialized sonar preprocessing, robust geospatial projection, and an interactive dashboard for analysis.

## System Architecture

```text
[ Sonar Imagery ] --> ( Preprocessing Engine ) --> [ YOLOv8-seg ] --> ( Post-Processing & Filtering )
                                                                                      |
                                                                                      v
                                                                             [ Geospatial Engine ]
                                                                                      |
                                                                                      v
                                                                             [ Dashboard / Reports ]
```

## Requirements

- Python 3.9 - 3.12
- Recommended: NVIDIA GPU with at least 6GB VRAM (CUDA support) or Apple Silicon. CPU inference is supported but slower.

## Installation

```bash
git clone <repository>
cd underwater-marine-debris-ai
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Dataset Setup

Place your dataset inside `data/sonar/images/`. The system will recursively discover images in all subfolders.

Supported formats: PNG, JPG, TIFF, BMP, WEBP.
Annotation formats: YOLO (.txt), COCO (.json), or Pascal VOC (.xml).

To verify the dataset, run:
```bash
python scripts/inspect_dataset.py
```

## How to Annotate

Please see [docs/annotation_guide.md](docs/annotation_guide.md) for instructions on how to annotate sonar imagery correctly for this pipeline.

## Usage Commands

- **Train**: `python training/train.py --config configs/config.yaml`
- **Evaluate**: `python training/evaluate.py --model outputs/.../best.pt --config configs/config.yaml`
- **Inference**: `python scripts/run_inference.py --input data/sonar/images --config configs/config.yaml`
- **Dashboard**: `streamlit run dashboard/app.py`
- **Export to Edge (ONNX)**: `python training/export.py --model outputs/.../best.pt --format onnx`

## Known Limitations
- Sonar geometry estimation is flat-Earth based and approximates nadir position; high-precision surveying requires rigorous slant-range correction.
- The default bounding boxes from segmentation are tight; heavy shadows might extend beyond the mask unless explicitly labeled.

## Future Improvements
- Integrating acoustic ray-tracing for more precise geographic mapping.
- Adding active learning capabilities to refine edge cases in challenging environments.

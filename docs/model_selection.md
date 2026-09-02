# Model Selection Justification

**Primary Model Chosen:** YOLOv8s-seg (YOLOv8 Small - Instance Segmentation) OR YOLOv8s (Object Detection)

## Decision Logic

The system dynamically adapts to the provided dataset's annotation structure:
- **Instance Segmentation (`yolov8s-seg`)**: Chosen if polygon/mask annotations (COCO, or YOLO seg format) are present. This is the preferred method for sonar imagery because acoustic shadows often follow irregular shapes and segmentation yields highly accurate physical dimension estimations.
- **Object Detection (`yolov8s`)**: Chosen if only bounding box annotations (Pascal VOC, CSV) are available.

## Why YOLOv8?

1. **Speed & Efficiency**: YOLOv8s operates at real-time speeds (>30 FPS) even on edge devices (NVIDIA Jetson) or moderate GPUs (RTX 3050). The small variant (approx 11MB-22MB) is perfect for constrained marine deployments.
2. **Unified Architecture**: YOLOv8 provides a unified framework for both bounding box detection and instance segmentation, allowing this single pipeline to support both seamlessly.
3. **Robustness**: It handles varying scales and object densities exceptionally well, a common scenario with scattered marine debris.

## Why Not Alternatives?

- **U-Net / DeepLab**: While excellent for semantic segmentation, they do not inherently separate overlapping instances of the same class (instance segmentation). They also struggle with real-time requirements without significant optimization.
- **Mask R-CNN / Faster R-CNN**: Highly accurate but computationally heavy. The inference latency is significantly higher than YOLOv8, making it less suitable for autonomous edge deployment (AUVs) where power budget is strictly limited.
- **YOLOv8x / YOLOv9**: The extra-large models offer diminishing returns on sonar imagery which inherently lacks the high-frequency textural detail of optical photos, but they triple the computational cost and VRAM usage.

## Hardware & Edge Deployment

- **VRAM Requirement**: < 2GB for inference at 640x640 resolution.
- **Export Paths**: Natively supports export to ONNX and TensorRT for deployment on NVIDIA Jetson Orin/Nano series.

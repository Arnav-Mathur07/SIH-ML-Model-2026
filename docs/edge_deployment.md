# Edge Deployment Guide

This guide explains how to deploy the trained marine debris detection model onto edge hardware, such as the NVIDIA Jetson Orin Nano, AGX Xavier, or autonomous underwater vehicles (AUVs) equipped with companion computers.

## 1. Exporting the Model

PyTorch (`.pt`) models are excellent for training, but less optimal for edge deployment due to higher overhead. To maximize frame rate and reduce power consumption, you should export to ONNX or TensorRT.

### Export to ONNX (General Edge)
```bash
python training/export.py --model outputs/training_run/weights/best.pt --format onnx
```

### Export to TensorRT (NVIDIA Jetson)
TensorRT is highly recommended for Jetson devices as it optimizes the network for the specific GPU architecture.
```bash
python training/export.py --model outputs/training_run/weights/best.pt --format engine --fp16
```
*Note: TensorRT engines must be built on the target hardware (i.e., run this export command directly on the Jetson).*

## 2. Running Inference on Edge

When running `scripts/run_inference.py`, simply pass the `.onnx` or `.engine` file as the `--model` argument. The YOLOv8 wrapper automatically handles ONNX/TensorRT inference.

```bash
python scripts/run_inference.py --input data/sonar/images --model outputs/training_run/weights/best.engine
```

## 3. Power Optimization Tips
- **Use FP16**: Half-precision floating point (`--fp16`) significantly speeds up inference on modern GPUs with almost zero loss in accuracy.
- **Tiling**: If tiling is enabled in `configs/config.yaml`, it will drastically increase memory usage and inference time. Disable tiling if edge memory is constrained (< 4GB).
- **Batch Size**: For real-time inference on streams, ensure batch size is 1.

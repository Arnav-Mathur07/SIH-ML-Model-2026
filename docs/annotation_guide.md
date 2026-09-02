# Annotation Guide for Sonar Imagery

To train the AI models for marine debris detection, you need to properly annotate your side-scan sonar images. This guide explains how to do it.

## 1. Allowed Annotation Formats

The project supports the following formats out of the box:
- **YOLO TXT**: Native format. (Recommended for bounding boxes & segmentation masks)
- **COCO JSON**: Commonly used for polygons. (Will be auto-converted)
- **Pascal VOC XML**: Standard bounding box format. (Will be auto-converted)

## 2. Choosing an Annotation Tool

We recommend one of the following tools:
- **[LabelImg](https://github.com/HumanSignal/labelImg)**: Great for simple bounding boxes.
- **[CVAT (Computer Vision Annotation Tool)](https://github.com/cvat-ai/cvat)**: Excellent for instance segmentation (polygons) and large datasets.
- **[Roboflow](https://roboflow.com/)**: Good cloud-based alternative.

## 3. How to Annotate Sonar Data

1. **Identify the Target**: Look for the bright acoustic return of the object.
2. **Identify the Shadow**: The dark acoustic shadow stretching away from the nadir (center/top/bottom depending on configuration).
3. **Bounding Boxes (Detection)**: Draw a tight box around *both* the bright return and its immediate shadow, as the shadow encodes height information.
4. **Polygons (Segmentation)**: Draw a tight polygon around the actual object (bright return), and if your taxonomy allows, a separate class/mask for the acoustic shadow. If using a single class, wrap the bright return closely.
5. **Ignore the Water Column**: Do not annotate noise in the nadir/water column region unless it's explicitly a mid-water target.

## 4. Where to Place the Data

After annotating:
- Place all your images in: `data/sonar/images/`
- Place your label files in: `data/sonar/labels/` (if using YOLO format) OR
- Keep your COCO/VOC files alongside the images or in `data/sonar/` and the converter script will handle them.

> [!WARNING]
> Do NOT use tools that apply JPEG compression or severe image degradation to the exported files. Sonar data is highly sensitive to compression artifacts.

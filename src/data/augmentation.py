import albumentations as A
from src.utils.config import Config

class SonarAugmentor:
    def __init__(self, config: Config):
        aug_cfg = config.augmentation
        
        # We exclusively use sonar-valid transformations.
        # Strict constraints: rotation <= 15, no color jitter, no perspective.
        
        transforms = []
        
        if aug_cfg.flipud > 0:
            transforms.append(A.VerticalFlip(p=aug_cfg.flipud))
            
        if aug_cfg.fliplr > 0:
            transforms.append(A.HorizontalFlip(p=aug_cfg.fliplr))
            
        transforms.append(
            A.Affine(
                scale=(1.0 - aug_cfg.scale, 1.0 + aug_cfg.scale),
                translate_percent=aug_cfg.translate,
                rotate=(-aug_cfg.degrees, aug_cfg.degrees),
                p=0.5
            )
        )
        
        if aug_cfg.noise_injection:
            transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.2))
            
        if aug_cfg.random_erasing > 0:
            transforms.append(A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=aug_cfg.random_erasing))
            
        # Optional brightness/contrast for acoustic returns
        transforms.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3))
        
        self.transform = A.Compose(
            transforms,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']),
            # Add mask_params if we do segmentation directly via albumentations
        )

    def __call__(self, image, bboxes, class_labels, masks=None):
        if masks is not None:
            return self.transform(image=image, bboxes=bboxes, class_labels=class_labels, masks=masks)
        return self.transform(image=image, bboxes=bboxes, class_labels=class_labels)

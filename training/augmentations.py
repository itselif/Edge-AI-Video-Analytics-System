# training/augmentations.py

import albumentations as A


def get_train_augmentations(img_size: int):
    """
    Strong Albumentations pipeline for YOLO training.
    img_size: target image size (e.g. 640)
    """
    return A.Compose(
        [
            
            A.RandomResizedCrop(
                height=img_size,
                width=img_size,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
                p=0.5,
            ),

            # Motion / Median / Gaussian blur
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=(3, 7), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.3,
            ),

            # Color jitter: brightness/contrast vs. hue/saturation/value
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3,
                        contrast_limit=0.3,
                        p=1.0,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0,
                    ),
                ],
                p=0.7,
            ),

            # Noise (Albumentations yeni API: std_range, mean, per_channel vs.)
            A.OneOf(
                [
                    A.GaussNoise(
                        std_range=(0.2, 0.44),
                        mean=0.0,
                        per_channel=True,
                        p=1.0,
                    ),
                    A.ISONoise(
                        color_shift=(0.01, 0.05),
                        intensity=(0.1, 0.5),
                        p=1.0,
                    ),
                ],
                p=0.2,
            ),

            # Cutout / CoarseDropout (yeni API: hole_*_range, num_holes_range, fill)
            A.CoarseDropout(
                hole_height_range=(0.1, 0.2),   # img_size'in %10–20'si
                hole_width_range=(0.1, 0.2),
                num_holes_range=(1, 2),
                fill=0.0,
                p=0.4,
            ),

            # Horizontal flip
            A.HorizontalFlip(p=0.5),
        ],
        # YOLO formatında bbox + class label
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )

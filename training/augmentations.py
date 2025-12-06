# training/augmentations.py

import albumentations as A


def get_train_augmentations(img_size: int = 640):
    """
    Strong Albumentations pipeline for YOLO11 training.

    Covers:
    - RandomResizedCrop (RandomCrop + scale)
    - MotionBlur / GaussianBlur / MedianBlur
    - Color jitter (Brightness/Contrast + Hue/Sat/Value)
    - Noise (GaussNoise / ISONoise)
    - CutOut (CoarseDropout)
    """
    size = (img_size, img_size)

    return [
        # 1) Random crop / scale
        A.RandomResizedCrop(
            size=size,                 # <--- KRİTİK: size veriyoruz
            scale=(0.8, 1.0),
            ratio=(0.75, 1.33),
            p=0.5,
        ),

        # 2) Blur / MotionBlur
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            ],
            p=0.3,
        ),

        # 3) Color jitter
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

        # 4) Noise
        A.OneOf(
            [
                # Albumentations yeni API: std_range kullanıyoruz
                A.GaussNoise(std_range=(0.2, 0.44), p=1.0),
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=1.0,
                ),
            ],
            p=0.2,
        ),

        # 5) CutOut (CoarseDropout) - yeni API
        A.CoarseDropout(
            hole_height_range=(0.1, 0.2),
            hole_width_range=(0.1, 0.2),
            num_holes_range=(1, 2),
            fill=0.0,
            p=0.4,
        ),
    ]

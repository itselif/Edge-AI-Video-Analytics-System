# training/augmentations.py

import albumentations as A


def get_train_augmentations(img_size: int = 640):

    return [
        # 1) Random crop / scale
        A.RandomResizedCrop(
            height=img_size,
            width=img_size,
            scale=(0.8, 1.0), 
            ratio=(0.75, 1.33),
            p=0.5,
        ),

        # 2) Blur / MotionBlur
        A.OneOf(
            [
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=7, p=1.0),
            ],
            p=0.3,
        ),

        # 3) Color augments
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
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=1.0,
                ),
            ],
            p=0.2,
        ),

        # 5) CutOut
        A.CoarseDropout(
            max_holes=8,
            max_height=0.2 * img_size,
            max_width=0.2 * img_size,
            min_holes=1,
            min_height=0.05 * img_size,
            min_width=0.05 * img_size,
            fill_value=0,
            p=0.4,
        ),
    ]

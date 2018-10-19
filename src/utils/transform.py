from albumentations import (
    HorizontalFlip,  ShiftScaleRotate, RGBShift, CenterCrop,
    RandomSizedCrop, SmallestMaxSize, RandomCrop,
    ShiftScaleRotate, HueSaturationValue, Normalize,
    RandomContrast, RandomBrightness, Flip, OneOf, Compose
)

train_transform = Compose([
    SmallestMaxSize(max_size=256),
    ShiftScaleRotate(scale_limit=(0.5, 1), rotate_limit=5),
    RandomCrop(224, 224, p=1.0),
    HorizontalFlip(0.5),
    OneOf([
        HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=1),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5)
    ], p=0.2),
    OneOf([
        RandomBrightness(limit=0.2, p=1.),
        RandomContrast(limit=0.2, p=1.),
    ], p=0.2),
    Normalize(p=1.),
], p = 1.0)


valid_transform = Compose([
    SmallestMaxSize(max_size=256),
    CenterCrop(224, 224, p=1.0),
    Normalize(p=1.),
], p = 1.0)
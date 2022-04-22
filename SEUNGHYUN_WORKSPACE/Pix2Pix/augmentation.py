import albumentations as A
from albumentations.pytorch import ToTensorV2

transform_both = A.Compose(
    [A.Resize(width=256, height=256),], 
    additional_targets={"image0": "image"}, # X, y 모두 적용되는 augmentations
)

transform_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_target = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
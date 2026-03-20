from __future__ import annotations

from torchvision import transforms

from .config import AugmentationConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, augmentation: AugmentationConfig, training: bool) -> transforms.Compose:
    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(augmentation.resize_scale_min, 1.0),
                    ratio=(0.85, 1.15),
                ),
                transforms.ColorJitter(
                    brightness=augmentation.color_jitter,
                    contrast=augmentation.color_jitter,
                    saturation=augmentation.color_jitter,
                    hue=min(0.05, augmentation.color_jitter / 4),
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3)],
                    p=augmentation.blur_prob,
                ),
                transforms.RandomGrayscale(p=augmentation.grayscale_prob),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=augmentation.random_erasing_prob, scale=(0.01, 0.08)),
            ]
        )
    resize_size = int(image_size * 1.14)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


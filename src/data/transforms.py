"""PIL preprocessing matching A's baseline, with opt-in training augmentation."""

from dataclasses import dataclass, asdict
import math

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision.transforms import ColorJitter


@dataclass(frozen=True)
class TransformConfig:
    image_size: int = 224
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    augment: bool = False
    horizontal_flip: float = 0.5
    rotation_degrees: float = 5.0
    brightness: float = 0.1
    contrast: float = 0.1
    saturation: float = 0.1
    # Retain at least this fraction of EACH axis. 1 disables cropping.
    crop_min_fraction: float = 1.0

    def __post_init__(self):
        if type(self.image_size) is not int or self.image_size < 1:
            raise ValueError("image_size must be a positive integer")
        if type(self.augment) is not bool:
            raise ValueError("augment must be boolean")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("RGB normalization needs three means and standard deviations")
        if not all(math.isfinite(v) for v in (*self.mean, *self.std)) or min(self.std) <= 0:
            raise ValueError("Normalization must be finite with positive std")
        for name, low, high in (("horizontal_flip", 0, 1), ("rotation_degrees", 0, 15),
                                ("brightness", 0, .2), ("contrast", 0, .2),
                                ("saturation", 0, .2), ("crop_min_fraction", .95, 1)):
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}]")

    def as_dict(self):
        return asdict(self)


class ImageTransform:
    def __init__(self, config: TransformConfig, *, training: bool):
        if config.augment and not training:
            raise ValueError("Augmentation is allowed only for training")
        self.config = config
        self.mean = torch.tensor(config.mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(config.std, dtype=torch.float32).view(3, 1, 1)
        self.jitter = ColorJitter(config.brightness, config.contrast, config.saturation)

    def __call__(self, source):
        image = source.convert("RGB")
        cfg = self.config
        if cfg.augment:
            if torch.rand(()).item() < cfg.horizontal_flip:
                image = ImageOps.mirror(image)
            if cfg.rotation_degrees:
                angle = (2 * torch.rand(()).item() - 1) * cfg.rotation_degrees
                image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
            if cfg.crop_min_fraction < 1:
                width, height = image.size
                # No square-aspect fallback: rectangular images retain >=95% per axis.
                cut_x = int(width * (1 - cfg.crop_min_fraction) * torch.rand(()).item())
                cut_y = int(height * (1 - cfg.crop_min_fraction) * torch.rand(()).item())
                left = int(torch.randint(cut_x + 1, ()).item())
                top = int(torch.randint(cut_y + 1, ()).item())
                image = image.crop((left, top, width - cut_x + left, height - cut_y + top))
            image = self.jitter(image)
        # Same PIL bilinear resize and arithmetic as A PR #2, no-augmentation path.
        image = image.resize((cfg.image_size, cfg.image_size), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return (tensor - self.mean) / self.std

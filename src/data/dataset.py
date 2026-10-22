"""Read-only Dataset retaining official sample identities and split boundaries."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from ..glaucoma.data import load_records
from .transforms import ImageTransform, TransformConfig


class GlaucomaDataset(Dataset):
    def __init__(self, root, split, *, config=None, indices=None):
        self.root, self.split = Path(root), split
        records = load_records(self.root, split)
        if indices is not None:
            indices = list(indices)
            if (not indices or len(set(indices)) != len(indices)
                    or any(type(i) is not int or i < 0 or i >= len(records) for i in indices)):
                raise ValueError("indices must be nonempty, unique integer positions in this split")
            records = [records[i] for i in indices]
        self.records = tuple(records)
        self.config = config or TransformConfig()
        self.transform = ImageTransform(self.config, training=split == "train")

    @property
    def labels(self):
        if self.split == "test":
            raise ValueError("Test split has no training/validation labels")
        return tuple(row.label for row in self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row.path) as image:
            tensor = self.transform(image)
        result = {"image": tensor, "image_name": row.image_name}
        if self.split != "test":
            result["label"] = row.label
        return result

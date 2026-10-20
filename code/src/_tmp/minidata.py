"""临时最小数据读取（A 自测用，对接 B 后删除本目录）。

仅实现手册 1.3 真实数据小批量自测所需的最小逻辑：
- 解析 CSV（Image_name, Age, True_Label）；
- 处理 CSV 中 .jpg 与实际文件 .JPG 的大小写映射；
- 返回 image / label / image_name（与 1.4 接口约定一致）。
正式 Dataset/增强/采样由 B 交付后替换。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    """最小 Dataset：resize + normalize，无增强。"""

    def __init__(self, data_root: str, csv_name: str, image_size: int = 224,
                 norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
        root = Path(data_root)
        self.df = pd.read_csv(root / csv_name)
        self.img_dir = root / Path(csv_name).stem  # train.csv -> train/
        self.image_size = image_size
        self.mean = torch.tensor(norm_mean).view(3, 1, 1)
        self.std = torch.tensor(norm_std).view(3, 1, 1)
        # Preserve CSV identifiers; use the directory spelling only for opening images.
        self.names = self.df["Image_name"].tolist()
        index = {}
        for path in self.img_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                key = path.name.casefold()
                if key in index:
                    raise ValueError(f"Ambiguous image names: {index[key].name}, {path.name}")
                index[key] = path
        self.paths = []
        seen = set()
        for name in self.names:
            if not isinstance(name, str) or not name or name != name.strip() or "/" in name or "\\" in name:
                raise ValueError(f"Invalid image name: {name!r}")
            key = name.casefold()
            if key in seen:
                raise ValueError(f"Duplicate image identifier: {name}")
            seen.add(key)
            if key not in index:
                raise ValueError(f"Image not found: {name}")
            self.paths.append(index[key])
        if not self.df["True_Label"].isin([0, 1]).all():
            raise ValueError("Labels must be 0 or 1")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.names[idx]
        with Image.open(self.paths[idx]) as source:
            img = source.convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        x = (x - self.mean) / self.std
        label = int(self.df.iloc[idx]["True_Label"])
        return {"image": x, "label": label, "image_name": name}


def make_subset(dataset: CSVDataset, n_total: int, seed: int = 42):
    """分层抽小子集（含两类），用于自测与快速评估。"""
    rng = np.random.RandomState(seed)
    labels = dataset.df["True_Label"].to_numpy()
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    n_pos = max(2, n_total // 10)  # 保证两类都在
    take = np.concatenate([
        rng.choice(idx_pos, size=min(n_pos, len(idx_pos)), replace=False),
        rng.choice(idx_neg, size=n_total - min(n_pos, len(idx_pos)), replace=False),
    ])
    rng.shuffle(take)
    return torch.utils.data.Subset(dataset, take.tolist())

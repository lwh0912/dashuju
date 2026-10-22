"""Build B's loaders/loss for A's config without duplicating a training loop."""

from dataclasses import dataclass
import hashlib
import json

import torch
from torch.utils.data import DataLoader

from .dataset import GlaucomaDataset
from .sampling import build_training_policy, seed_worker
from .transforms import TransformConfig


@dataclass
class TrainingBundle:
    train_loader: DataLoader
    valid_loader: DataLoader
    criterion: object
    metadata: dict


def identity_digest(dataset):
    rows = [(r.image_name, r.label) for r in dataset.records]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def build_training_bundle(cfg, *, train_indices=None, valid_indices=None):
    if cfg.get("train_csv", "train.csv") != "train.csv" or cfg.get("valid_csv", "valid.csv") != "valid.csv":
        raise ValueError("Use the official split CSVs; pass training subset positions for fold experiments")
    if cfg.get("resize_mode", "resize") != "resize":
        raise ValueError("Only the baseline PIL bilinear resize is supported")
    for key in ("batch_size", "eval_batch_size"):
        value = cfg.get(key, cfg["batch_size"])
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    workers = cfg.get("num_workers", 0)
    if type(workers) is not int or workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    norm = cfg["normalize"]
    common = dict(image_size=cfg["image_size"], mean=tuple(norm["mean"]), std=tuple(norm["std"]))
    augmentation = cfg.get("augmentation", {})
    if set(augmentation) - {"augment", "horizontal_flip", "rotation_degrees", "brightness", "contrast", "saturation", "crop_min_fraction"}:
        raise ValueError("Unknown augmentation option")
    train_cfg = TransformConfig(**common, **augmentation)
    valid_cfg = TransformConfig(**common)
    train = GlaucomaDataset(cfg["data_root"], "train", config=train_cfg, indices=train_indices)
    valid = GlaucomaDataset(cfg["data_root"], "valid", config=valid_cfg, indices=valid_indices)
    # Legacy A config defaults to weighted BCE; explicit strategies cannot silently stack.
    if "pos_weight" in cfg and type(cfg["pos_weight"]) is not bool:
        raise ValueError("Legacy pos_weight must be boolean")
    strategy = cfg.get("data_strategy", "pos_weight" if cfg.get("pos_weight", True) else "bce")
    if "data_strategy" in cfg and "pos_weight" in cfg and cfg["pos_weight"] != (strategy == "pos_weight"):
        raise ValueError("Conflicting data_strategy and legacy pos_weight; remove pos_weight")
    seed = cfg["seed"]
    policy = build_training_policy(train, strategy, seed=seed + 1, focal_gamma=cfg.get("focal_gamma", 2.0),
                                   negative_to_positive=cfg.get("negative_to_positive", 10.0))
    drop_last = cfg.get("drop_last", True)
    if type(drop_last) is not bool:
        raise ValueError("drop_last must be boolean")
    train_loader = DataLoader(train, batch_size=cfg["batch_size"],
        shuffle=policy.sampler is None, sampler=policy.sampler,
        num_workers=workers, drop_last=drop_last, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))
    valid_loader = DataLoader(valid, batch_size=cfg.get("eval_batch_size", cfg["batch_size"]),
        shuffle=False, num_workers=workers, drop_last=False, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed + 2))
    if len(train_loader) == 0:
        raise ValueError("No training batches; decrease batch size or disable drop_last")
    metadata = {**policy.metadata, "loader_seed": seed, "drop_last": drop_last,
        "optimizer_steps_per_epoch": len(train_loader), "train_transform": train_cfg.as_dict(),
        "valid_transform": valid_cfg.as_dict(), "train_identity_sha256": identity_digest(train),
        "valid_identity_sha256": identity_digest(valid), "valid_samples": len(valid),
        "train_subset": train_indices is not None, "valid_subset": valid_indices is not None,
        "preprocessing": "PIL RGB; optional train augmentation; PIL bilinear resize; float32 /255; normalize"}
    return TrainingBundle(train_loader, valid_loader, policy.criterion, metadata)

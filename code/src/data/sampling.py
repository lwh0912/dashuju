"""Mutually exclusive imbalance policies fitted to the supplied training subset."""

from dataclasses import dataclass
import math
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler, WeightedRandomSampler

from ..losses import BinaryFocalLoss


def seed_worker(worker_id):
    """DataLoader seeds torch; synchronize Python/NumPy inside each worker."""
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class NegativeUndersampler(Sampler):
    def __init__(self, labels, generator, negative_to_positive=10.0):
        if not math.isfinite(negative_to_positive) or negative_to_positive < 1:
            raise ValueError("negative_to_positive must be finite and at least 1")
        self.groups = [torch.tensor([i for i, label in enumerate(labels) if label == c])
                       for c in (0, 1)]
        self.counts = [min(len(self.groups[0]), math.ceil(len(self.groups[1]) * negative_to_positive)),
                       len(self.groups[1])]
        self.generator = generator

    def __iter__(self):
        selected = torch.cat([group[torch.randperm(len(group), generator=self.generator)[:count]]
                              for group, count in zip(self.groups, self.counts)])
        return iter(selected[torch.randperm(len(selected), generator=self.generator)].tolist())

    def __len__(self):
        return sum(self.counts)


@dataclass
class TrainingPolicy:
    criterion: nn.Module
    sampler: object
    metadata: dict


def build_training_policy(dataset, strategy="bce", *, seed=42, focal_gamma=2.0,
                          negative_to_positive=10.0):
    if dataset.split != "train":
        raise ValueError("Loss weights and samplers must be fitted only to the training split")
    labels = dataset.labels
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        raise ValueError("Training policy requires both classes in the selected training subset")
    if strategy not in {"bce", "pos_weight", "oversample", "undersample", "focal"}:
        raise ValueError(f"Unknown single-factor strategy: {strategy}")
    generator = torch.Generator().manual_seed(seed)
    criterion, sampler = nn.BCEWithLogitsLoss(), None
    if strategy == "pos_weight":
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
    elif strategy == "oversample":
        sampler = WeightedRandomSampler([1 / (pos if y else neg) for y in labels],
                                         num_samples=len(labels), replacement=True, generator=generator)
    elif strategy == "undersample":
        sampler = NegativeUndersampler(labels, generator, negative_to_positive)
    elif strategy == "focal":
        criterion = BinaryFocalLoss(focal_gamma)
    metadata = {"strategy": strategy, "seed": seed, "train_positive": pos, "train_negative": neg,
                "pos_weight": neg / pos if strategy == "pos_weight" else None,
                "focal_gamma": focal_gamma if strategy == "focal" else None,
                "focal_alpha": None,
                "negative_to_positive": negative_to_positive if strategy == "undersample" else None,
                "samples_per_epoch_before_drop_last": len(sampler) if sampler is not None else len(labels),
                "sampling_with_replacement": strategy == "oversample"}
    return TrainingPolicy(criterion, sampler, metadata)

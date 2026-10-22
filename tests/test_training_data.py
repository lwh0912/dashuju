import csv
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import GlaucomaDataset, ImageTransform, TransformConfig, build_training_policy
from src.data.pipeline import build_training_bundle
from src.losses import BinaryFocalLoss
from src.glaucoma.data import Record
from tools.analyze_errors import analyze


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pixels = np.random.default_rng(8).integers(0, 256, (31, 45, 3), dtype=np.uint8)
        for split in ("train", "valid", "test"):
            (self.root / split).mkdir()
            with (self.root / f"{split}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Image_name", "Age"] + ([] if split == "test" else ["True_Label"]))
                for i in range(8):
                    Image.fromarray(self.pixels).save(self.root / split / f"sample{i}.PNG")
                    writer.writerow([f"sample{i}.png", ""] + ([] if split == "test" else [int(i >= 6)]))
        self.config = TransformConfig(image_size=16)

    def dataset(self, split="train", **kwargs):
        return GlaucomaDataset(self.root, split, config=self.config, **kwargs)

    def cfg(self, **kwargs):
        return dict(data_root=str(self.root), image_size=16, seed=42, batch_size=2,
                    normalize={"mean": self.config.mean, "std": self.config.std}, **kwargs)

    def test_baseline_pixels_identity_and_no_source_change(self):
        path = self.root / "train/sample0.PNG"
        before = path.read_bytes()
        row = self.dataset()[0]
        source = Image.fromarray(self.pixels).resize((16, 16), Image.Resampling.BILINEAR)
        expected = torch.from_numpy(np.array(source)).permute(2, 0, 1).float() / 255
        expected = (expected - torch.tensor(self.config.mean)[:, None, None]) / torch.tensor(self.config.std)[:, None, None]
        torch.testing.assert_close(row["image"], expected, rtol=0, atol=0)
        self.assertEqual(row["image_name"], "sample0.png")
        self.assertEqual(path.read_bytes(), before)

    def test_validation_and_test_are_deterministic_and_no_fake_labels(self):
        for split in ("valid", "test"):
            ds = self.dataset(split)
            torch.testing.assert_close(ds[0]["image"], ds[0]["image"], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                GlaucomaDataset(self.root, split, config=replace(self.config, augment=True))
            with self.assertRaises(ValueError):
                build_training_policy(ds)
        batch = next(iter(DataLoader(self.dataset("test"), batch_size=3)))
        self.assertNotIn("label", batch)
        self.assertEqual(batch["image"].shape, (3, 3, 16, 16))

    def test_augmentation_seed_and_input_immutability(self):
        transform = ImageTransform(replace(self.config, augment=True, crop_min_fraction=.95), training=True)
        image = Image.fromarray(self.pixels)
        before = image.tobytes()
        torch.manual_seed(21)
        first = transform(image)
        torch.manual_seed(21)
        second = transform(image)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(first).all())
        self.assertEqual(image.tobytes(), before)

    def test_subset_weights_do_not_use_full_training_or_validation(self):
        ds = self.dataset(indices=[0, 6, 7])
        policy = build_training_policy(ds, "pos_weight")
        self.assertEqual(policy.criterion.pos_weight.item(), .5)
        self.assertEqual(policy.metadata["train_negative"], 1)
        with self.assertRaises(ValueError):
            self.dataset(indices=[0, 0])
        with self.assertRaises(ValueError):
            build_training_policy(self.dataset(indices=[0, 1]))

    def test_sampling_exposure_repeatability_and_no_weight_stacking(self):
        ds = self.dataset()
        p1 = build_training_policy(ds, "oversample", seed=5)
        p2 = build_training_policy(ds, "oversample", seed=5)
        self.assertEqual(list(p1.sampler), list(p2.sampler))
        self.assertEqual(len(p1.sampler), len(ds))
        self.assertIsNone(p1.criterion.pos_weight)
        positive_mass = sum(w for w, y in zip(p1.sampler.weights, ds.labels) if y)
        negative_mass = sum(w for w, y in zip(p1.sampler.weights, ds.labels) if not y)
        torch.testing.assert_close(positive_mass, negative_mass)
        under = list(build_training_policy(ds, "undersample", seed=5, negative_to_positive=1).sampler)
        self.assertEqual(len(under), 4)
        self.assertEqual(len(set(under)), 4)
        self.assertEqual(sum(ds.labels[i] for i in under), 2)
        self.assertEqual(len(build_training_policy(ds, "undersample").sampler), len(ds))

    def test_focal_equivalence_extremes_and_gradients(self):
        logits = torch.tensor([-1000., 1000., 0., 3.], requires_grad=True)
        labels = torch.tensor([1., 0., 1., 1.])
        torch.testing.assert_close(BinaryFocalLoss(0)(logits, labels), torch.nn.BCEWithLogitsLoss()(logits, labels))
        loss = BinaryFocalLoss()(logits, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        saturated = torch.tensor([-1000., 1000.], requires_grad=True)
        BinaryFocalLoss(.5)(saturated, torch.tensor([0., 1.])).backward()
        self.assertTrue(torch.isfinite(saturated.grad).all())
        with self.assertRaises(ValueError):
            BinaryFocalLoss(float("nan"))

    def test_bundle_conflicts_and_training_budget(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            build_training_bundle(self.cfg(data_strategy="oversample", pos_weight=True))
        weighted = build_training_bundle(self.cfg())
        self.assertEqual(weighted.metadata["strategy"], "pos_weight")
        self.assertEqual(weighted.criterion.pos_weight.item(), 3.)
        for strategy in ("bce", "pos_weight", "oversample", "focal"):
            bundle = build_training_bundle(self.cfg(data_strategy=strategy, augmentation={"augment": True}))
            self.assertEqual(bundle.metadata["optimizer_steps_per_epoch"], 4)
            self.assertFalse(bundle.valid_loader.dataset.config.augment)
        self.assertEqual(len(build_training_bundle(self.cfg(data_strategy="undersample", negative_to_positive=1)).train_loader), 2)

    def test_worker_repeatability(self):
        cfg = self.cfg(data_strategy="oversample", num_workers=2, augmentation={"augment": True})
        first = list(build_training_bundle(cfg).train_loader)
        second = list(build_training_bundle(cfg).train_loader)
        for a, b in zip(first, second):
            self.assertEqual(a["image_name"], b["image_name"])
            torch.testing.assert_close(a["image"], b["image"], rtol=0, atol=0)


class ErrorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.records = [Record(f"image{i}", Path("unused"), None, y) for i, y in enumerate([0, 1, 0, 1])]
        self.rows = [dict(Image_name=r.image_name, True_Label=str(r.label), Probability=str(p))
                     for r, p in zip(self.records, [.8, .2, .1, .9])]

    def test_threshold_changes_counts_not_probability_metrics(self):
        a, _ = analyze(self.records, self.rows)
        b, _ = analyze(self.records, self.rows, threshold=.9)
        self.assertEqual((a["TP"], a["FN"], a["FP"], a["TN"]), (1, 1, 1, 1))
        self.assertEqual(b["FP"], 0)
        self.assertEqual(a["auroc"], b["auroc"])
        self.assertEqual(a["auprc"], b["auprc"])

    def test_incomplete_duplicates_wrong_labels_nonfinite_fail(self):
        bad_cases = [self.rows[:2], self.rows + self.rows[:1],
                     [dict(self.rows[0], True_Label="1"), *self.rows[1:]],
                     [dict(self.rows[0], Probability="nan"), *self.rows[1:]]]
        for bad in bad_cases:
            with self.assertRaises(ValueError):
                analyze(self.records, bad)
        report, _ = analyze(self.records, self.rows[:2], allow_subset=True)
        self.assertFalse(report["full_coverage"])


if __name__ == "__main__":
    unittest.main()

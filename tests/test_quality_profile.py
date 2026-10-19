import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from profile_image_quality import load_table, summarize


class QualityProfileTests(unittest.TestCase):
    def make_fixture(self, root):
        features = root / "features"
        features.mkdir()
        rows = []
        for split in ("train", "valid", "test"):
            official = []
            for i in range(2):
                name = f"{split}_{i}.jpg"
                row = {"Image_name": name, "Age": 40 + i}
                if split != "test":
                    row["True_Label"] = i
                official.append(row)
                rows.append({"split": split, "image_name": name, "label": i if split != "test" else None,
                             "width": 400, "height": 300, "brightness_mean": 20 + i * 20,
                             "contrast_std": 10, "dark_fraction_below_10": .2})
            pd.DataFrame(official).to_csv(root / f"{split}.csv", index=False)
        pd.DataFrame(rows).to_csv(features / "features.csv", index=False)
        (features / "report.json").write_text(json.dumps({"status": "COMPLETE", "scope": "full", "errors": [],
                "success": 6, "dataset_total": 6, "dataset_counts": {s: 2 for s in ("train", "valid", "test")}}), encoding="utf-8")
        return features

    def test_reordered_csv_aligns_by_name_and_changed_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features = self.make_fixture(root)
            official = pd.read_csv(root / "train.csv").iloc[::-1]
            official.to_csv(root / "train.csv", index=False)
            table, _ = load_table(root, features)
            self.assertEqual(table.loc[table.image_name == "train_0.jpg", "age"].iloc[0], 40)
            official["True_Label"] = 0
            official.to_csv(root / "train.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Labels do not match"):
                load_table(root, features)

    def test_nonfinite_and_missing_sample_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features = self.make_fixture(root)
            table = pd.read_csv(features / "features.csv", dtype={"brightness_mean": float})
            table.loc[0, "brightness_mean"] = float("inf")
            table.to_csv(features / "features.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                load_table(root, features)
            table.loc[0, "brightness_mean"] = 20
            table.loc[0, "image_name"] = "wrong.jpg"
            table.to_csv(features / "features.csv", index=False)
            with self.assertRaisesRegex(ValueError, "sample identifiers"):
                load_table(root, features)

    def test_validation_extremes_do_not_change_training_thresholds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features = self.make_fixture(root)
            table, _ = load_table(root, features)
            before, _ = summarize(table)
            table.loc[table.split == "valid", "brightness_mean"] = 255
            after, splits = summarize(table)
            self.assertEqual(before, after)
            self.assertEqual(splits["valid"]["train_reference_tails"]["brightness_mean"]["above_train_p99"], 2)
            self.assertEqual(splits["train"]["train_reference_tails"]["contrast_std"],
                             {"below_train_p01": 0, "above_train_p99": 0})


if __name__ == "__main__":
    unittest.main()

import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from glaucoma.data import load_records


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "train").mkdir()
        (self.root / "train" / "train_00000.JPG").write_bytes(b"test placeholder")

    def write_csv(self, rows):
        with (self.root / "train.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Image_name", "Age", "True_Label"])
            writer.writerows(rows)

    def test_maps_case_preserves_identifier_and_missing_age(self):
        self.write_csv([["train_00000.jpg", "", "1"]])
        row = load_records(self.root, "train")[0]
        self.assertEqual(row.image_name, "train_00000.jpg")
        self.assertEqual(row.path.name, "train_00000.JPG")
        self.assertIsNone(row.age)
        self.assertEqual(row.label, 1)

    def test_rejects_invalid_label(self):
        self.write_csv([["train_00000.jpg", "53", "2"]])
        with self.assertRaisesRegex(ValueError, "label must"):
            load_records(self.root, "train")

    def test_rejects_missing_image(self):
        self.write_csv([["missing.jpg", "53", "0"]])
        with self.assertRaisesRegex(ValueError, "image missing"):
            load_records(self.root, "train")

    def test_rejects_duplicate_identifier(self):
        self.write_csv([["train_00000.jpg", "53", "0"], ["train_00000.JPG", "53", "0"]])
        with self.assertRaisesRegex(ValueError, "duplicate image"):
            load_records(self.root, "train")

    def test_test_split_needs_no_labels(self):
        (self.root / "test").mkdir()
        (self.root / "test" / "test_00000.JPG").write_bytes(b"test placeholder")
        (self.root / "test.csv").write_text("Image_name,Age\ntest_00000.jpg,44\n", encoding="utf-8")
        row = load_records(self.root, "test")[0]
        self.assertIsNone(row.label)
        self.assertEqual(row.age, 44)


if __name__ == "__main__":
    unittest.main()

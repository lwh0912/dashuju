import contextlib
import csv
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_image_features import exclusive_run, extract_one, run


class FullFeatureTests(unittest.TestCase):
    def fixture(self, root):
        for split in ("train", "valid", "test"):
            (root / split).mkdir(parents=True)
            Image.new("RGB", (40, 20), (100, 100, 100)).save(root / split / "first.JPG")
            with (root / f"{split}.csv").open("w", encoding="utf-8", newline="") as h:
                writer = csv.writer(h)
                writer.writerow(["Image_name", "Age"] + ([] if split == "test" else ["True_Label"]))
                writer.writerow(["first.jpg", 50] + ([] if split == "test" else [1]))

    def quiet_run(self, root, output, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(root, output, workers=2, checkpoint_every=1, **kwargs)

    def test_resume_and_file_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "data"
            self.fixture(root)
            output = base / "results"
            first = self.quiet_run(root, output)
            self.assertEqual(first["status"], "COMPLETE")
            self.assertEqual(first["computed"], 3)
            second = self.quiet_run(root, output)
            self.assertEqual(second["reused"], 3)
            path = root / "train/first.JPG"
            old = path.stat().st_mtime_ns
            Image.new("RGB", (40, 20), (200, 200, 200)).save(path)
            os.utime(path, ns=(old + 10_000_000, old + 10_000_000))
            third = self.quiet_run(root, output)
            self.assertEqual(third["computed"], 1)
            self.assertEqual(third["reused"], 2)
            with (output / "features.csv").open(encoding="utf-8-sig", newline="") as h:
                rows = list(csv.DictReader(h))
            self.assertEqual(len(rows), 3)
            self.assertAlmostEqual(float(rows[0]["brightness_mean"]), 200)

    def test_sample_not_full_and_corrupt_image_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "data"
            self.fixture(root)
            output = base / "results"
            path = root / "test/first.JPG"
            path.write_bytes(b"not an image")
            first = self.quiet_run(root, output, limit_per_split=1)
            self.assertEqual(first["status"], "INCOMPLETE")
            self.assertEqual(len(first["errors"]), 1)
            Image.new("RGB", (40, 20), (100, 100, 100)).save(path)
            second = self.quiet_run(root, output, limit_per_split=1)
            self.assertEqual(second["status"], "COMPLETE_SAMPLE")
            self.assertEqual(second["reused"], 2)
            self.assertEqual(second["computed"], 1)

    def test_excludes_simultaneous_output_writers(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with exclusive_run(output):
                with self.assertRaises(RuntimeError):
                    with exclusive_run(output):
                        pass

    def test_interrupt_preserves_completed_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "data"
            self.fixture(root)
            output = base / "results"

            def interrupted(items, workers):
                yield extract_one(items[0])
                raise KeyboardInterrupt()

            with patch("extract_image_features.bounded_results", interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    self.quiet_run(root, output)
            report = self.quiet_run(root, output)
            self.assertEqual(report["reused"], 1)
            self.assertEqual(report["computed"], 2)
            self.assertEqual(report["status"], "COMPLETE")


if __name__ == "__main__":
    unittest.main()

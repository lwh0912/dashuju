"""Exercise exact duplicates, cross-split overlap and cache invalidation."""

import contextlib
import csv
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_duplicates import run_audit


class DuplicateTests(unittest.TestCase):
    def fixture(self, root):
        # train/one and train/two and valid/same are byte-identical.
        # test/different has the same size but different bytes.
        data = {"train": [("one.JPG", b"same", 0), ("two.JPG", b"same", 0), ("unique.JPG", b"longer!", 1)],
                "valid": [("same.JPG", b"same", 1)], "test": [("different.JPG", b"else", None)]}
        for split, rows in data.items():
            (root / split).mkdir()
            with (root / f"{split}.csv").open("w", newline="", encoding="utf-8") as h:
                writer = csv.writer(h)
                writer.writerow(["Image_name", "Age"] + ([] if split == "test" else ["True_Label"]))
                for name, content, label in rows:
                    (root / split / name).write_bytes(content)
                    writer.writerow([name.lower(), 50] + ([] if label is None else [label]))

    def run_silently(self, root, output):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_audit(root, output, workers=2)

    def test_exact_cross_split_conflict_and_unique_size(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "data"
            root.mkdir()
            self.fixture(root)
            report = self.run_silently(root, base / "report")
            self.assertEqual(report["status"], "COMPLETE")
            self.assertEqual(report["total_files"], 5)
            self.assertEqual(report["unique_size_files"], 1)
            self.assertEqual(report["candidate_files"], 4)
            self.assertEqual(report["duplicate_groups"], 1)
            self.assertEqual(report["duplicate_files"], 3)
            self.assertEqual(report["within_split_groups"]["train"], 1)
            self.assertEqual(report["cross_split_pair_groups"]["train-valid"], 1)
            self.assertEqual(report["cross_split_pair_groups"]["train-test"], 0)
            self.assertEqual(report["label_conflict_groups"], 1)
            self.assertEqual((root / "train/one.JPG").read_bytes(), b"same")

    def test_resume_and_changed_file_rehashed(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "data"
            root.mkdir()
            self.fixture(root)
            output = base / "report"
            self.run_silently(root, output)
            report = self.run_silently(root, output)
            self.assertEqual(report["hashes_reused"], 4)
            path = root / "valid/same.JPG"
            old = path.stat().st_mtime_ns
            path.write_bytes(b"else")
            os.utime(path, ns=(old + 10_000_000, old + 10_000_000))
            report = self.run_silently(root, output)
            self.assertEqual(report["hashes_reused"], 3)
            self.assertEqual(report["hashes_computed"], 1)
            self.assertEqual(report["duplicate_groups"], 2)
            self.assertEqual(report["cross_split_pair_groups"]["train-valid"], 0)
            self.assertEqual(report["cross_split_pair_groups"]["valid-test"], 1)
            self.assertEqual(report["label_conflict_groups"], 0)


if __name__ == "__main__":
    unittest.main()

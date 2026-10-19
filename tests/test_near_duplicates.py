import random
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from find_near_duplicates import run, search


def row(value, label="0"):
    return {"dhash64": f"{value:016x}", "label": label}


class NearDuplicateTests(unittest.TestCase):
    def test_index_matches_bruteforce_including_partition_boundaries(self):
        rng = random.Random(2026)
        left = [row(rng.getrandbits(64)) for _ in range(80)]
        right = [row(rng.getrandbits(64)) for _ in range(40)]
        for i in range(8):
            value = int(left[i]["dhash64"], 16)
            right.append(row(value ^ sum(1 << bit for bit in [0, 13, 26, 39][:i % 5])))
        for radius in (0, 1, 2, 4, 6):
            expected = sorted(((int(a["dhash64"], 16) ^ int(b["dhash64"], 16)).bit_count(), i, j)
                              for i, a in enumerate(left) for j, b in enumerate(right)
                              if (int(a["dhash64"], 16) ^ int(b["dhash64"], 16)).bit_count() <= radius)
            result = search(left, right, radius, 10000)
            self.assertEqual(result["pairs"], expected)
            self.assertEqual(result["candidate_count"], len(expected))

    def test_repeated_hashes_count_all_pairs_and_keep_best_when_truncated(self):
        left = [row(0, "0"), row(0, "1"), row(1, "0")]
        right = [row(0, "1"), row(0, "1"), row(3, "0")]
        all_pairs = search(left, right, radius=2, max_pairs=100)
        capped = search(left, right, radius=2, max_pairs=3)
        self.assertEqual(capped["candidate_count"], 9)
        self.assertEqual(capped["pairs"], all_pairs["pairs"][:3])
        self.assertEqual(capped["distance_histogram"]["0"], 4)
        self.assertEqual(capped["different_known_label_candidates"], 5)

    def test_diverse_previews_and_human_review_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root, feature_dir, output = base / "data", base / "features", base / "review"
            feature_dir.mkdir()
            rows = []
            counts = {"train": 2, "valid": 2, "test": 1}
            for split, count in counts.items():
                (root / split).mkdir(parents=True)
                for i in range(count):
                    path = root / split / f"{i}.png"
                    Image.new("RGB", (40, 30), (50 + i * 50, 30, 20)).save(path)
                    stat = path.stat()
                    rows.append({"split": split, "image_name": path.name, "label": "" if split == "test" else "0",
                                 "relative_path": f"{split}/{path.name}", "size_bytes": stat.st_size,
                                 "mtime_ns": stat.st_mtime_ns, "dhash64": "0000000000000000"})
            with (feature_dir / "features.csv").open("w", encoding="utf-8", newline="") as h:
                writer = csv.DictWriter(h, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (feature_dir / "report.json").write_text(json.dumps({"status": "COMPLETE", "scope": "full", "errors": [],
                "success": 5, "dataset_counts": counts, "data_root": str(root), "feature_signature": "synthetic"}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                first = run(feature_dir, output, radius=0, max_pairs=10, preview_count=2)
            self.assertEqual(first["candidate_count"], 4)
            self.assertEqual(first["preview_errors"], [])
            with (output / "review.csv").open(encoding="utf-8-sig", newline="") as h:
                review = list(csv.DictReader(h))
            self.assertEqual(len({r["left_image"] for r in review}), 2)
            self.assertEqual(len({r["right_image"] for r in review}), 2)
            review[0]["review_status"] = "不同图"
            with (output / "review.csv").open("w", encoding="utf-8-sig", newline="") as h:
                writer = csv.DictWriter(h, fieldnames=list(review[0]))
                writer.writeheader()
                writer.writerows(review)
            original = (output / "review.csv").read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                second = run(feature_dir, output, radius=0, max_pairs=10, preview_count=2)
            self.assertEqual(second["review_file"], "review-template.csv")
            self.assertEqual((output / "review.csv").read_bytes(), original)
            with self.assertRaises(ValueError):
                run(feature_dir, output, radius=1, max_pairs=10, preview_count=2)


if __name__ == "__main__":
    unittest.main()

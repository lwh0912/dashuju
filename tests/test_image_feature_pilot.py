import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pilot_image_features import features, select_sample
from glaucoma.data import Record


class FeaturePilotTests(unittest.TestCase):
    def test_constant_image_stats_and_source_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "constant.png"
            Image.new("RGB", (400, 200), (100, 100, 100)).save(path)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            values, thumb = features(path)
            self.assertEqual((values["width"], values["height"]), (400, 200))
            self.assertEqual(thumb.size, (256, 128))
            self.assertAlmostEqual(values["brightness_mean"], 100)
            self.assertEqual(values["contrast_std"], 0)
            self.assertEqual(values["dhash64"], "0000000000000000")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_sampler_reproducible_and_both_classes_included(self):
        rows = [Record(f"{i}.jpg", Path(f"{i}.jpg"), None, int(i >= 95)) for i in range(100)]
        chosen = select_sample(rows, 10, 2026, True)
        self.assertEqual(chosen, select_sample(rows, 10, 2026, True))
        self.assertEqual(len(chosen), 10)
        self.assertEqual(sum(r.label == 1 for r in chosen), 5)
        self.assertEqual(len({r.image_name for r in chosen}), 10)

    def test_sampler_handles_small_minority_without_repeats(self):
        rows = [Record(f"{i}.jpg", Path(f"{i}.jpg"), None, int(i == 0)) for i in range(12)]
        chosen = select_sample(rows, 10, 3, True)
        self.assertEqual(len(chosen), 10)
        self.assertEqual(sum(r.label == 1 for r in chosen), 1)
        self.assertEqual(len({r.image_name for r in chosen}), 10)


if __name__ == "__main__":
    unittest.main()

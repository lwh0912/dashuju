import importlib.util
from pathlib import Path
import sys
import unittest

AVAILABLE = importlib.util.find_spec("cv2") is not None
if AVAILABLE:
    import cv2
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from verify_reviewed_pairs import compare
    from calibrate_pair_verification import priority_gate, GATE, variants
    from PIL import Image


@unittest.skipUnless(AVAILABLE, "OpenCV is available in the competition container")
class PairGeometryTests(unittest.TestCase):
    def test_priority_gate_rejects_degenerate_four_point_fit_and_missing_metrics(self):
        strong = {**GATE, "ransac_inliers": 100, "inlier_fraction": 1.0}
        self.assertTrue(priority_gate(strong))
        self.assertFalse(priority_gate({**strong, "ransac_inliers": 4}))
        self.assertFalse(priority_gate({**strong, "left_inlier_hull_fraction": 0.02}))
        self.assertFalse(priority_gate({**strong, "overlap_gray_correlation": None}))
        self.assertFalse(priority_gate({**strong, "overlap_gray_correlation": float("nan")}))

    def test_control_generation_preserves_source_pixels(self):
        image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (120, 160, 3), dtype=np.uint8))
        before = image.tobytes()
        generated = list(variants(image))
        self.assertEqual(len(generated), 7)
        self.assertEqual(len({name for name, _ in generated}), 7)
        self.assertEqual(before, image.tobytes())
        self.assertEqual(generated[-1][1].tobytes(), np.asarray(image)[:, ::-1].tobytes())

    def test_known_brightness_variant_has_many_correspondences(self):
        cv2.setNumThreads(2)
        rng = np.random.default_rng(2026)
        image = rng.integers(20, 230, (400, 400), dtype=np.uint8)
        variant = cv2.convertScaleAbs(image, alpha=0.8, beta=15)
        metrics, _ = compare(image, variant)
        self.assertGreater(metrics["mutual_ratio_matches"], 50)
        self.assertGreater(metrics["ransac_inliers"], 40)
        self.assertGreater(metrics["left_inlier_hull_fraction"], 0.1)
        self.assertGreater(metrics["overlap_gray_correlation"], 0.95)

    def test_unrelated_noise_not_confused_with_brightness_variant(self):
        rng = np.random.default_rng(2026)
        a = rng.integers(20, 230, (400, 400), dtype=np.uint8)
        b = rng.integers(20, 230, (400, 400), dtype=np.uint8)
        metrics, _ = compare(a, b)
        self.assertLess(metrics["mutual_ratio_matches"], 20)

    def test_blank_pair_is_insufficient_not_a_duplicate_verdict(self):
        metrics, _ = compare(np.full((100, 100), 50, dtype=np.uint8), np.full((100, 100), 60, dtype=np.uint8))
        self.assertEqual(metrics["geometry_status"], "INSUFFICIENT_MATCHES")
        self.assertIsNone(metrics["overlap_gray_correlation"])


if __name__ == "__main__":
    unittest.main()

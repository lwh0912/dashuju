import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from src import train
from src._tmp.minidata import CSVDataset


class BaselineTests(unittest.TestCase):
    def prepare(self, root):
        for split in ['train', 'valid']:
            directory = root / split
            directory.mkdir()
            with (root / f'{split}.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['Image_name', 'Age', 'True_Label'])
                for i in range(4):
                    name = f'{split}_{i}'
                    Image.fromarray(np.random.default_rng(i).integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(directory / f'{name}.JPG')
                    writer.writerow([f'{name}.jpg', 50, i % 2])

    def test_csv_identity_and_ambiguous_disk_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.prepare(root)
            dataset = CSVDataset(str(root), 'train.csv', 64)
            self.assertEqual(dataset[0]['image_name'], 'train_0.jpg')
            self.assertEqual(dataset.paths[0].name, 'train_0.JPG')
            (root / 'train/train_0.jpg').write_bytes((root / 'train/train_0.JPG').read_bytes())
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                CSVDataset(str(root), 'train.csv', 64)

    def test_last_checkpoint_and_best_predictions_match_their_epochs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.prepare(root)
            cfg = train.load_config('configs/baseline.yaml')
            cfg.update(data_root=str(root), output_dir=str(root / 'outputs'), epochs=2,
                       batch_size=2, eval_batch_size=2, num_workers=0, image_size=64)
            cfg['model']['pretrained'] = False
            original = train.evaluate
            call = 0

            def evaluate(*args, **kwargs):
                nonlocal call
                metrics = original(*args, **kwargs)
                call += 1
                # Force the best AUROC and best AP to occur at different epochs.
                metrics.update(auroc=0.9 if call == 1 else 0.8,
                               auprc=0.6 if call == 1 else 0.7)
                return metrics

            with patch.object(train, 'evaluate', side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
                train.train(cfg, train_batches=1)
            out = root / 'outputs'
            checkpoints = {name: torch.load(out / f'{name}.pt', map_location='cpu', weights_only=True)
                           for name in ['best', 'best_auprc', 'last']}
            self.assertEqual(checkpoints['best']['metrics']['epoch'], 1)
            self.assertEqual(checkpoints['best_auprc']['metrics']['epoch'], 2)
            self.assertEqual(checkpoints['last']['metrics'],
                             {'epoch': 2, 'valid_auroc': 0.8, 'valid_auprc': 0.7})
            self.assertTrue(any(not torch.equal(v, checkpoints['last']['state_dict'][k])
                                for k, v in checkpoints['best']['state_dict'].items()))
            for name in ['best', 'best_auprc']:
                with (out / f'{name}_valid_predictions.csv').open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([r['Image_name'] for r in rows], [f'valid_{i}.jpg' for i in range(4)])
                self.assertTrue(all(0 <= float(r['Probability']) <= 1 for r in rows))

    def test_unsupported_configuration_fails_before_training(self):
        cfg = train.load_config('configs/baseline.yaml')
        cfg['resize_mode'] = 'center_crop'
        with self.assertRaisesRegex(ValueError, 'Unsupported resize_mode'):
            train.train(cfg)


if __name__ == '__main__':
    unittest.main()

"""Bounded integration smoke with A's local code snapshot, NOT a model comparison."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.data import GlaucomaDataset, TransformConfig
from src.data.pipeline import build_training_bundle
from src.glaucoma.data import load_records


def subset(records, *, seed, negative=24, positive=8):
    rng = random.Random(seed)
    selected = []
    for label, count in ((0, negative), (1, positive)):
        group = [i for i, row in enumerate(records) if row.label == label]
        if len(group) < count:
            raise ValueError("Insufficient records for the fixed smoke subset")
        selected.extend(rng.sample(group, count))
    rng.shuffle(selected)
    return selected


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--a-code-root", type=Path, required=True, help="Trusted, reviewed local copy of A PR #2 code/")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (args.a_code_root / "src/train.py").is_file():
        raise ValueError("A code snapshot must contain src/train.py")
    source_files = [args.a_code_root / path for path in ("src/train.py", "src/models/__init__.py", "src/_tmp/minidata.py")]
    source_files += sorted((PROJECT / "src/data").glob("*.py")) + sorted((PROJECT / "src/losses").glob("*.py"))
    source_files += [Path(__file__).resolve(), PROJECT / "src/glaucoma/data.py"]
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("Use an empty output directory")
    # A's temporary nested code/src and B's root src coexist only for this check.
    # Production integration should place A's modules in the shared root src.
    import src
    src.__path__.append(str(args.a_code_root.resolve() / "src"))
    a = load_module("a_training_smoke", args.a_code_root / "src/train.py")
    old = load_module("a_dataset_smoke", args.a_code_root / "src/_tmp/minidata.py")
    seed = 42
    train_records = load_records(args.data_root, "train")
    valid_records = load_records(args.data_root, "valid")
    ti, vi = subset(train_records, seed=seed), subset(valid_records, seed=seed + 1)
    cfg = dict(seed=seed, data_root=str(args.data_root), train_csv="train.csv", valid_csv="valid.csv",
        image_size=224, resize_mode="resize", normalize={"mean": [.485, .456, .406], "std": [.229, .224, .225]},
        batch_size=8, eval_batch_size=8, num_workers=0, drop_last=True,
        model={"name": "resnet18", "pretrained": False},
        optimizer={"name": "adamw", "lr": .0003, "weight_decay": .0001})
    baseline = GlaucomaDataset(args.data_root, "train", indices=ti)
    old_dataset = old.CSVDataset(str(args.data_root), "train.csv", 224)
    for local_index, official_index in enumerate(ti[:4]):
        actual, expected = baseline[local_index], old_dataset[official_index]
        assert actual["image_name"] == expected["image_name"] and actual["label"] == expected["label"]
        torch.testing.assert_close(actual["image"], expected["image"], atol=0, rtol=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    results = []
    variants = [("B0_bce", "bce", False), ("B1_mild", "bce", True),
                ("B2_pos_weight", "pos_weight", False), ("B3_oversample", "oversample", False),
                ("B4_focal", "focal", False), ("B5_undersample", "undersample", False)]
    for name, strategy, augment in variants:
        a.set_seed(seed)
        run_cfg = {**cfg, "data_strategy": strategy, "augmentation": {"augment": augment}}
        bundle = build_training_bundle(run_cfg, train_indices=ti, valid_indices=vi)
        model = a.build_model("resnet18", pretrained=False).to(device)
        criterion = bundle.criterion.to(device)
        optimizer = a.build_optimizer(model, run_cfg)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        before = next(model.parameters()).detach().clone()
        start = time.perf_counter()
        loss = a.run_epoch(model, bundle.train_loader, criterion, optimizer, device, scaler,
                           False, max_batches=1, desc=name)
        if not torch.isfinite(torch.tensor(loss)) or torch.equal(before, next(model.parameters())):
            raise AssertionError("Nonfinite loss or parameters did not update")
        metrics = a.evaluate(model, bundle.valid_loader, device, False)
        run_dir = args.output_dir / name
        run_dir.mkdir()
        a.save_predictions(run_dir / "smoke_valid_predictions.csv", metrics)
        result = {"variant": name, "loss": loss, "smoke_auroc": metrics["auroc"],
            "smoke_auprc_ap": metrics["auprc"], "optimizer_steps_executed": 1,
            "train_draws_executed": 8, "elapsed_seconds": time.perf_counter() - start,
            "config": run_cfg, "data": bundle.metadata}
        results.append(result)
        if name == "B0_bce":
            checkpoint_path = run_dir / "smoke.pt"
            a.save_checkpoint(str(checkpoint_path), model, run_cfg, {"scope": "integration_smoke"})
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            restored = a.build_model("resnet18", pretrained=False).to(device)
            restored.load_state_dict(checkpoint["state_dict"])
            model.eval(); restored.eval()
            test_ds = GlaucomaDataset(args.data_root, "test", config=TransformConfig(), indices=list(range(4)))
            batch = next(iter(DataLoader(test_ds, batch_size=4)))
            with torch.no_grad():
                logits = model(batch["image"].to(device))
                recovered = restored(batch["image"].to(device))
                torch.testing.assert_close(logits, recovered, rtol=0, atol=1e-6)
                probs = torch.sigmoid(recovered.squeeze(1)).cpu().tolist()
            with (run_dir / "smoke_test_predictions_NOT_FOR_SUBMISSION.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Image_name", "Probability", "Pred_Label"])
                writer.writerows((n, p, int(p >= .5)) for n, p in zip(batch["image_name"], probs))
            del restored
        print(json.dumps({"variant": name, "status": "PASS", "loss": loss}), flush=True)
        del model, optimizer, scaler, before
    if source_hashes != {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}:
        raise RuntimeError("Source changed during verification; rerun against an unchanged version")
    report = {"status": "PASS", "scope": "INTEGRATION_SMOKE_ONLY", "device": device,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "source_sha256": source_hashes,
        "train_selected": [train_records[i].image_name for i in ti],
        "valid_selected": [valid_records[i].image_name for i in vi],
        "baseline_preprocessing_exact_match": True, "checkpoint_reload_test_csv_passed": True,
        "limitations": "Random initialization, one update per variant, deliberately stratified 32/32 subsets and four test images. No efficacy comparison or submission; A/C baseline/evaluation not yet merged or approved.",
        "results": results}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()

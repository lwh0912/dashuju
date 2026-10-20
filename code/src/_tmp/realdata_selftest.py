"""真实数据小批量自测（手册 1.3 最后一条，对接 B/C 后删除）。

流程：train 抽 320 张分层子集（含阳性）跑 10 个 batch 训练更新 →
valid 抽 256 张分层子集算 AUROC/AUPRC（sklearn 临时计算，
正式评估以 C 的统一评估函数为准）。
"""
import argparse
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src._tmp.minidata import CSVDataset, make_subset
from src.train import build_optimizer, load_config, save_checkpoint, set_seed, train_one_batch
from src.models import build_model


def evaluate(model, loader, device):
    model.eval()
    names, probs, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device)).squeeze(1)
            probs.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch["label"].tolist())
            names.extend(batch["image_name"])
    return (roc_auc_score(labels, probs),
            average_precision_score(labels, probs),
            names, labels, probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--train-batches", type=int, default=10)
    parser.add_argument("--valid-size", type=int, default=256)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = CSVDataset(cfg["data_root"], cfg["train_csv"], cfg["image_size"],
                          tuple(cfg["normalize"]["mean"]), tuple(cfg["normalize"]["std"]))
    valid_ds = CSVDataset(cfg["data_root"], cfg["valid_csv"], cfg["image_size"],
                          tuple(cfg["normalize"]["mean"]), tuple(cfg["normalize"]["std"]))
    train_sub = make_subset(train_ds, args.train_batches * cfg["batch_size"])
    valid_sub = make_subset(valid_ds, args.valid_size)
    g = torch.Generator(); g.manual_seed(cfg["seed"])
    train_ld = DataLoader(train_sub, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=cfg["num_workers"], generator=g)
    valid_ld = DataLoader(valid_sub, batch_size=64, shuffle=False,
                          num_workers=cfg["num_workers"])

    model = build_model(cfg["model"]["name"], pretrained=False).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, cfg)

    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    t0 = time.perf_counter()
    losses, nb = [], 0
    for batch in train_ld:
        losses.append(train_one_batch(model, batch["image"], batch["label"].float(),
                                      criterion, optimizer, device))
        nb += 1
        if nb >= args.train_batches:
            break
    if device == "cuda":
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t0

    auroc, auprc, names, labels, probs = evaluate(model, valid_ld, device)
    ckpt = save_checkpoint(cfg["output_dir"] + "/realdata_selftest.pt", model, cfg,
                           metrics={"valid_auroc": auroc, "valid_auprc": auprc})
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")

    n_pos_tr = int(sum(train_ds.df.iloc[i]["True_Label"] for i in train_sub.indices))
    print({
        "device": device,
        "train_batches": nb,
        "train_samples": len(train_sub),
        "train_positives": n_pos_tr,
        "mean_loss_last3": round(float(np.mean(losses[-3:])), 4),
        "train_sec": round(train_sec, 1),
        "peak_mem_MB": round(peak_mb, 1),
        "valid_subset": len(valid_sub),
        "valid_positives": int(sum(labels)),
        "valid_auroc": round(auroc, 4),
        "valid_auprc": round(auprc, 4),
        "ckpt": ckpt,
        "status": "PASS",
    })


if __name__ == "__main__":
    main()

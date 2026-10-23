"""训练入口（A 负责）。

已交付能力：
- 固定随机种子（torch / numpy / random）；
- --config 驱动全部路径与超参，禁止硬编码个人路径；
- 正式训练循环（已接入 B 数据、采样与损失模块）：
  BCEWithLogitsLoss（支持类别权重，应对 3.2% 阳性不均衡，
  正式采样/损失对照由 B 负责）+ AMP + AdamW + 余弦学习率
  + 按 valid AUROC 保存 best.pt（记录对应 epoch）；
- checkpoint 保存/模型重载（含配置与指标，不支持完整断点续训）；--selftest 随机张量自测。

对接约定（数据已接入 B；评估待 C 统一）：
- Dataset 返回 image / label / image_name；
- 模型输出 logits [B]，标签 [B] 为 0/1，BCE 前不 sigmoid；
- 评估用概率算 AUROC/AUPRC（C 交付统一评估后替换 sklearn 临时实现）。
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

class Progress:
    """极简进度显示（无第三方依赖）：每 20 批刷新一行，含 loss、已耗与预计剩余时间。"""

    def __init__(self, total: int, desc: str, interval: int = 20):
        self.total, self.desc, self.interval = total, desc, interval
        self.n, self.loss_sum = 0, 0.0
        self._t0 = time.perf_counter()

    def update(self, loss: float | None = None) -> None:
        self.n += 1
        if loss is not None:
            self.loss_sum += loss
        if self.n % self.interval == 0 or self.n == self.total:
            el = time.perf_counter() - self._t0
            eta = el / self.n * (self.total - self.n)
            loss_str = f"loss={self.loss_sum / self.n:.4f} " if self.n else ""
            print(f"\r{self.desc}: {self.n}/{self.total} "
                  f"({self.n / self.total * 100:.1f}%) {loss_str}"
                  f"已耗 {el / 60:.1f}min 预计剩余 {eta / 60:.1f}min",
                  end="", flush=True)
            if self.n == self.total:
                print()

from src.models import build_model


# ---------------------------------------------------------------- 基础工具

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    opt = cfg["optimizer"]
    if opt["name"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=opt["lr"],
                                 weight_decay=opt["weight_decay"])
    raise ValueError(f"unknown optimizer: {opt['name']}")


def save_checkpoint(path: str, model: torch.nn.Module, cfg: dict, metrics: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "metrics": metrics}, path)


def save_predictions(path: Path, metrics: dict) -> None:
    """Write the exact validation predictions corresponding to a saved checkpoint."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Image_name", "True_Label", "Probability"])
        writer.writerows(zip(metrics["names"], metrics["labels"], metrics["probs"], strict=True))


# ---------------------------------------------------------------- 训练/评估

def run_epoch(model, loader, criterion, optimizer, device, scaler, use_amp: bool,
              max_batches: int | None = None, desc: str = "train"):
    model.train()
    total_loss, nb = 0.0, 0
    total = max_batches or len(loader)
    pbar = Progress(total, desc)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())
        nb += 1
        pbar.update(float(loss.item()))
        if max_batches is not None and nb >= max_batches:
            break
    return total_loss / max(nb, 1)


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool, max_batches: int | None = None):
    model.eval()
    probs, labels, names = [], [], []
    nb = 0
    total = max_batches or len(loader)
    pbar = Progress(total, "valid")
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(images).squeeze(1)
        probs.extend(torch.sigmoid(logits.float()).cpu().tolist())
        labels.extend(batch["label"].tolist())
        names.extend(batch["image_name"])
        nb += 1
        pbar.update()
        if max_batches is not None and nb >= max_batches:
            break
    return {
        "auroc": roc_auc_score(labels, probs),
        "auprc": average_precision_score(labels, probs),
        "names": names, "labels": labels, "probs": probs,
    }


def train(cfg: dict, train_batches: int | None = None,
          valid_batches: int | None = None) -> None:
    for key, expected in {"resize_mode": "resize", "lr_scheduler": "cosine", "save_best_by": "valid_auroc"}.items():
        if cfg.get(key, expected) != expected:
            raise ValueError(f"Unsupported {key}: {cfg[key]!r}; expected {expected!r}")
    if cfg["epochs"] < 1:
        raise ValueError("epochs must be positive")
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = cfg.get("amp", True) and device == "cuda"
    out_dir = Path(cfg["output_dir"])
    if any((out_dir / name).exists() for name in ("best.pt", "best_auprc.pt", "last.pt", "history.json")):
        raise FileExistsError("Experiment outputs already exist; choose a new output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    # B's bundle owns datasets, transforms, sampling and loss construction.
    from src.data.pipeline import build_training_bundle
    bundle = build_training_bundle(cfg)
    train_ld, valid_ld = bundle.train_loader, bundle.valid_loader
    cfg["data_pipeline"] = bundle.metadata

    # --- 模型 / 损失 / 优化器 ---
    model = build_model(cfg["model"]["name"],
                        pretrained=cfg["model"]["pretrained"]).to(device)
    criterion = bundle.criterion.to(device)
    optimizer = build_optimizer(model, cfg)
    epochs = cfg["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"valid_auroc": -1.0, "epoch": -1}
    best_auprc = {"valid_auprc": -1.0, "epoch": -1}
    history = []
    final_metrics = {}
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        loss = run_epoch(model, train_ld, criterion, optimizer, device, scaler,
                         use_amp, max_batches=train_batches, desc=f"epoch {epoch}")
        if device == "cuda":
            torch.cuda.synchronize()
        metrics = evaluate(model, valid_ld, device, use_amp,
                           max_batches=valid_batches)
        sched.step()
        final_metrics = {"valid_auroc": metrics["auroc"], "valid_auprc": metrics["auprc"], "epoch": epoch}
        rec = {"epoch": epoch, "train_loss": round(loss, 4),
               "valid_auroc": round(metrics["auroc"], 4),
               "valid_auprc": round(metrics["auprc"], 4),
               "epoch_sec": round(time.perf_counter() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        if metrics["auroc"] > best["valid_auroc"]:
            best = {"valid_auroc": metrics["auroc"], "epoch": epoch}
            save_predictions(out_dir / "best_valid_predictions.csv", metrics)
            save_checkpoint(str(out_dir / "best.pt"), model, cfg,
                            {"valid_auroc": metrics["auroc"],
                             "valid_auprc": metrics["auprc"], "epoch": epoch})
        if metrics["auprc"] > best_auprc["valid_auprc"]:
            best_auprc = {"valid_auprc": metrics["auprc"], "epoch": epoch}
            save_predictions(out_dir / "best_auprc_valid_predictions.csv", metrics)
            save_checkpoint(str(out_dir / "best_auprc.pt"), model, cfg,
                            {"valid_auroc": metrics["auroc"],
                             "valid_auprc": metrics["auprc"], "epoch": epoch})

    save_checkpoint(str(out_dir / "last.pt"), model, cfg,
                    final_metrics)
    summary = {"best": best, "best_auprc": best_auprc, "auprc_definition": "sklearn.average_precision_score"}
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg, **summary, "history": history}, f,
                  ensure_ascii=False, indent=2)
    print(json.dumps({**summary, "history_path": str(out_dir / "history.json"),
                      "status": "DONE"}, ensure_ascii=False))


# ---------------------------------------------------------------- 自测

def train_one_batch(model, images, labels, criterion, optimizer, device) -> float:
    model.train()
    images, labels = images.to(device), labels.to(device)
    logits = model(images).squeeze(1)
    loss = criterion(logits, labels.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def selftest(cfg: dict) -> None:
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg["model"]["name"],
                        pretrained=cfg["model"]["pretrained"]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, cfg)
    batch, size = cfg["batch_size"], cfg["image_size"]
    images = torch.randn(batch, 3, size, size)
    labels = torch.randint(0, 2, (batch,)).float()
    loss = train_one_batch(model, images, labels, criterion, optimizer, device)
    assert np.isfinite(loss) and model(images.to(device)).squeeze(1).shape == (batch,)
    ckpt_path = str(Path(cfg["output_dir"]) / "selftest_ckpt.pt")
    save_checkpoint(ckpt_path, model, cfg, metrics={"valid_auroc": None})
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd, saved_cfg = checkpoint["state_dict"], checkpoint["config"]
    m2 = build_model(saved_cfg["model"]["name"], pretrained=False).to(device)
    m2.load_state_dict(sd); m2.eval(); model.eval()
    with torch.no_grad():
        o1 = torch.sigmoid(model(images.to(device)).squeeze(1))
        o2 = torch.sigmoid(m2(images.to(device)).squeeze(1))
    assert torch.allclose(o1, o2, atol=1e-6)
    print(json.dumps({"loss": round(loss, 4), "ckpt": ckpt_path, "status": "PASS"},
                     ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--selftest", action="store_true", help="随机张量自测，不读真实数据")
    parser.add_argument("--train-batches", type=int, default=None, help="每 epoch 限批次数（冒烟测试用）")
    parser.add_argument("--valid-batches", type=int, default=None, help="验证限批次数（冒烟测试用）")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 epoch 数")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.selftest:
        selftest(cfg)
    else:
        train(cfg, train_batches=args.train_batches, valid_batches=args.valid_batches)


if __name__ == "__main__":
    main()

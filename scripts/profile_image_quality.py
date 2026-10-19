"""Profile cached image features and export a small, read-only review gallery."""

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageOps, ImageDraw

from find_near_duplicates import safe_source

PROJECT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")
METRICS = {"brightness_mean": "平均灰度亮度", "contrast_std": "灰度标准差",
           "dark_fraction_below_10": "暗像素比例", "short_side": "短边像素",
           "aspect_ratio": "长短边比"}
REASONS = [("brightness_mean", True, "亮度最低"), ("brightness_mean", False, "亮度最高"),
           ("contrast_std", True, "灰度标准差最低"), ("dark_fraction_below_10", False, "暗像素比例最高"),
           ("short_side", True, "短边最小"), ("aspect_ratio", False, "长短边比最高")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_table(root, feature_dir):
    report = json.loads((feature_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETE" or report.get("scope") != "full" or report.get("errors"):
        raise ValueError("A complete, error-free full feature report is required")
    table = pd.read_csv(feature_dir / "features.csv", dtype={"image_name": str, "relative_path": str, "dhash64": str})
    if table.duplicated(["split", "image_name"]).any():
        raise ValueError("Duplicate feature identifiers")
    if set(table["split"]) != set(SPLITS):
        raise ValueError("Unexpected or absent split")
    if len(table) != report["success"] or len(table) != report["dataset_total"]:
        raise ValueError("Feature count does not match completed report")
    numeric = ["width", "height", "brightness_mean", "contrast_std", "dark_fraction_below_10"]
    if not np.isfinite(table[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite feature value")
    if ((table[["width", "height"]] <= 0).any().any()
            or (table[["width", "height"]] % 1 != 0).any().any()
            or not table["brightness_mean"].between(0, 255).all()
            or not table["contrast_std"].between(0, 127.5 + 1e-4).all()
            or not table["dark_fraction_below_10"].between(0, 1).all()):
        raise ValueError("Feature outside expected range")
    pieces = []
    for split in SPLITS:
        official = pd.read_csv(root / f"{split}.csv", dtype={"Image_name": str})
        part = table[table["split"] == split].copy()
        if (official["Image_name"].duplicated().any()
                or set(part["image_name"]) != set(official["Image_name"])
                or len(part) != report["dataset_counts"][split]):
            raise ValueError(f"Official sample identifiers/counts do not match: {split}")
        aligned = official.set_index("Image_name").loc[part["image_name"]]
        if split != "test":
            if not aligned["True_Label"].isin([0, 1]).all() or not np.array_equal(part["label"], aligned["True_Label"]):
                raise ValueError(f"Labels do not match official CSV: {split}")
        elif part["label"].notna().any() or "True_Label" in official:
            raise ValueError("Test data must not contain labels")
        ages = pd.to_numeric(aligned["Age"], errors="raise").to_numpy(dtype=float)
        if np.isinf(ages).any():
            raise ValueError("Infinite age")
        part["age"] = ages
        pieces.append(part)
    table = pd.concat(pieces, ignore_index=True)
    table["short_side"] = table[["width", "height"]].min(axis=1)
    table["aspect_ratio"] = table[["width", "height"]].max(axis=1) / table["short_side"]
    return table, report


def describe(values):
    values = values.dropna()
    if values.empty:
        return {"count": 0}
    quantiles = values.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1])
    return {"count": len(values), "mean": float(values.mean()),
            **{name: float(value) for name, value in zip(
                ["min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"], quantiles)}}


def summarize(table):
    train = table[table["split"] == "train"]
    thresholds = {metric: {"low": float(train[metric].quantile(.01)), "high": float(train[metric].quantile(.99))}
                  for metric in METRICS}
    splits = {}
    for split in SPLITS:
        part = table[table["split"] == split]
        metrics = {metric: describe(part[metric]) for metric in METRICS}
        tails = {metric: {"below_train_p01": int((part[metric] < thresholds[metric]["low"]).sum()),
                          "above_train_p99": int((part[metric] > thresholds[metric]["high"]).sum())}
                 for metric in METRICS}
        counts = part.groupby(["width", "height"]).size().sort_values(ascending=False).head(5)
        splits[split] = {"rows": len(part), "positive": None if split == "test" else int(part["label"].sum()),
                         "positive_rate": None if split == "test" else float(part["label"].mean()),
                         "metrics": metrics, "train_reference_tails": tails,
                         "age": {**describe(part["age"]), "missing": int(part["age"].isna().sum())},
                         "top_dimensions": [{"width": int(w), "height": int(h), "count": int(n)} for (w, h), n in counts.items()],
                         "by_label": {str(int(label)): {"rows": len(group), "metrics": {m: describe(group[m]) for m in METRICS}}
                                      for label, group in part.groupby("label")}}
    return thresholds, splits


def choose_reviews(table):
    chosen, seen = [], set()
    for split in SPLITS:
        part = table[table["split"] == split]
        for metric, ascending, reason in REASONS:
            ranked = part.sort_values([metric, "image_name"], ascending=[ascending, True])
            for _, row in ranked.iterrows():
                key = (split, row["image_name"])
                if key in seen:
                    continue
                seen.add(key)
                chosen.append({"split": split, "image_name": row["image_name"], "reason": reason,
                               "rank_within_reason": int((ranked["image_name"].to_numpy() == row["image_name"]).argmax()) + 1,
                               "metric": metric, "value": float(row[metric]), "review_status": "待复核"})
                break
    return chosen


def plot_distributions(table, target):
    titles = ["Mean grayscale brightness (0-255)", "Grayscale standard deviation",
              "Dark pixel fraction (<10)", "Short side (pixels)", "Long / short side ratio"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    colors = ["#2563eb", "#d97706", "#15803d"]
    for axis, metric, title in zip(axes.flat, METRICS, titles):
        for split, color in zip(SPLITS, colors):
            values = np.sort(table.loc[table["split"] == split, metric].to_numpy())
            axis.plot(values, np.arange(1, len(values) + 1) / len(values), label=split, color=color, linewidth=1.5)
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("Cumulative fraction")
        axis.set_ylim(0, 1.01)
        axis.grid(alpha=.2)
    axes.flat[0].legend()
    axes.flat[5].axis("off")
    axes.flat[5].text(.02, .9, "63,100 cached feature rows\n\nEach curve uses its own split denominator.\nDark borders are included.\nStatistical tails are not bad-image verdicts.\nNo image or label changes.", va="top", fontsize=10)
    fig.suptitle("Image quality profile: descriptive split comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def run(root, feature_dir, output):
    root, feature_dir, output = root.resolve(), feature_dir.resolve(), output.resolve()
    if output == root or root in output.parents or output == feature_dir or feature_dir in output.parents:
        raise ValueError("Output must be outside raw data and cached features")
    table, extraction = load_table(root, feature_dir)
    thresholds, splits = summarize(table)
    output.mkdir(parents=True, exist_ok=True)
    reviews = choose_reviews(table)
    previews = output / "previews"
    previews.mkdir(exist_ok=True)
    lookup = table.set_index(["split", "image_name"])
    cards = []
    for number, item in enumerate(reviews, 1):
        row = lookup.loc[(item["split"], item["image_name"])].to_dict()
        with Image.open(safe_source(root, row)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((520, 380), Image.Resampling.LANCZOS)
        name = f"{number:02d}-{item['split']}-{Path(item['image_name']).stem}.jpg"
        image.save(previews / name, quality=90)
        item["preview"] = f"previews/{name}"
        card = Image.new("RGB", (540, 420), "white")
        card.paste(image, ((540 - image.width) // 2, 35 + (380 - image.height) // 2))
        ImageDraw.Draw(card).text((10, 10), f"{number:02d} | {item['split']} | {item['image_name']}", fill="black")
        cards.append(card)
    sheet = Image.new("RGB", (540 * 3, 420 * ((len(cards) + 2) // 3)), "#dddddd")
    for i, card in enumerate(cards):
        sheet.paste(card, ((i % 3) * 540, (i // 3) * 420))
    sheet.save(output / "contact-sheet.jpg", quality=90)
    plot_distributions(table, output / "distributions.png")
    age_questions = table[table["age"].notna() & ((table["age"] <= 0) | (table["age"] > 100))]
    age_records = [{"split": r["split"], "image_name": r["image_name"], "age": float(r["age"])} for _, r in age_questions.iterrows()]
    report = {"status": "PROFILE_COMPLETE", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "pillow": PIL.__version__, "matplotlib": matplotlib.__version__},
              "inputs_sha256": {str(p): sha(p) for p in [feature_dir / "features.csv", feature_dir / "report.json", *[root / f"{s}.csv" for s in SPLITS]]},
              "source_feature_signature": extraction["feature_signature"], "feature_versions": extraction["versions"],
              "rows": len(table), "train_reference_thresholds": thresholds, "splits": splits,
              "age_values_for_source_verification": age_records, "review_samples": reviews,
              "limitations": ["Cached extraction snapshot; only the 18 preview source files are rechecked by size/mtime, no new full image scan.",
                              "Brightness/contrast/dark fraction are grayscale thumbnail statistics including borders; not medical image-quality scores.",
                              "Tail thresholds fitted on train only, strict comparisons; descriptive review flags, never removal or training cutoffs.",
                              "Age <=0 or >100 is only a source-verification question, not an invalidity or medical judgment.",
                              "No patient/device identifiers available in required CSVs; no patient leakage or device-causation conclusion.",
                              "Near duplicates remain unresolved; no source images, labels, official splits or human reviews altered."]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 队员 B：数据画像与质量复核初稿", "", "本报告复用已完成的全量特征，核对官方 CSV 的样本与标签，不重新扫描全量原图。统计为提取时快照，仅对 18 张预览核对文件大小/修改时间并读取图片。", "",
             "## 样本与类别", "", "| 划分 | 数量 | 阳性数量 | 阳性比例 | 年龄缺失 | 年龄范围 |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for split, info in splits.items():
        pos = "无标签" if info["positive"] is None else str(info["positive"])
        rate = "—" if info["positive_rate"] is None else f"{info['positive_rate']:.4%}"
        lines.append(f"| {split} | {info['rows']:,} | {pos} | {rate} | {info['age']['missing']} | {info['age']['min']:g}–{info['age']['max']:g} |")
    lines += ["", "## 图像特征分布", "", "下表为中位数 [P01, P99]。亮度取值 0–255，暗像素为灰度<10 的比例；统计包含黑色边框。灰度标准差只衡量明暗变化，不能替代清晰度或医学可用性判定。", "",
              "| 特征 | train | valid | test |", "| --- | --- | --- |"]
    for metric, label in METRICS.items():
        cells = [f"{splits[s]['metrics'][metric]['median']:.3f} [{splits[s]['metrics'][metric]['p01']:.3f}, {splits[s]['metrics'][metric]['p99']:.3f}]" for s in SPLITS]
        lines.append("| " + " | ".join([label, *cells]) + " |")
    lines += ["", "![各划分累积分布](distributions.png)", "", "## 相对训练集尾部的数量", "", "边界只用训练集 P01/P99 计算，严格小于/大于才计数；并列值会导致数量不恰好为 1%。同一张图可命中多项，不能相加作为坏图总数。", "",
              "| 特征 | 训练参考范围 | train 低/高 | valid 低/高 | test 低/高 |", "| --- | --- | --- | --- | --- |"]
    for metric, label in METRICS.items():
        cells = [f"{splits[s]['train_reference_tails'][metric]['below_train_p01']}/{splits[s]['train_reference_tails'][metric]['above_train_p99']}" for s in SPLITS]
        lines.append("| " + " | ".join([label, f"{thresholds[metric]['low']:.3f}–{thresholds[metric]['high']:.3f}", *cells]) + " |")
    lines += ["", "## 各类别描述性对照", "", "仅比较 train/valid 的类别中位数，不使用测试标签；类别不平衡、采集条件等可能影响分布，不能把差异解释为病因或设备效应。", "",
              "| 划分/标签 | 数量 | 亮度中位数 | 灰度标准差中位数 | 暗像素比例中位数 |", "| --- | ---: | ---: | ---: | ---: |"]
    for split in ("train", "valid"):
        for label, group in splits[split]["by_label"].items():
            vals = [f"{group['metrics'][m]['median']:.3f}" for m in list(METRICS)[:3]]
            lines.append("| " + " | ".join([f"{split}/{label}", str(group["rows"]), *vals]) + " |")
    lines += ["", "## 尺寸与年龄待核实", ""]
    for split in SPLITS:
        sizes = "、".join(f"{r['width']}×{r['height']}（{r['count']:,} 张）" for r in splits[split]["top_dimensions"])
        lines.append(f"- {split} 最常见尺寸：{sizes}。")
    lines += [f"- 年龄≤0 或>100 的来源核实样本共 {len(age_records)} 条，明细在 report.json；这些条件是核实线索，不是自动清洗规则。缺失年龄保持原值，尚未填补。", "",
              "## 少量质量复核", "", "打开 [review.html](review.html) 查看 18 张原比例缩略图，每个划分选 6 张。按统计极端程度挑选并去重，不代表总体质量比例；重复命中时使用下一张，因此附带排名。可在聊天按编号反馈“可辨认 / 内容难辨 / 不确定”及原因；网页没有自动保存表单。", "",
              "## 当前结论与交接", "", "- 既有审计已完成全量解码且错误为 0，精确字节重复为 0；本轮未重跑这些检查。",
              "- 近重复校准已发现裁剪/翻转盲区，仍未排除近重复或患者重叠。",
              "- 先人工确认少量尾部样本的主要问题，不按亮度、暗区或年龄阈值直接删图、改标签或划分。",
              "- 不凭本轮统计直接决定增强强度；训练增强与验证预处理分开，参数须在固定基线上消融。",
              "- 逐图结果及预览仅保存在本地，代码和方法说明可交 C 复核。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    gallery = []
    for number, item in enumerate(reviews, 1):
        gallery.append(f'<article><h2>{number:02d} · {html.escape(item["split"])} · {html.escape(item["image_name"])}</h2><p>{html.escape(item["reason"])}，排名 {item["rank_within_reason"]}；数值 {item["value"]:.3f}</p><a href="{item["preview"]}"><img src="{item["preview"]}" alt="{html.escape(item["image_name"])}"></a><p>待复核 · 统计极端不等于坏图</p></article>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>数据质量样本复核</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 20px;background:#f4f6f8;color:#172033}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}article{background:white;padding:16px;border-radius:10px}h2{font-size:17px}img{max-width:100%;height:auto}p{line-height:1.6}</style><h1>数据质量样本复核 · 18 张</h1><p>每个划分 6 张，按亮度、灰度标准差、暗区、尺寸等统计选取。只需描述可见内容，不能据此判定病情。点击图可放大；可以在聊天按编号反馈，页面不会保存填写结果。</p><main>' + "".join(gallery) + "</main></html>"
    (output / "review.html").write_text(page, encoding="utf-8")
    print(f"PROFILE_COMPLETE: {len(table)} rows; {len(reviews)} previews; age questions={len(age_records)}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, default=PROJECT / "outputs/image-features")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/data-profile")
    args = parser.parse_args()
    run(args.data_root, args.features_dir, args.output_dir)

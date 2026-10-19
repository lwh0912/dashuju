"""Small, reproducible image-feature pilot; never modifies source images.

Extracts a simple 64-bit difference hash and grayscale thumbnail statistics.
This pilot does not detect all near duplicates or establish a quality cutoff.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image, ImageOps, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from glaucoma.data import load_records  # noqa: E402


def select_sample(records, count, seed, labelled):
    """Include both classes in labelled pilots; this is not prevalence sampling."""
    rng = random.Random(seed)
    if not labelled:
        return rng.sample(records, min(count, len(records)))
    groups = [[r for r in records if r.label == label] for label in (0, 1)]
    chosen = []
    for group, budget in zip(groups, ((count + 1) // 2, count // 2)):
        chosen.extend(rng.sample(group, min(budget, len(group))))
    selected = {r.image_name for r in chosen}
    remainder = [r for r in records if r.image_name not in selected]
    chosen.extend(rng.sample(remainder, min(max(0, count - len(chosen)), len(remainder))))
    rng.shuffle(chosen)
    return chosen


def features(path):
    started = time.perf_counter()
    with Image.open(path) as image:
        image.load()
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    width, height = rgb.size
    thumb = rgb.copy()
    thumb.thumbnail((256, 256), Image.Resampling.LANCZOS)
    gray = thumb.convert("L")
    pixels = np.asarray(gray, dtype=np.float32)
    small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return {"width": width, "height": height, "dhash64": f"{value:016x}",
            "brightness_mean": float(pixels.mean()), "contrast_std": float(pixels.std()),
            "dark_fraction_below_10": float((pixels < 10).mean()),
            "feature_seconds": time.perf_counter() - started}, thumb


def run(root, output, per_split, seed):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside source data")
    if per_split < 2:
        raise ValueError("per-split must be at least 2")
    output.mkdir(parents=True, exist_ok=True)
    (output / "thumbnails").mkdir(exist_ok=True)
    started = time.perf_counter()
    manifest_started = time.perf_counter()
    selected, populations = [], {}
    for offset, split in enumerate(("train", "valid", "test")):
        print(f"[load] {split}", flush=True)
        records = load_records(root, split)
        populations[split] = len(records)
        selected.extend((split, row) for row in select_sample(records, per_split, seed + offset, split != "test"))
    manifest_seconds = time.perf_counter() - manifest_started
    results, errors, previews = [], [], []
    fields = ["split", "image_name", "label", "width", "height", "dhash64", "brightness_mean",
              "contrast_std", "dark_fraction_below_10", "feature_seconds"]
    with (output / "features.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, (split, row) in enumerate(selected, start=1):
            try:
                values, thumb = features(row.path)
                values.update(split=split, image_name=row.image_name, label=row.label)
                writer.writerow(values)
                handle.flush()
                results.append(values)
                thumb.save(output / "thumbnails" / f"{split}_{Path(row.image_name).stem}.jpg", quality=90)
                if sum(p[0] == split for p in previews) < 5:
                    previews.append((split, row.image_name, thumb.copy()))
            except Exception as exc:
                errors.append({"split": split, "image_name": row.image_name, "error": str(exc)})
            if i % 10 == 0 or i == len(selected):
                print(f"[features] {i}/{len(selected)}, errors={len(errors)}, "
                      f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
    if previews:
        sheet = Image.new("RGB", (5 * 256, 3 * 282), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (split, name, thumb) in enumerate(previews):
            x, y = (index % 5) * 256, (index // 5) * 282
            sheet.paste(thumb, (x + (256 - thumb.width) // 2, y))
            draw.text((x + 4, y + 258), f"{split}: {name}", fill="black")
        sheet.save(output / "contact-sheet.jpg", quality=90)
    by_split, estimated_features = {}, 0
    for split, total in populations.items():
        group = [r for r in results if r["split"] == split]
        seconds = sum(r["feature_seconds"] for r in group)
        average = seconds / len(group) if group else None
        by_split[split] = {"population": total, "sampled": sum(s == split for s, _ in selected),
                           "success": len(group), "feature_seconds": seconds,
                           "mean_seconds_per_image": average,
                           "sample_label_counts": {str(label): sum(r["label"] == label for r in group)
                                                   for label in ([0, 1] if split != "test" else [None])}}
        if average is not None:
            estimated_features += average * total
    report = {"status": "PASS" if not errors else "FAIL", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "data_root": str(root), "seed": seed, "requested_per_split": per_split,
              "selected": len(selected), "success": len(results), "errors": errors,
              "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "Pillow": Image.__version__},
              "feature_config": {"thumbnail_max_size": [256, 256], "apply_exif_orientation": True,
                                 "resampling": "LANCZOS", "dhash_grid": [9, 8],
                                 "dhash_comparison": "right > left", "dark_pixel_threshold": 10},
              "manifest_seconds": manifest_seconds, "elapsed_seconds": time.perf_counter() - started,
              "rough_serial_feature_minutes": estimated_features / 60 if not errors else None,
              "splits": by_split,
              "limitations": ["Pilot only; no full-dataset near-duplicate search has run.",
                              "64-bit dHash is a coarse fingerprint, not proof of image identity and not crop/rotation invariant.",
                              "Statistics use <=256px grayscale thumbnails and include dark borders; no medical quality cutoff is inferred.",
                              "Labelled samples balance classes, so sample prevalence is not dataset prevalence.",
                              "Time extrapolation uses a small sample and excludes full search, checkpointing and output overhead; it is not an ETA."]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 图像特征小样本试跑", "", f"状态：{report['status']}；成功：{len(results)}/{len(selected)}；错误：{len(errors)}。",
             f"总耗时：{report['elapsed_seconds']:.1f} 秒；目录与 CSV 核对：{manifest_seconds:.1f} 秒。", "",
             "| 划分 | 样本成功数 | 每张特征提取均时（秒） |", "| --- | ---: | ---: |"]
    for split, values in by_split.items():
        avg = values["mean_seconds_per_image"]
        lines.append(f"| {split} | {values['success']} | {avg:.4f} |" if avg is not None else f"| {split} | 0 | 无法估计 |")
    lines += ["", "生成 features.csv、thumbnails/ 和 contact-sheet.jpg；均留在本地。",
              "本轮只验证特征提取，没有执行全量相似图搜索，也没有设置自动删图阈值。",
              "指纹只是候选筛选工具，相同或相近指纹仍需人工复核；裁剪、旋转版本可能漏检。",
              "亮度/对比度基于缩略图，包含黑边，不等同于医学图像质量结论。",
              "有标签划分在样本中平衡两类，不能用这批样本重新估算阳性比例。",
              f"按小样本均时粗算全量串行特征提取约 {estimated_features / 60:.1f} 分钟（仅供规划，非完成时间承诺）。",
              "该估计不包含全量候选检索、缓存和输出开销；样本尺寸与缓存状态会影响速度。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"RESULT: {report['status']}; {len(results)}/{len(selected)}; elapsed={report['elapsed_seconds']:.1f}s", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/image-feature-pilot")
    parser.add_argument("--per-split", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    return int(run(args.data_root, args.output_dir, args.per_split, args.seed)["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())

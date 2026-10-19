"""Bounded feasibility study; never classifies or removes dataset duplicates."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import PIL
from PIL import Image, ImageEnhance, ImageOps

from find_near_duplicates import safe_source
from pilot_image_features import features
from verify_reviewed_pairs import compare

PROJECT = Path(__file__).resolve().parents[1]
# Declared before the experiment. This is a review-priority gate, NOT a verdict.
GATE = {"ransac_inliers": 20, "inlier_fraction": 0.5,
        "left_inlier_hull_fraction": 0.05, "right_inlier_hull_fraction": 0.05,
        "overlap_gray_correlation": 0.75}


def priority_gate(metrics):
    return all(metrics.get(key) is not None and np.isfinite(metrics[key])
               and metrics[key] >= value for key, value in GATE.items())


def variants(rgb):
    """Synthetic controls on a <=1000px copy, not training augmentation advice."""
    w, h = rgb.size
    yield "brightness_0.7", ImageEnhance.Brightness(rgb).enhance(0.7)
    channels = np.asarray(rgb, dtype=np.float32) * np.array([1.15, 0.8, 0.95])
    yield "color_channels", Image.fromarray(np.uint8(np.clip(channels, 0, 255)))
    buffer = BytesIO()
    rgb.save(buffer, format="JPEG", quality=50)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        yield "jpeg_q50", compressed.convert("RGB")
    yield "resize_0.65", rgb.resize((max(1, round(w * .65)), max(1, round(h * .65))), Image.Resampling.LANCZOS)
    yield "rotation_10deg", rgb.rotate(10, resample=Image.Resampling.BICUBIC)
    yield "center_crop_80pct", rgb.crop((w // 10, h // 10, w - w // 10, h - h // 10))
    yield "horizontal_flip", ImageOps.mirror(rgb)


def image_hash(rgb):
    # Same feature function, recomputed for BOTH control sides in this environment.
    buffer = BytesIO()
    rgb.save(buffer, format="PNG")
    buffer.seek(0)
    return int(features(buffer)[0]["dhash64"], 16)


def read_rgb(path):
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        rgb.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        return rgb


def run(root, feature_file, pair_file, output, seed=2026):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside source data")
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    started = time.perf_counter()
    with Path(feature_file).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(r["split"], r["image_name"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate feature identifiers")
    with Path(pair_file).open(encoding="utf-8-sig", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    if not 1 <= len(pairs) <= 20:
        raise ValueError("This bounded study accepts only 1..20 preview pairs")
    rng = random.Random(seed)
    controls = []
    for label in ("0", "1"):
        pool = sorted((r for r in rows if r["split"] == "train" and r["label"] == label), key=lambda r: r["image_name"])
        controls.extend(rng.sample(pool, 4))
    # Source identifiers are retained; synthetic positives are not confirmed real pairs.
    loaded = {}

    def get_image(row):
        key = (row["split"], row["image_name"])
        if key not in loaded:
            loaded[key] = read_rgb(safe_source(root, row))
        return loaded[key]

    results = []
    for number, row in enumerate(controls, 1):
        rgb = get_image(row)
        gray = np.asarray(rgb.convert("L"))
        original_hash = image_hash(rgb)
        for name, changed in variants(rgb):
            metrics, _ = compare(gray, np.asarray(changed.convert("L")))
            distance = (original_hash ^ image_hash(changed)).bit_count()
            results.append({"source": row["image_name"], "label": row["label"],
                            "transform": name, "known_relation": "synthetic_same_image",
                            "dhash_distance": distance, "dhash_radius4_pass": distance <= 4,
                            "geometry_priority_pass": priority_gate(metrics), "metrics": metrics})
        print(f"[controls] {number}/{len(controls)} sources; {len(results)} comparisons", flush=True)
    candidates = []
    for number, pair in enumerate(pairs, 1):
        a = get_image(lookup[(pair["left_split"], pair["left_image"])])
        b = get_image(lookup[(pair["right_split"], pair["right_image"])])
        metrics, preview = compare(np.asarray(a.convert("L")), np.asarray(b.convert("L")))
        passed = priority_gate(metrics)
        item = {"pair_id": pair["pair_id"], "left_image": pair["left_image"],
                "right_image": pair["right_image"], "geometry_priority_pass": passed,
                "metrics": metrics, "duplicate_verdict": None}
        if passed:
            target = output / f"{pair['pair_id']}-priority.jpg"
            if not cv2.imwrite(str(target), preview):
                raise OSError(f"Cannot write {target}")
            item["preview"] = target.name
        candidates.append(item)
        print(f"[candidates] {number}/{len(pairs)}; {pair['pair_id']}; inliers={metrics['ransac_inliers']}; priority={passed}", flush=True)
    summary = {}
    for name in sorted({r["transform"] for r in results}):
        group = [r for r in results if r["transform"] == name]
        summary[name] = {"count": len(group),
                         "dhash_radius4_pass": sum(r["dhash_radius4_pass"] for r in group),
                         "geometry_priority_pass": sum(r["geometry_priority_pass"] for r in group),
                         "both_pass": sum(r["dhash_radius4_pass"] and r["geometry_priority_pass"] for r in group)}
    report = {"status": "BOUNDED_CALIBRATION_COMPLETE", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "seed": seed, "elapsed_seconds": time.perf_counter() - started,
              "versions": {"opencv": cv2.__version__, "numpy": np.__version__, "pillow": PIL.__version__},
              "inputs_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (feature_file, pair_file)},
              "priority_gate": GATE, "source_classes": dict(Counter(r["label"] for r in controls)),
              "control_summary": summary, "controls": results, "candidates": candidates,
              "candidate_priority_count": sum(r["geometry_priority_pass"] for r in candidates),
              "limitations": [
                  "Eight training sources, four per class; seven deterministic variants each. Not independent population samples.",
                  "Controls are created after resizing to max side 1000; dHash recomputed for both sides in container versions, not mixed with cached hashes.",
                  "Gate declared before the run; this is a feasibility study without held-out real duplicate ground truth or a measured production false-positive rate.",
                  "Twenty candidates are the existing diversified previews, not a random sample of all candidates.",
                  "Low matches or failed gate cannot rule out duplicates or same-patient overlap. High matches require manual acquisition-level verification.",
                  "User judgments, feature cache, source images, labels and official splits are unchanged."]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 近重复二阶段方法：小规模校准", "", "这是方法可行性检查；通过门槛只表示优先人工复核，不代表确认重复。",
             "训练集随机选 8 张（每类 4 张，seed=2026），每张生成 7 种已知同图变体，共 56 对；全部在内存中操作。", "",
             "| 已知变体 | 对照数 | dHash 距离≤4 | 局部结构门槛通过 | 两阶段都通过 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, values in summary.items():
        lines.append(f"| {name} | {values['count']} | {values['dhash_radius4_pass']} | {values['geometry_priority_pass']} | {values['both_pass']} |")
    lines += ["", f"现有 20 对预览中，优先复核门槛通过 {report['candidate_priority_count']} 对。这不等于其余候选已排除。", "",
              "门槛预先固定：内点≥20、内点比例≥0.5、两侧匹配覆盖面积各≥5%、对齐灰度相关性≥0.75。没有用本轮结果反复调参。", "",
              "| 候选 | 内点 | 优先复核门槛 |", "| --- | ---: | --- |"]
    for item in candidates:
        lines.append(f"| {item['pair_id']} | {item['metrics']['ransac_inliers']} | {'通过，需人工核实' if item['geometry_priority_pass'] else '未通过，不能据此排除'} |")
    lines += ["", "## 使用边界", "", "- 对照是在最长边 1000 像素的副本上生成；重新计算两侧指纹，没有混用已有全量缓存。",
              "- 8 张源图的 56 个变体不等于 56 个独立样本；不能据此声称全量召回率、精确率或没有泄漏。",
              "- 20 对是已有的多样化预览，存在选择偏差；用户第 1、3 对的疑似结论保持不变。",
              "- dHash 漏掉的候选无法由后续局部验证补回，须先解决候选召回问题再考虑扩大范围。",
              "- 本次仅作工具校准，不用于选模型、改标签或重划官方数据。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"RESULT: {report['status']}; controls={len(results)}; candidates={len(candidates)}; priority={report['candidate_priority_count']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=PROJECT / "outputs/image-features/features.csv")
    parser.add_argument("--pairs", type=Path, default=PROJECT / "outputs/near-duplicates/train-valid/review-template.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/near-duplicates/train-valid/calibration")
    args = parser.parse_args()
    run(args.data_root, args.features, args.pairs, args.output_dir)

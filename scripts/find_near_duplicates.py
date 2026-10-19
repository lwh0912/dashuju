"""Cross-split dHash candidate search. Candidates are NOT confirmed duplicates."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import heapq
import html
from itertools import islice, product
import json
from pathlib import Path
import sys
import time

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT = Path(__file__).resolve().parents[1]


def partitions(radius):
    if not 0 <= radius <= 6:
        raise ValueError("Use a 64-bit Hamming radius between 0 and 6")
    count = radius + 1
    width, extra = divmod(64, count)
    offset, parts = 0, []
    for index in range(count):
        bits = width + int(index < extra)
        parts.append((offset, (1 << bits) - 1))
        offset += bits
    return parts


def build_index(values, radius):
    # With <=radius bit differences, at least one of radius+1 parts is exact.
    parts = partitions(radius)
    tables = [defaultdict(list) for _ in parts]
    for value in values:
        for table, (shift, mask) in zip(tables, parts):
            table[(value >> shift) & mask].append(value)
    return parts, tables


def query(value, parts, tables, radius):
    pool = set()
    for table, (shift, mask) in zip(tables, parts):
        pool.update(table.get((value >> shift) & mask, ()))
    for other in pool:
        distance = (value ^ other).bit_count()
        if distance <= radius:
            yield other, distance


def search(left, right, radius=4, max_pairs=100000, callback=None):
    if max_pairs < 1:
        raise ValueError("max-pairs must be positive")
    grouped_left, grouped_right = defaultdict(list), defaultdict(list)
    for i, row in enumerate(left):
        grouped_left[int(row["dhash64"], 16)].append(i)
    for i, row in enumerate(right):
        grouped_right[int(row["dhash64"], 16)].append(i)
    parts, tables = build_index(grouped_left, radius)
    best, histogram = [], Counter()
    total, different_labels = 0, 0
    last_log = time.monotonic()
    for number, (value, right_ids) in enumerate(grouped_right.items(), start=1):
        for other, distance in query(value, parts, tables, radius):
            left_ids = grouped_left[other]
            pair_count = len(left_ids) * len(right_ids)
            total += pair_count
            histogram[distance] += pair_count
            a = Counter(left[i]["label"] for i in left_ids)
            b = Counter(right[i]["label"] for i in right_ids)
            different_labels += a["0"] * b["1"] + a["1"] * b["0"]
            # Only the first max_pairs from a tied hash group can enter top max_pairs.
            best_possible = (-distance, -left_ids[0], -right_ids[0])
            if len(best) >= max_pairs and best_possible <= best[0]:
                continue
            for i, j in islice(product(left_ids, right_ids), max_pairs):
                entry = (-distance, -i, -j)
                if len(best) < max_pairs:
                    heapq.heappush(best, entry)
                elif entry > best[0]:
                    heapq.heapreplace(best, entry)
        if callback and (number % 200 == 0 or time.monotonic() - last_log >= 5):
            callback(number, len(grouped_right), total)
            last_log = time.monotonic()
    ranked = sorted((-d, -i, -j) for d, i, j in best)
    return {"candidate_count": total, "pairs": ranked,
            "distance_histogram": {str(d): histogram[d] for d in range(radius + 1)},
            "different_known_label_candidates": different_labels,
            "unique_left_hashes": len(grouped_left), "unique_right_hashes": len(grouped_right)}


def load_features(directory):
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETE" or report.get("scope") != "full" or report.get("errors"):
        raise ValueError("Input must be a completed full feature extraction without errors")
    table = directory / "features.csv"
    with table.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(r["split"], r["image_name"]) for r in rows]
    if len(rows) != report["success"] or len(set(keys)) != len(keys):
        raise ValueError("Feature row count or identifiers are invalid")
    if Counter(r["split"] for r in rows) != Counter(report["dataset_counts"]):
        raise ValueError("Split counts do not match the extraction report")
    for row in rows:
        value = row["dhash64"]
        if len(value) != 16 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Malformed 64-bit hash")
        if row["label"] not in {"", "0", "1"}:
            raise ValueError("Invalid known label")
    return report, rows, hashlib.sha256(table.read_bytes()).hexdigest()


def safe_source(root, row):
    relative = Path(row["relative_path"])
    if relative.is_absolute():
        raise ValueError("Source path must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Source path escapes data root")
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (int(row["size_bytes"]), int(row["mtime_ns"])):
        raise ValueError("Source file changed since feature extraction")
    return path


def make_pair_preview(root, left, right, distance):
    card = Image.new("RGB", (960, 410), "white")
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default(size=16)
    for side, row in enumerate((left, right)):
        with Image.open(safe_source(root, row)) as source:
            picture = ImageOps.exif_transpose(source).convert("RGB")
        picture.thumbnail((470, 335), Image.Resampling.LANCZOS)
        card.paste(picture, (side * 480 + (480 - picture.width) // 2, 0))
        draw.text((side * 480 + 6, 343), f"{row['split']}: {row['image_name']}", fill="black", font=font)
        draw.text((side * 480 + 6, 367), f"Known label: {row['label'] or 'unknown'}", fill="black", font=font)
    draw.text((6, 390), f"dHash distance: {distance}/64 | Candidate only; human review required", fill="black", font=font)
    return card


def run(directory, output, left_split="train", right_split="valid", radius=4, max_pairs=100000, preview_count=20):
    if left_split == right_split:
        raise ValueError("This tool searches between different splits")
    if preview_count < 0:
        raise ValueError("preview-count must not be negative")
    directory, output = Path(directory).resolve(), Path(output).resolve()
    source_report, rows, digest = load_features(directory)
    root = Path(source_report["data_root"]).resolve()
    if output == root or root in output.parents or output == directory:
        raise ValueError("Do not overwrite source data or feature directory")
    existing_report = output / "report.json"
    if (output / "review.csv").exists() and existing_report.exists():
        previous = json.loads(existing_report.read_text(encoding="utf-8"))
        if (previous.get("scope"), previous.get("radius"), previous.get("max_pairs"), previous.get("feature_csv_sha256")) != (
                f"{left_split}-{right_split}", radius, max_pairs, digest):
            raise ValueError("Existing review has a different search configuration; use a new output directory")
    left = [r for r in rows if r["split"] == left_split]
    right = [r for r in rows if r["split"] == right_split]
    if not left or not right:
        raise ValueError("Missing split in feature table")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def progress(done, total, matches):
        print(f"[search] {done}/{total} unique {right_split} hashes; candidates={matches}", flush=True)

    print(f"[index] {len(left)} {left_split} vs {len(right)} {right_split}; radius={radius}/64", flush=True)
    result = search(left, right, radius, max_pairs, progress)
    print(f"[search complete] {result['candidate_count']} candidate pairs; exporting {len(result['pairs'])}", flush=True)
    pair_fields = ["pair_id", "distance", "left_split", "left_image", "left_label", "right_split", "right_image", "right_label", "different_known_labels"]
    pair_rows = []
    for index, (distance, i, j) in enumerate(result["pairs"], start=1):
        a, b = left[i], right[j]
        pair_rows.append(dict(zip(pair_fields, [f"pair-{index:05d}", distance, a["split"], a["image_name"], a["label"],
                                               b["split"], b["image_name"], b["label"],
                                               bool(a["label"] and b["label"] and a["label"] != b["label"])])))
    with (output / "candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(pair_rows)
    (output / "previews").mkdir(exist_ok=True)
    preview_errors, review_rows, cards, html_cards = [], [], [], []
    preview_indices, seen_left, seen_right = [], set(), set()
    for index, (_, i, j) in enumerate(result["pairs"]):
        if len(preview_indices) >= preview_count:
            break
        if i not in seen_left and j not in seen_right:
            preview_indices.append(index)
            seen_left.add(i)
            seen_right.add(j)
    if len(preview_indices) < min(preview_count, len(pair_rows)):
        chosen = set(preview_indices)
        for index in range(len(pair_rows)):
            if len(preview_indices) >= preview_count:
                break
            if index not in chosen:
                preview_indices.append(index)
    for index in preview_indices:
        pair = pair_rows[index]
        distance, i, j = result["pairs"][index]
        relative = f"previews/{pair['pair_id']}.jpg"
        try:
            card = make_pair_preview(root, left[i], right[j], distance)
            card.save(output / relative, quality=92)
            if len(cards) < 4:
                cards.append(card.copy())
            card.close()
            html_cards.append(f"<section><h2>{html.escape(pair['pair_id'])} · dHash {distance}/64</h2>"
                              f"<img src='{html.escape(relative)}' alt='candidate pair'><p>候选，尚待人工复核。</p></section>")
            preview_status = "ready"
        except Exception as exc:
            preview_errors.append({"pair_id": pair["pair_id"], "error": str(exc)})
            preview_status = "failed"
        review_rows.append({**pair, "preview": relative, "preview_status": preview_status,
                            "review_status": "待复核", "reviewer": "", "review_notes": ""})
    if cards:
        sheet = Image.new("RGB", (960, 410 * len(cards)), "white")
        for i, card in enumerate(cards):
            sheet.paste(card, (0, i * 410))
            card.close()
        sheet.save(output / "contact-sheet.jpg", quality=92)
    # Never overwrite human decisions in a previously completed review sheet.
    review_path = output / "review.csv"
    if review_path.exists():
        review_path = output / "review-template.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields + ["preview", "preview_status", "review_status", "reviewer", "review_notes"])
        writer.writeheader()
        writer.writerows(review_rows)
    page = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>近重复候选复核</title>"
            "<style>body{font:16px sans-serif;max-width:1000px;margin:30px auto;padding:16px;background:#f4f5f7}"
            "section{background:white;padding:16px;margin:20px 0}img{max-width:100%;height:auto}</style>"
            "<h1>近重复候选：人工复核</h1><p>仅按亮度结构指纹筛选。相似或相同指纹不证明同图或同一患者；不得据此自动删图、改标签。</p>"
            "<p>请在 review.csv 中填写：确认近重复 / 不同图 / 待进一步核实，并记录依据。此页面不会自动保存判定。</p>"
            + "".join(html_cards) + "</html>")
    (output / "review.html").write_text(page, encoding="utf-8")
    truncated = result["candidate_count"] > len(result["pairs"])
    report = {"status": "SEARCH_COMPLETE", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": f"{left_split}-{right_split}", "left_count": len(left), "right_count": len(right),
              "radius": radius, "hash_bits": 64, "feature_csv_sha256": digest,
              "feature_signature": source_report["feature_signature"],
              **{k: v for k, v in result.items() if k != "pairs"},
              "exported_pairs": len(pair_rows), "export_truncated": truncated,
              "max_pairs": max_pairs, "preview_requested": min(preview_count, len(pair_rows)),
              "preview_selection": "distance-ranked, prioritizing distinct left and right image IDs; not a random sample",
              "preview_errors": preview_errors, "review_file": review_path.name,
              "confirmed_duplicate_pairs": None, "human_review_status": "NOT_REVIEWED",
              "elapsed_seconds": round(time.monotonic() - started, 2),
              "limitations": ["Only this cross-split pair and <=radius dHash candidates searched.",
                              "Radius is exploratory, not a validated duplicate threshold.",
                              "Same or close dHash can represent different images; human review is mandatory.",
                              "Rotation/crop/re-encoding effects may lie outside the radius; no claim of all near duplicates removed.",
                              "Different known labels are candidate metadata, not confirmed label errors.",
                              "No source file, label or official split was changed."]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 跨划分近重复候选初筛", "", f"检索范围：{report['scope']}；指纹距离上限：{radius}/64（探索性参数，未经复核校准）。",
             f"候选对：{result['candidate_count']:,}；导出：{len(pair_rows):,}；导出截断：{truncated}。",
             f"不同已知标签的候选对：{result['different_known_label_candidates']:,}（不代表错标）。",
             "人工确认重复对：尚未判定。", "", "| 指纹距离 | 候选对数 |", "| --- | ---: |"]
    lines += [f"| {d} | {n} |" for d, n in result["distance_histogram"].items()]
    lines += ["", f"预览错误：{len(preview_errors)}；总耗时：{report['elapsed_seconds']} 秒。",
              "按距离升序排列，最多导出 max_pairs；若截断，保留排序最靠前的候选。",
              f"先查看 review.html 或 contact-sheet.jpg，在 {review_path.name} 中记录人工判断及依据。",
              "预览按距离优先并尽量选择不同左右样本，不是随机抽样；复核表只包含这些预览对，不代表完整候选审查。",
              "不自动删图、改标签或调整官方划分，不修改测试预测。",
              "dHash 描述亮度结构，相似不能证明同图/同患者；裁剪、旋转等情况可能漏检。",
              "当前仅检索指定跨划分范围，不包含各划分内部或其他组合。",
              "重复运行会重建候选和预览，但保留已有 review.csv 的人工结论，另写 review-template.csv。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"RESULT: SEARCH_COMPLETE; candidates={result['candidate_count']}; "
          f"previews={min(preview_count, len(pair_rows)) - len(preview_errors)}; confirmed=NOT_REVIEWED", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=PROJECT / "outputs/image-features")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/near-duplicates/train-valid")
    parser.add_argument("--left-split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--right-split", choices=["train", "valid", "test"], default="valid")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=100000)
    parser.add_argument("--preview-count", type=int, default=20)
    args = parser.parse_args()
    report = run(args.features_dir, args.output_dir, args.left_split, args.right_split,
                 args.radius, args.max_pairs, args.preview_count)
    return int(bool(report["preview_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())

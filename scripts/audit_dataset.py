"""Audit CSVs, image headers and optional ZIP inventory. Does not edit data."""

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from glaucoma.data import load_records  # noqa: E402


def inspect_image(path):
    from PIL import Image
    try:
        with Image.open(path) as image:
            return {"width": image.width, "height": image.height, "mode": image.mode}
    except Exception as exc:
        return {"error": f"{path.name}: {type(exc).__name__}: {exc}"}


def audit_split(root, split, samples, workers):
    from PIL import Image
    rows = load_records(root, split)
    labels = collections.Counter(r.label for r in rows)
    ages = [r.age for r in rows if r.age is not None]
    print(f"Checking {split}: {len(rows)} image headers...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        headers = list(pool.map(inspect_image, (r.path for r in rows)))
    errors = [h["error"] for h in headers if "error" in h]
    valid = [h for h in headers if "error" not in h]
    # Include both classes in the decoding sample even when positives are rare.
    rng = random.Random(2026)
    sampled = []
    for label in ([0, 1] if split != "test" else [None]):
        group = [r for r in rows if r.label == label]
        sampled.extend(rng.sample(group, min(samples, len(group))))
    for row in sampled:
        try:
            with Image.open(row.path) as image:
                image.load()
        except Exception as exc:
            errors.append(f"Decode {row.image_name}: {type(exc).__name__}: {exc}")
    return {
        "rows": len(rows), "negative": labels.get(0, 0) if split != "test" else None,
        "positive": labels.get(1, 0) if split != "test" else None,
        "positive_rate": labels.get(1, 0) / len(rows) if split != "test" else None,
        "missing_age": len(rows) - len(ages),
        "age_min": min(ages) if ages else None, "age_max": max(ages) if ages else None,
        "age_median": statistics.median(ages) if ages else None,
        "filename_case_differences": sum(r.image_name != r.path.name for r in rows),
        "modes": dict(collections.Counter(h["mode"] for h in valid)),
        "width_min": min((h["width"] for h in valid), default=None),
        "width_max": max((h["width"] for h in valid), default=None),
        "height_min": min((h["height"] for h in valid), default=None),
        "height_max": max((h["height"] for h in valid), default=None),
        "headers_checked": len(headers), "images_fully_decoded": len(sampled),
        "error_count": len(errors), "errors": errors[:20],
    }


def audit_archive(root, archive):
    missing, wrong_size = [], []
    groups = collections.defaultdict(list)
    count = 0
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            if info.is_dir():
                continue
            parts = info.filename.split("/")
            if parts[0] != root.name or ".." in parts:
                raise ValueError("ZIP top-level directory must match data_root directory name")
            path = root.joinpath(*parts[1:])
            count += 1
            if not path.is_file():
                missing.append(info.filename)
            elif path.stat().st_size != info.file_size:
                wrong_size.append(info.filename)
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                groups[(info.CRC, info.file_size)].append(info.filename)
    candidates = [names for names in groups.values() if len(names) > 1]
    return {
        "files_checked": count, "missing_count": len(missing),
        "size_mismatch_count": len(wrong_size),
        "missing_examples": missing[:20], "size_mismatch_examples": wrong_size[:20],
        "same_crc_and_size_candidate_groups": len(candidates),
        "cross_split_candidate_groups": sum(len({n.split('/')[1] for n in group}) > 1 for group in candidates),
        "scope": "ZIP metadata and extracted sizes only; not full extracted-file CRC verification. Duplicate candidates are not confirmed duplicates; near-duplicates are not checked.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/paths.local.json")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/data-audit")
    args = parser.parse_args()
    if args.samples_per_class < 1 or args.workers < 1:
        parser.error("samples-per-class and workers must be positive")
    root = args.data_root
    if root is None:
        if not args.config.is_file():
            parser.error("Provide --data-root or copy configs/paths.example.json to configs/paths.local.json")
        root = Path(json.loads(args.config.read_text(encoding="utf-8-sig"))["data_root"])
    root = root.resolve()
    output = args.output_dir.resolve()
    if output == root or root in output.parents:
        parser.error("Reports must be written outside the source dataset")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "data_root": str(root), "splits": {}, "errors": []}
    for split in ("train", "valid", "test"):
        try:
            report["splits"][split] = audit_split(root, split, args.samples_per_class, args.workers)
        except Exception as exc:
            report["errors"].append(f"{split}: {type(exc).__name__}: {exc}")
    if args.archive:
        try:
            report["archive"] = audit_archive(root, args.archive)
        except Exception as exc:
            report["errors"].append(f"ZIP: {type(exc).__name__}: {exc}")
    failures = len(report["errors"]) + sum(s["error_count"] for s in report["splits"].values())
    archive = report.get("archive", {})
    failures += archive.get("missing_count", 0) + archive.get("size_mismatch_count", 0)
    report["status"] = "PASS_WITH_NOTES" if failures == 0 else "FAIL"
    report["limitations"] = ["Only sampled images are fully decoded; all image headers are inspected.", "No patient identifiers are present in the required CSV schema; patient overlap is not assessed.", "Missing age is preserved; no imputation or source changes are performed."]
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# 数据集检查报告", "", f"状态：{report['status']}", "", f"数据目录：`{root}`", "",
             "| 数据集 | 样本数 | 阴性 | 阳性 | 阳性占比 | 年龄缺失 | 文件名大小写差异 | 图片读取错误 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split, s in report["splits"].items():
        rate = f"{s['positive_rate']:.4%}" if s["positive_rate"] is not None else "无标签"
        lines.append(f"| {split} | {s['rows']} | {s['negative'] if s['negative'] is not None else '-'} | {s['positive'] if s['positive'] is not None else '-'} | {rate} | {s['missing_age']} | {s['filename_case_differences']} | {s['error_count']} |")
    lines += ["", "## 年龄与图像尺寸", "", "| 数据集 | 非缺失年龄范围 | 图片宽度范围 | 图片高度范围 | 完整解码抽样数 |", "| --- | --- | --- | --- | ---: |"]
    for split, s in report["splits"].items():
        lines.append(f"| {split} | {s['age_min']} - {s['age_max']} | {s['width_min']} - {s['width_max']} | {s['height_min']} - {s['height_max']} | {s['images_fully_decoded']} |")
    lines += ["", "年龄极端值需核实含义；范围统计不代表已确认数值合理，不自动删改。", "", "## 检查范围与下一步", "", "- 全量核对 CSV、图片对应关系、标签合法性和图片头信息；只对抽样图片进行完整解码。",
              "- 文件名大小写差异由读取代码映射处理，保留 CSV 原始图片标识。",
              "- 缺失年龄保留为空；如后续使用年龄特征，应仅使用训练集估计填补参数。",
              "- 未核对患者级重叠，未进行近似重复图片检测。", "- 详细尺寸、抽样数、错误与压缩包对照结果见同目录 report.json。",
              "- 本报告保存在 Git 忽略目录中。"]
    if archive:
        lines += ["", f"ZIP 对照文件数：{archive['files_checked']}；缺失：{archive['missing_count']}；大小不一致：{archive['size_mismatch_count']}。",
                  f"ZIP 内相同 CRC 与大小的候选组：{archive['same_crc_and_size_candidate_groups']}。这不是全量文件内容校验或近似重复检测。"]
    if report["errors"]:
        lines += ["", "## 错误", ""] + [f"- {error}" for error in report["errors"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())

"""Find byte-identical images without changing the official dataset.

Unique file sizes cannot be byte-identical. Hash only shared-size candidates,
then verify matching SHA-256 groups with byte comparisons. Cache successful
hashes by source root, relative path, size and mtime for interrupted runs.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from glaucoma.data import load_records  # noqa: E402

CHUNK = 1024 * 1024


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def fingerprint(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def hash_file(item):
    try:
        path = Path(item["path"])
        expected = item["size"], item["mtime_ns"]
        if fingerprint(path) != expected:
            raise RuntimeError("File changed after inventory")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(CHUNK):
                digest.update(chunk)
        if fingerprint(path) != expected:
            raise RuntimeError("File changed during hashing")
        return item, digest.hexdigest(), None
    except Exception as exc:
        return item, None, f"{type(exc).__name__}: {exc}"


def same_bytes(first, second):
    with Path(first["path"]).open("rb") as left, Path(second["path"]).open("rb") as right:
        while True:
            a, b = left.read(CHUNK), right.read(CHUNK)
            if a != b:
                return False
            if not a:
                return True


def run_audit(root, output, workers=4):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("Write reports outside the source dataset")
    if workers < 1:
        raise ValueError("workers must be positive")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"started_at_utc": utc_now(), "data_root": str(root), "errors": []}
    errors, inventory = report["errors"], []
    split_counts = {}
    last_progress = 0

    def progress(stage, done, total=None):
        elapsed = time.monotonic() - started
        snapshot = {"stage": stage, "completed": done, "total": total,
                    "elapsed_seconds": round(elapsed, 1), "updated_at_utc": utc_now()}
        save_json(output / "progress.json", snapshot)
        print(f"[{stage}] {done}" + (f"/{total}" if total is not None else "")
              + f"; elapsed {elapsed:.0f}s", flush=True)

    for split in ("train", "valid", "test"):
        progress(f"load_csv_{split}", 0)
        rows = load_records(root, split)
        split_counts[split] = len(rows)
        for row in rows:
            try:
                size, mtime = fingerprint(row.path)
                inventory.append({"split": split, "image_name": row.image_name,
                                  "label": row.label, "path": str(row.path),
                                  "relative_path": row.path.relative_to(root).as_posix(),
                                  "size": size, "mtime_ns": mtime})
            except OSError as exc:
                errors.append({"split": split, "image_name": row.image_name, "error": str(exc)})
            if time.monotonic() - last_progress >= 5:
                progress("inventory", len(inventory))
                last_progress = time.monotonic()
        progress(f"inventory_{split}_complete", len(rows), len(rows))

    by_size = defaultdict(list)
    for item in inventory:
        by_size[item["size"]].append(item)
    candidates = [item for group in by_size.values() if len(group) > 1 for item in group]
    report.update({"split_counts": split_counts, "total_files": sum(split_counts.values()),
                   "files_statted": len(inventory), "candidate_files": len(candidates),
                   "unique_size_files": sum(len(g) == 1 for g in by_size.values()),
                   "candidate_bytes": sum(i["size"] for i in candidates)})
    print(f"Size filter: {len(candidates)} candidates; "
          f"{report['unique_size_files']} unique-size files need no hashing; "
          f"{report['candidate_bytes'] / 1024**3:.2f} GiB candidate content", flush=True)

    cache_path = output / "hash-cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
                if entry.get("root") == str(root):
                    cache[(entry["relative_path"], entry["size"], entry["mtime_ns"])] = entry["sha256"]
            except (ValueError, KeyError):
                pass  # An interrupted last cache write is simply recomputed.
    hashed = defaultdict(list)
    pending, reused = [], 0
    for item in candidates:
        key = item["relative_path"], item["size"], item["mtime_ns"]
        digest = cache.get(key)
        if digest:
            hashed[(item["size"], digest)].append(item)
            reused += 1
        else:
            pending.append(item)
    progress("hash", reused, len(candidates))
    completed = reused
    hashes_computed = 0
    with cache_path.open("a", encoding="utf-8") as journal, ThreadPoolExecutor(max_workers=workers) as pool:
        journal.write("\n")  # Separate a possibly interrupted last cache entry.
        for item, digest, error in pool.map(hash_file, pending):
            completed += 1
            if error:
                errors.append({"split": item["split"], "image_name": item["image_name"], "error": error})
            else:
                hashes_computed += 1
                hashed[(item["size"], digest)].append(item)
                entry = {"root": str(root), "relative_path": item["relative_path"],
                         "size": item["size"], "mtime_ns": item["mtime_ns"], "sha256": digest}
                journal.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if completed % 100 == 0 or time.monotonic() - last_progress >= 5:
                journal.flush()
                progress("hash", completed, len(candidates))
                last_progress = time.monotonic()
    progress("hash", completed, len(candidates))

    groups = []
    digest_groups = [(key, items) for key, items in hashed.items() if len(items) > 1]
    for number, ((size, digest), items) in enumerate(sorted(digest_groups), start=1):
        try:
            for item in items:
                if fingerprint(Path(item["path"])) != (item["size"], item["mtime_ns"]):
                    raise RuntimeError("File changed before byte comparison")
            if not all(same_bytes(items[0], item) for item in items[1:]):
                raise RuntimeError("Matching hash but unequal file bytes; manual review required")
            for item in items:
                if fingerprint(Path(item["path"])) != (item["size"], item["mtime_ns"]):
                    raise RuntimeError("File changed during byte comparison")
            labels = {item["label"] for item in items if item["label"] is not None}
            groups.append({"group_id": len(groups) + 1, "sha256": digest, "file_bytes": size,
                           "cross_split": len({i["split"] for i in items}) > 1,
                           "label_conflict": len(labels) > 1, "items": items})
        except Exception as exc:
            errors.append({"sha256": digest, "error": f"{type(exc).__name__}: {exc}"})
        if number % 50 == 0 or time.monotonic() - last_progress >= 5:
            progress("verify_bytes", number, len(digest_groups))
            last_progress = time.monotonic()

    pairs = Counter()
    within = Counter()
    for group in groups:
        counts = Counter(i["split"] for i in group["items"])
        for split, count in counts.items():
            within[split] += int(count > 1)
        for first, second in (("train", "valid"), ("train", "test"), ("valid", "test")):
            pairs[f"{first}-{second}"] += int(first in counts and second in counts)
    report.update({"status": "COMPLETE" if not errors else "INCOMPLETE",
                   "hashes_reused": reused, "hashes_computed": hashes_computed,
                   "duplicate_groups": len(groups), "duplicate_files": sum(len(g["items"]) for g in groups),
                   "redundant_copies": sum(len(g["items"]) - 1 for g in groups),
                   "cross_split_groups": sum(g["cross_split"] for g in groups),
                   "within_split_groups": {s: within[s] for s in split_counts},
                   "cross_split_pair_groups": {p: pairs[p] for p in ("train-valid", "train-test", "valid-test")},
                   "label_conflict_groups": sum(g["label_conflict"] for g in groups),
                   "groups": groups, "finished_at_utc": utc_now(),
                   "elapsed_seconds": round(time.monotonic() - started, 1),
                   "scope": "Exact file bytes: size filter, SHA-256 and byte comparison. "
                            "No decoded-pixel/near-duplicate or patient identity check. "
                            "Cache assumes immutable data; changed size/mtime invalidates each entry."})
    save_json(output / "report.json", report)
    with (output / "duplicates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group_id", "sha256", "split", "image_name", "label", "file_bytes", "cross_split", "label_conflict"])
        for group in groups:
            for item in group["items"]:
                writer.writerow([group["group_id"], group["sha256"], item["split"], item["image_name"],
                                 item["label"], group["file_bytes"], group["cross_split"], group["label_conflict"]])
    lines = ["# 图片精确内容查重报告", "", f"状态：{report['status']}", "",
             f"图片总数：{report['total_files']:,}；大小已核对：{len(inventory):,}。",
             f"大小唯一、无需哈希：{report['unique_size_files']:,}；SHA-256 候选：{len(candidates):,}。",
             f"重复组：{len(groups)}；涉及图片：{report['duplicate_files']}；冗余副本：{report['redundant_copies']}。", "",
             "| 范围 | 重复组数 |", "| --- | ---: |"]
    lines.extend(f"| {s} 内部 | {within[s]} |" for s in split_counts)
    lines.extend(f"| {p} 跨划分 | {pairs[p]} |" for p in report["cross_split_pair_groups"])
    lines += ["", f"已知标签冲突组：{report['label_conflict_groups']}；错误：{len(errors)}。",
              "同一组可能计入多个范围，不能直接把表格相加作为总组数。",
              "", "## 范围与处理", "",
              "- 先核对全部文件大小，仅对相同大小候选计算 SHA-256，再逐字节比较相同哈希组。",
              "- 无重复仅表示未发现字节完全相同文件，不能排除重编码、缩放、裁剪或相似病例。",
              "- 未核实患者身份；不自动删图、改标签或重划官方训练/验证/测试集。",
              "- 跨划分重复或标签冲突交团队复核，不据此手工修改测试预测。",
              "- hash-cache.jsonl 支持同路径、大小和修改时间未变的数据恢复扫描；原始数据应保持不变。",
              "- 逐图清单和报告留在本地。"]
    if errors:
        lines += ["", "存在读取/变更错误，本次不能判定全量查重完成，详情见 report.json。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress("complete" if not errors else "incomplete", len(inventory), report["total_files"])
    print(f"RESULT: {report['status']}; duplicate groups={len(groups)}; "
          f"cross-split={report['cross_split_groups']}; label conflicts={report['label_conflict_groups']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/exact-duplicates")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = run_audit(args.data_root, args.output_dir, args.workers)
    return int(report["status"] != "COMPLETE")


if __name__ == "__main__":
    raise SystemExit(main())

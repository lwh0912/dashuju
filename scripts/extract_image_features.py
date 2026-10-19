"""Resumable, bounded-memory image feature extraction; source files are read-only."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
from PIL import Image

from pilot_image_features import features

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from glaucoma.data import load_records  # noqa: E402


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def exclusive_run(output):
    """OS lock is released even after a crash; do not delete the lock file."""
    handle = (output / "run.lock").open("a+b")
    try:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another extraction is using this output directory") from exc
        yield
    finally:
        handle.close()


def fingerprint(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def extract_one(item):
    try:
        expected = item["size_bytes"], item["mtime_ns"]
        if fingerprint(item["path"]) != expected:
            raise RuntimeError("Source changed after inventory")
        values, thumb = features(item["path"])
        thumb.close()  # Full runs store features, not 63,100 thumbnail files.
        if fingerprint(item["path"]) != expected:
            raise RuntimeError("Source changed during extraction")
        return item, values, None
    except Exception as exc:
        return item, None, f"{type(exc).__name__}: {exc}"


def bounded_results(items, workers):
    """Limit queued images so Ctrl+C need not drain thousands of queued jobs."""
    pool = ThreadPoolExecutor(max_workers=workers)
    iterator = iter(items)
    pending = set()
    try:
        for _ in range(workers * 2):
            item = next(iterator, None)
            if item is None:
                break
            pending.add(pool.submit(extract_one, item))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                item = next(iterator, None)
                if item is not None:
                    pending.add(pool.submit(extract_one, item))
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def run(root, output, workers=4, limit_per_split=0, checkpoint_every=50):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("Reports must be outside the source dataset")
    if workers < 1 or limit_per_split < 0 or checkpoint_every < 1:
        raise ValueError("Invalid workers, limit or checkpoint interval")
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output):
        return _run(root, output, workers, limit_per_split, checkpoint_every)


def _run(root, output, workers, limit_per_split, checkpoint_every):
    started = time.monotonic()
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "Pillow": Image.__version__}
    signature = hashlib.sha256((json.dumps(versions, sort_keys=True) + inspect.getsource(features)).encode()).hexdigest()
    report = {"status": "RUNNING", "started_at_utc": now(), "data_root": str(root),
              "scope": "sample" if limit_per_split else "full", "limit_per_split": limit_per_split,
              "workers": workers, "versions": versions, "feature_signature": signature,
              "checkpoint_every": checkpoint_every,
              "limitations": ["Feature extraction only; no near-duplicate search or medical quality diagnosis.",
                              "dHash is coarse, not crop/rotation invariant; candidates require review.",
                              "Grayscale thumbnail statistics include dark borders.",
                              "Cache assumes immutable source files; size/mtime and implementation/version changes invalidate it."]}
    save_json(output / "report.json", report)
    errors, inventory, populations = [], [], {}
    progress_path = output / "progress.json"

    def progress(stage, **values):
        data = {"stage": stage, "updated_at_utc": now(), "elapsed_seconds": round(time.monotonic() - started, 1), **values}
        save_json(progress_path, data)
        return data

    try:
        for split in ("train", "valid", "test"):
            print(f"[load] {split}: checking CSV and filenames...", flush=True)
            progress("inventory", split=split, inventoried=len(inventory))
            records = load_records(root, split)
            populations[split] = len(records)
            if limit_per_split:
                records = records[:limit_per_split]
            last_log = time.monotonic()
            for row in records:
                try:
                    size, mtime = fingerprint(row.path)
                    inventory.append({"split": split, "image_name": row.image_name, "label": row.label,
                                      "path": row.path, "relative_path": row.path.relative_to(root).as_posix(),
                                      "size_bytes": size, "mtime_ns": mtime})
                except OSError as exc:
                    errors.append({"split": split, "image_name": row.image_name, "error": str(exc)})
                if time.monotonic() - last_log >= 5:
                    progress("inventory", split=split, inventoried=len(inventory))
                    print(f"[inventory] {len(inventory)} files", flush=True)
                    last_log = time.monotonic()
            print(f"[inventory] {split}: {len(records)} selected", flush=True)
    except BaseException:
        report.update(status="INTERRUPTED_OR_FAILED", finished_at_utc=now())
        save_json(output / "report.json", report)
        raise

    expected = sum(min(limit_per_split, n) if limit_per_split else n for n in populations.values())
    report.update(dataset_counts=populations, dataset_total=sum(populations.values()), selected=expected)
    connection = sqlite3.connect(output / "feature-cache.sqlite3")
    connection.execute("CREATE TABLE IF NOT EXISTS cache (root TEXT, path TEXT, size INTEGER, mtime INTEGER, signature TEXT, payload TEXT, PRIMARY KEY(root,path))")
    cache = {path: (size, mtime, sig, payload) for path, size, mtime, sig, payload in
             connection.execute("SELECT path,size,mtime,signature,payload FROM cache WHERE root=?", (str(root),))}
    results, pending = {}, []
    for item in inventory:
        cached = cache.get(item["relative_path"])
        if cached and cached[:3] == (item["size_bytes"], item["mtime_ns"], signature):
            try:
                values = json.loads(cached[3])
                if all(k in values for k in ("dhash64", "width", "height", "brightness_mean", "contrast_std", "dark_fraction_below_10", "feature_seconds")):
                    results[item["relative_path"]] = values
                    continue
            except ValueError:
                pass
        pending.append(item)
    reused = len(results)
    computed, attempted = 0, 0
    feature_started = time.monotonic()
    last_log = feature_started
    print(f"[resume] {reused}/{expected} reused; {len(pending)} images to extract", flush=True)

    def checkpoint(stage="features"):
        connection.commit()
        seconds = time.monotonic() - feature_started
        rate = attempted / seconds if attempted and seconds else None
        eta = (len(pending) - attempted) / rate if rate else None
        progress(stage, total=expected, success=len(results), reused=reused,
                 computed=computed, errors=len(errors), images_per_second=rate,
                 estimated_remaining_seconds=eta)
        eta_text = f"{eta / 60:.1f} min" if eta is not None else "measuring"
        print(f"[{stage}] {len(results)}/{expected}; reused={reused}; "
              f"errors={len(errors)}; ETA={eta_text}", flush=True)

    try:
        checkpoint()
        for item, values, error in bounded_results(pending, workers):
            attempted += 1
            if error:
                errors.append({"split": item["split"], "image_name": item["image_name"], "error": error})
            else:
                computed += 1
                results[item["relative_path"]] = values
                connection.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?)",
                                   (str(root), item["relative_path"], item["size_bytes"], item["mtime_ns"], signature, json.dumps(values)))
            if attempted % checkpoint_every == 0 or time.monotonic() - last_log >= 5:
                checkpoint()
                last_log = time.monotonic()
        checkpoint("export")
    except BaseException:
        checkpoint("interrupted")
        report.update(status="INTERRUPTED", success=len(results), reused=reused, computed=computed,
                      errors=errors, finished_at_utc=now())
        save_json(output / "report.json", report)
        raise
    finally:
        connection.close()

    columns = ["split", "image_name", "label", "relative_path", "size_bytes", "mtime_ns", "width", "height",
               "dhash64", "brightness_mean", "contrast_std", "dark_fraction_below_10", "feature_seconds"]
    temporary_csv = output / "features.csv.tmp"
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in inventory:
            if item["relative_path"] in results:
                writer.writerow({**{k: item[k] for k in columns if k in item}, **results[item["relative_path"]]})
    temporary_csv.replace(output / "features.csv")
    complete = len(results) == expected and not errors
    status = ("COMPLETE_SAMPLE" if limit_per_split else "COMPLETE") if complete else "INCOMPLETE"
    report.update(status=status, success=len(results), reused=reused, computed=computed,
                  errors=errors, finished_at_utc=now(), elapsed_seconds=round(time.monotonic() - started, 1),
                  selected_by_split={s: min(limit_per_split, n) if limit_per_split else n for s, n in populations.items()})
    save_json(output / "report.json", report)
    save_json(output / "errors.json", errors)
    lines = ["# 图像特征提取报告", "", f"状态：{status}；范围：{report['scope']}。",
             f"数据集总数：{report['dataset_total']:,}；本轮选择：{expected:,}；成功：{len(results):,}；错误：{len(errors)}。",
             f"缓存复用：{reused:,}；本轮新提取：{computed:,}；本轮耗时：{report['elapsed_seconds']} 秒。", "",
             "原始图片和标签未修改；本脚本不输出整套缩略图，只保存特征及恢复缓存。",
             "当前尚未做全量近重复候选检索，不能用此报告宣称没有近重复或患者重叠。",
             "亮度和对比度为含黑边的灰度缩略图统计，不是医学质量判定。",
             "中断后运行相同命令及输出目录恢复；固定数据、算法和 Python/NumPy/Pillow 版本。",
             "features.csv 中 dhash64 按字符串读取，保留前导零。详细配置与错误见 report.json。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress("complete" if complete else "incomplete", total=expected, success=len(results), errors=len(errors))
    print(f"RESULT: {status}; {len(results)}/{expected}; errors={len(errors)}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/image-features")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-per-split", type=int, default=0, help="Smoke check only; 0 selects all images")
    args = parser.parse_args()
    return int(run(args.data_root, args.output_dir, args.workers, args.limit_per_split)["status"] not in {"COMPLETE", "COMPLETE_SAMPLE"})


if __name__ == "__main__":
    raise SystemExit(main())

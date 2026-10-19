"""Small diagnostic of human-reviewed pairs, not an automatic deduplication rule.

Uses existing container OpenCV: mutual SIFT ratio matches, RANSAC geometry and
overlap correlation. Even strong geometry is not proof of the same acquisition.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from find_near_duplicates import safe_source


def read_gray(path, max_side=1000):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(image).copy()


def descriptor(image):
    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    mask = np.uint8(image > 15) * 255
    mask = cv2.erode(mask, np.ones((9, 9), dtype=np.uint8))
    return cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02).detectAndCompute(normalized, mask)


def compare(left, right, ratio=0.7):
    cv2.setRNGSeed(2026)
    key_left, des_left = descriptor(left)
    key_right, des_right = descriptor(right)
    metrics = {"keypoints_left": len(key_left), "keypoints_right": len(key_right),
               "mutual_ratio_matches": 0, "ransac_inliers": 0,
               "inlier_fraction": None, "left_inlier_hull_fraction": None,
               "right_inlier_hull_fraction": None, "overlap_fraction_of_right": None,
               "overlap_gray_correlation": None, "geometry_status": "INSUFFICIENT_MATCHES"}
    matches, mask, homography = [], None, None
    if des_left is not None and des_right is not None and len(des_left) >= 2 and len(des_right) >= 2:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        forward = [pair[0] for pair in matcher.knnMatch(des_left, des_right, k=2)
                   if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
        reverse = {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(des_right, des_left, k=2)
                   if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance}
        matches = [match for match in forward if reverse.get(match.trainIdx) == match.queryIdx]
        metrics["mutual_ratio_matches"] = len(matches)
        if len(matches) >= 4:
            a = np.float32([key_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            b = np.float32([key_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(a, b, cv2.RANSAC, 3.0, maxIters=5000, confidence=0.995)
            if homography is not None and mask is not None and np.isfinite(homography).all():
                inliers = mask.ravel().astype(bool)
                count = int(inliers.sum())
                metrics.update(geometry_status="MODEL_FITTED_NOT_A_DUPLICATE_VERDICT", ransac_inliers=count,
                               inlier_fraction=count / len(matches))
                if count >= 3:
                    metrics["left_inlier_hull_fraction"] = float(cv2.contourArea(cv2.convexHull(a[inliers])) / left.size)
                    metrics["right_inlier_hull_fraction"] = float(cv2.contourArea(cv2.convexHull(b[inliers])) / right.size)
                height, width = right.shape
                warped = cv2.warpPerspective(left, homography, (width, height))
                warped_mask = cv2.warpPerspective(np.uint8(left > 15), homography, (width, height), flags=cv2.INTER_NEAREST)
                overlap = (warped_mask > 0) & (right > 15)
                metrics["overlap_fraction_of_right"] = float(overlap.mean())
                if overlap.sum() >= 100:
                    x, y = warped[overlap].astype(float), right[overlap].astype(float)
                    if x.std() > 1e-6 and y.std() > 1e-6:
                        metrics["overlap_gray_correlation"] = float(np.corrcoef(x, y)[0, 1])
            else:
                metrics["geometry_status"] = "NO_FINITE_MODEL"
    visible = matches
    visible_mask = None
    if mask is not None and len(mask) == len(matches):
        visible = [m for m, keep in zip(matches, mask.ravel()) if keep]
    visible = sorted(visible, key=lambda m: m.distance)[:40]
    # Draw only inliers if a model exists; otherwise these are tentative matches.
    preview = cv2.drawMatches(left, key_left, right, key_right, visible, None,
                             matchColor=(0, 200, 0), singlePointColor=None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return metrics, preview


def run(root, review_path, output, feature_path=None):
    root, review_path, output = Path(root).resolve(), Path(review_path).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside source data")
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    reviews = json.loads(review_path.read_text(encoding="utf-8"))["reviews"]
    wanted = {(split, review[field]) for review in reviews
              for split, field in (("train", "left_image"), ("valid", "right_image"))}
    feature_path = Path(feature_path) if feature_path else PROJECT / "outputs/image-features/features.csv"
    lookup = {}
    with feature_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["split"], row["image_name"]
            if key in wanted:
                if key in lookup:
                    raise ValueError("Duplicate source identifier in feature table")
                lookup[key] = safe_source(root, row)
    if lookup.keys() != wanted:
        raise ValueError("Some reviewed images are absent from the feature table")
    print(f"[load] {len(lookup)} reviewed source paths checked", flush=True)
    results = []
    first_left = None
    for review in reviews:
        left = read_gray(lookup[("train", review["left_image"])])
        right = read_gray(lookup[("valid", review["right_image"])])
        if first_left is None:
            first_left = left.copy()
        metrics, preview = compare(left, right)
        target = output / f"{review['pair_id']}-matches.jpg"
        if not cv2.imwrite(str(target), preview):
            raise OSError(f"Cannot write {target}")
        results.append({**review, "diagnostic": metrics, "preview": target.name})
        print(f"[{review['pair_id']}] mutual={metrics['mutual_ratio_matches']}; "
              f"inliers={metrics['ransac_inliers']}; corr={metrics['overlap_gray_correlation']}", flush=True)
    controls = []
    # Known same-image brightness/resize control; never added to official data.
    if first_left is not None:
        changed = cv2.convertScaleAbs(first_left, alpha=0.8, beta=15)
        changed = cv2.resize(changed, None, fx=0.85, fy=0.85, interpolation=cv2.INTER_AREA)
        metrics, preview = compare(first_left, changed)
        cv2.imwrite(str(output / "synthetic-brightness-resize-control.jpg"), preview)
        controls.append({"name": "same_image_brightness_and_resize", "known_relation": "synthetic same-image variant",
                         "diagnostic": metrics, "preview": "synthetic-brightness-resize-control.jpg"})
    report = {"status": "DIAGNOSTIC_COMPLETE", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "opencv": cv2.__version__, "numpy": np.__version__, "method": "SIFT mutual ratio 0.7; RANSAC 3px at max side 1000; CLAHE",
              "reviews_source": str(review_path), "results": results, "controls": controls,
              "limitations": ["Only four reviewed pairs and one synthetic control, not a calibrated classifier.",
                              "Four random matches can fit a homography; inspect count, spread and aligned appearance together.",
                              "Low correspondence does not prove different patients or universally exclude image transformations.",
                              "Strong correspondence can reflect the same eye at different acquisitions; not proof of exact duplication.",
                              "Human annotations preserved, not overwritten by diagnostic metrics; source data unchanged."]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 四对人工复核候选的局部结构诊断", "", "这不是自动重复判定，也不是病情诊断。用户原始判断独立保留。", "",
             "| 预览行 | 用户判断 | 双向匹配 | RANSAC 内点 | 对齐灰度相关性 |", "| --- | --- | ---: | ---: | ---: |"]
    for r in results:
        d = r["diagnostic"]
        corr = d["overlap_gray_correlation"]
        corr_text = f"{corr:.3f}" if corr is not None else "无可靠模型/无法计算"
        lines.append(f"| {r['display_order']} | {r['user_comment']} | {d['mutual_ratio_matches']} | {d['ransac_inliers']} | {corr_text} |")
    lines += ["", "已知同图的亮度+缩放合成对照结果见 report.json，预览文件仅作方法诊断。",
              "特征匹配及几何模型参考 OpenCV 官方教程；参数仍需用更多已知正负对照校准。",
              "只验证这四对，不能外推全量候选的真假比例，不覆盖所有近重复变换。",
              "不因颜色差异、指纹相同或少量几何内点直接删图、改标签。",
              "来源：https://docs.opencv.org/4.12.0/d1/de0/tutorial_py_feature_homography.html"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("RESULT: DIAGNOSTIC_COMPLETE; human decisions unchanged", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, default=PROJECT / "outputs/near-duplicates/train-valid/user-review.json")
    parser.add_argument("--features", type=Path, default=PROJECT / "outputs/image-features/features.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/near-duplicates/train-valid/geometry-check")
    args = parser.parse_args()
    run(args.data_root, args.reviews, args.output_dir, args.features)


if __name__ == "__main__":
    main()

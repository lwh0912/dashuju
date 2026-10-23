"""Analyze labeled validation predictions; outputs contain IDs and stay local."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.glaucoma.data import load_records


def analyze(records, predictions, *, threshold=.5, allow_subset=False):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0,1]")
    truth = {r.image_name: r.label for r in records}
    if any(label not in (0, 1) for label in truth.values()):
        raise ValueError("Error analysis requires labeled data, never test predictions")
    rows, seen = [], set()
    for row in predictions:
        name = row["Image_name"]
        if name not in truth or name in seen:
            raise ValueError(f"Unknown or repeated prediction identifier: {name}")
        seen.add(name)
        if row["True_Label"] not in ("0", "1") or int(row["True_Label"]) != truth[name]:
            raise ValueError(f"Prediction label disagrees with official label: {name}")
        probability = float(row["Probability"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Probabilities must be finite and in [0,1]")
        label = truth[name]
        predicted = int(probability >= threshold)
        outcome = ("TP" if label else "FP") if predicted else ("FN" if label else "TN")
        rows.append(dict(Image_name=name, True_Label=label, Probability=probability,
                         Pred_Label=predicted, outcome=outcome))
    if not rows or (not allow_subset and seen != set(truth)):
        raise ValueError("Incomplete predictions; --allow-subset is required for explicit smoke subsets")
    counts = {kind: sum(r["outcome"] == kind for r in rows) for kind in ("TP", "FN", "FP", "TN")}
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels, probs = [r["True_Label"] for r in rows], [r["Probability"] for r in rows]
    both = len(set(labels)) == 2
    def ratio(a, b):
        return a / b if b else None
    summary = {"samples": len(rows), "official_samples": len(truth), "full_coverage": seen == set(truth),
        "threshold": threshold, "threshold_rule": "probability >= threshold", **counts,
        "recall": ratio(counts["TP"], counts["TP"] + counts["FN"]),
        "precision": ratio(counts["TP"], counts["TP"] + counts["FP"]),
        "auroc": float(roc_auc_score(labels, probs)) if both else None,
        "auprc": float(average_precision_score(labels, probs)) if both else None,
        "auprc_definition": "sklearn.average_precision_score",
        "metric_limitation": None if both else "Both classes are required; AUROC/AUPRC withheld"}
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--features", type=Path, help="Optional existing features.csv; never re-extract images")
    args = parser.parse_args()
    with args.predictions.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"Image_name", "True_Label", "Probability"} <= set(reader.fieldnames or []):
            raise ValueError("Expected Image_name,True_Label,Probability columns")
        summary, rows = analyze(load_records(args.data_root, "valid"), reader,
                               threshold=args.threshold, allow_subset=args.allow_subset)
    quality_keys = ["brightness_mean", "contrast_std", "dark_fraction_below_10"]
    if args.features:
        wanted = {r["Image_name"] for r in rows}
        features = {}
        with args.features.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["split"] == "valid" and row["image_name"] in wanted:
                    if row["image_name"] in features:
                        raise ValueError("Duplicate feature identity")
                    values = {key: float(row[key]) for key in quality_keys}
                    if not all(math.isfinite(v) for v in values.values()):
                        raise ValueError("Nonfinite cached quality feature")
                    features[row["image_name"]] = values
        if set(features) != wanted:
            raise ValueError("Missing validation quality features")
        for row in rows:
            row.update(features[row["Image_name"]])
        summary["quality_by_outcome"] = {
            kind: {key: sum(r[key] for r in rows if r["outcome"] == kind) / count for key in quality_keys}
            for kind in ("TP", "FN", "FP", "TN") if (count := summary[kind])}
        summary["features_sha256"] = hashlib.sha256(args.features.read_bytes()).hexdigest()
    summary["predictions_sha256"] = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    summary["limitations"] = "Quality summaries are descriptive; no causal, diagnostic or label-correction conclusion. Subsets are not full validation estimates."
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new empty output directory; existing results are not overwritten")
    for kind in ("FN", "FP"):
        selected = sorted((r for r in rows if r["outcome"] == kind), key=lambda r: r["Probability"], reverse=kind == "FP")
        fields = list(rows[0])
        with (args.output_dir / f"{kind.lower()}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    (args.output_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

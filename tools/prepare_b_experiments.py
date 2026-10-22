"""Materialize explicit B experiment configs into a new local output directory."""

import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/b-data-experiments.json"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new empty directory")
    for name, override in plan["variants"].items():
        if not name or Path(name).name != name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError("Invalid experiment name")
        cfg = copy.deepcopy(plan["base"])
        cfg.update(override)
        cfg.update(data_root=args.data_root, output_dir=str(args.output_dir / name),
                   experiment_name=name, data_version=plan["data_version"],
                   a_baseline_commit=plan["a_baseline_commit"])
        (args.output_dir / f"{name}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Prepared {len(plan['variants'])} configs; no training launched. Integrate A's loop first.")


if __name__ == "__main__":
    main()

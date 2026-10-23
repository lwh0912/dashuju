"""Read the official CSVs without changing source images or labels."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Record:
    image_name: str  # Keep the CSV spelling for submission identifiers.
    path: Path      # Use the actual disk spelling for Linux compatibility.
    age: float | None
    label: int | None


def image_index(directory: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = path.name.casefold()
        if key in result:
            raise ValueError(f"Ambiguous image names: {result[key].name}, {path.name}")
        result[key] = path
    return result


def load_records(root: Path, split: str) -> list[Record]:
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unknown split: {split}")
    root = Path(root)
    images = image_index(root / split)
    records = []
    seen = set()
    with (root / f"{split}.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"Image_name", "Age"}
        if split != "test":
            required.add("True_Label")
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError(f"{split}.csv has missing or duplicate columns: {fields}")
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{split}.csv line {line}: malformed row")
            name = row["Image_name"]
            if not name or name != name.strip() or "/" in name or "\\" in name:
                raise ValueError(f"{split}.csv line {line}: invalid image name")
            key = name.casefold()
            if key in seen:
                raise ValueError(f"{split}.csv line {line}: duplicate image {name}")
            seen.add(key)
            if key not in images:
                raise ValueError(f"{split}.csv line {line}: image missing: {name}")
            raw_age = row["Age"].strip()
            age = None if raw_age.lower() in {"", "nan", "na", "none", "null"} else float(raw_age)
            if age is not None and (not math.isfinite(age) or age < 0):
                raise ValueError(f"{split}.csv line {line}: invalid age")
            label = None
            if split != "test":
                if row["True_Label"].strip() not in {"0", "1"}:
                    raise ValueError(f"{split}.csv line {line}: label must be 0 or 1")
                label = int(row["True_Label"])
            records.append(Record(name, images[key], age, label))
    if not records:
        raise ValueError(f"{split}.csv is empty")
    extra = set(images) - seen
    if extra:
        raise ValueError(f"{split}: {len(extra)} images have no CSV row")
    return records

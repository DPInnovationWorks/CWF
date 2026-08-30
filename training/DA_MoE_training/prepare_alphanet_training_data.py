import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


TRAINING_DIR = Path(__file__).resolve().parent
MOE_PEFT_DIR = TRAINING_DIR.parent

DEFAULT_AGE_SOURCES = {
    "child": MOE_PEFT_DIR / "train_data/child_popular_science.json",
    "teens": MOE_PEFT_DIR / "train_data/teens_popular_science.json",
    "adult": MOE_PEFT_DIR / "train_data/adult_popular_science.json",
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def age_name_to_zh(age_name: str) -> str:
    return {
        "child": "儿童",
        "children": "儿童",
        "teens": "青少年",
        "teen": "青少年",
        "adult": "成年人",
        "adults": "成年人",
    }.get(age_name, age_name)


def load_age_records(age_name: str, path: Path, max_samples: int = 0) -> List[Dict[str, str]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")

    audience = age_name_to_zh(age_name)
    records = []
    for item in data:
        if not isinstance(item, dict):
            continue
        source = item.get("instruction") or item.get("original") or item.get("input") or ""
        target = item.get("output") or item.get("rewritten") or item.get("answer") or ""
        if not source or not target:
            continue
        records.append(
            {
                "instruction": (
                    f"请将下面的科普内容改写为适合{audience}阅读的表达，"
                    f"保持科学含义准确：\n{source.strip()}"
                ),
                "output": target.strip(),
                "age_group": age_name,
            }
        )
        if max_samples and len(records) >= max_samples:
            break
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Prepare age-specific popular-science data for AlphaNet training."
    )
    parser.add_argument("--child_data", type=Path, default=DEFAULT_AGE_SOURCES["child"])
    parser.add_argument("--teens_data", type=Path, default=DEFAULT_AGE_SOURCES["teens"])
    parser.add_argument("--adult_data", type=Path, default=DEFAULT_AGE_SOURCES["adult"])
    parser.add_argument("--output_dir", type=Path, default=TRAINING_DIR / "alphanet_data")
    parser.add_argument("--train_name", default="age_popular_science_train.json")
    parser.add_argument("--val_name", default="age_popular_science_val.json")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_per_age",
        type=int,
        default=0,
        help="Limit samples per age group for quick smoke tests. 0 means use all.",
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=1000,
        help="Total samples to keep after balancing and shuffling. 0 means use all.",
    )
    args = parser.parse_args()

    by_age = {
        "child": load_age_records("child", args.child_data, args.max_per_age),
        "teens": load_age_records("teens", args.teens_data, args.max_per_age),
        "adult": load_age_records("adult", args.adult_data, args.max_per_age),
    }

    records = []
    if args.total_samples:
        per_age = max(1, args.total_samples // len(by_age))
        remainder = args.total_samples % len(by_age)
        for idx, (age_name, age_records) in enumerate(by_age.items()):
            random.seed(args.seed + idx)
            random.shuffle(age_records)
            keep = per_age + (1 if idx < remainder else 0)
            records.extend(age_records[:keep])
    else:
        for age_records in by_age.values():
            records.extend(age_records)

    if not records:
        raise ValueError("No AlphaNet training records were loaded.")

    random.seed(args.seed)
    random.shuffle(records)

    val_size = int(len(records) * args.val_ratio)
    val_records = records[:val_size]
    train_records = records[val_size:]

    output_dir = args.output_dir.resolve()
    train_path = output_dir / args.train_name
    val_path = output_dir / args.val_name
    write_json(train_path, train_records)
    write_json(val_path, val_records)

    counts = {}
    for item in records:
        counts[item["age_group"]] = counts.get(item["age_group"], 0) + 1

    print(f"Total samples: {len(records)}")
    print(f"Age counts: {counts}")
    print(f"Train samples: {len(train_records)} -> {train_path}")
    print(f"Val samples: {len(val_records)} -> {val_path}")


if __name__ == "__main__":
    main()

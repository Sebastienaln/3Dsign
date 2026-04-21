"""Merge multiple CSV datasets, deduplicate by 'fr' column, shuffle."""

import csv
import random
import sys


def merge(files: list[str], output: str):
    seen = set()
    pairs = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fr = row.get("fr", "").strip()
                if fr and fr not in seen:
                    seen.add(fr)
                    pairs.append(row)
        print(f"{path}: {len(seen)} unique après merge")

    random.seed(42)
    random.shuffle(pairs)

    fieldnames = ["fr", "gloss", "category"]
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in pairs:
            writer.writerow({k: p.get(k, "") for k in fieldnames})

    print(f"\nTotal : {len(pairs)} paires → {output}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_datasets.py output.csv input1.csv input2.csv ...")
        sys.exit(1)
    merge(sys.argv[2:], sys.argv[1])

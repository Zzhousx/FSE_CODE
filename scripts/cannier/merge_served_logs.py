"""Merge clean CANNIER event shards into the published served.csv.gz log."""
import csv
import gzip
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cannier"
SOURCES = [OUT / "served_legacy.csv.gz"] + sorted(OUT.glob("served_resume_*.csv.gz"))
TARGET = OUT / "served.csv.gz"


def main():
    tmp = TARGET.with_suffix(".csv.gz.tmp")
    total = 0
    with gzip.open(tmp, "wt", newline="", encoding="utf-8", compresslevel=1) as dst:
        writer = csv.writer(dst)
        writer.writerow(["project", "seed", "r", "method", "item_id", "k", "y"])
        for source in SOURCES:
            with gzip.open(source, "rt", newline="", encoding="utf-8") as src:
                reader = csv.reader(src)
                next(reader, None)
                for row in reader:
                    writer.writerow(row)
                    total += 1
            print(f"merged {source.name}: cumulative_events={total}", flush=True)
    os.replace(tmp, TARGET)
    print(f"wrote {TARGET}: events={total}")


if __name__ == "__main__":
    main()

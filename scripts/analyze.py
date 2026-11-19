"""Measurements on one model and progress measures over a whole trajectory.

    python scripts/analyze.py snapshot authors     # authors' weights, epoch 40 000 -> results/authors_snapshot.json
    python scripts/analyze.py snapshot seed0       # our model, last checkpoint    -> results/seed0_snapshot.json
    python scripts/analyze.py trajectory authors   # progress measures at 401 points -> results/authors_trajectory.csv
    python scripts/analyze.py trajectory seed0
"""
import argparse
import csv
import json

from grokking import measure, paths


def write_snapshot(name):
    s = measure.snapshot(name)
    paths.RESULTS.mkdir(exist_ok=True)
    json.dump(s, open(paths.RESULTS / f"{name}_snapshot.json", "w"), indent=1, ensure_ascii=False)
    for k, v in s.items():
        print(f"{k}: {v}")


def write_trajectory(name):
    def progress(row):
        if row["epoch"] % 5000 == 0:
            print(name, row["epoch"], {k: f"{v:.3g}" for k, v in row.items() if k != "epoch"}, flush=True)

    rows, key = measure.trajectory(name, on_row=progress)
    paths.RESULTS.mkdir(exist_ok=True)
    with open(paths.RESULTS / f"{name}_trajectory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("key freqs", key)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["snapshot", "trajectory"])
    ap.add_argument("name", help="'authors' or a run name under runs/")
    args = ap.parse_args()
    (write_snapshot if args.what == "snapshot" else write_trajectory)(args.name)

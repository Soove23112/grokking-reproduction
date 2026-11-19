"""Training curves: ours (runs/<name>/metrics.csv) and the authors' (from their trajectory file).

The authors' file is 456 MB, so their curves are extracted once into the small
data/authors_curves.csv and read from there afterwards.
"""
import csv

import torch

from grokking import paths

NEVER = 10 ** 9      # stands for "did not happen within the run" (a censored epoch)


def read_csv(path):
    """-> {column: list of floats}."""
    rows = list(csv.DictReader(open(path)))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def our_curves(name="seed0"):
    """Our run: epoch, train_loss, test_loss, train_acc, test_acc, sum_sq_weights — every 100 epochs."""
    return read_csv(paths.run_dir(name) / "metrics.csv")


def authors_curves():
    """The authors' run every 100 epochs: epoch, train_loss, test_loss, sum_sq_weights."""
    if not paths.AUTHORS_CURVES.exists():
        rows = authors_curve_rows(torch.load(paths.AUTHORS_RUN, map_location="cpu", weights_only=True))
        with open(paths.AUTHORS_CURVES, "w", newline="") as f:
            csv.writer(f).writerows([["epoch", "train_loss", "test_loss", "sum_sq_weights"], *rows])
    return read_csv(paths.AUTHORS_CURVES)


def authors_curve_rows(run):
    """state_dicts[i] was saved after the step of epoch 100*i, and the authors log losses before the
    step (train every epoch, test every 10). Epoch 100*i therefore gets train_losses[100*i] and
    test_losses[10*i], matching our metrics.csv; the sum of squared weights comes from state_dicts[i]."""
    rows = []
    for i, sd in enumerate(run["state_dicts"]):
        sq = sum(v.double().pow(2).sum().item() for k, v in sd.items() if "mask" not in k and "IGNORE" not in k)
        rows.append([100 * i, run["train_losses"][100 * i], run["test_losses"][10 * i], sq])
    return rows


def first_epoch(curves, key, cond):
    """First epoch where cond(value) holds, or None."""
    for e, v in zip(curves["epoch"], curves[key]):
        if cond(v):
            return int(e)
    return None


def grok_epoch(curves):
    """First epoch with test accuracy above 99%, or NEVER."""
    e = first_epoch(curves, "test_acc", lambda v: v > 0.99)
    return NEVER if e is None else e


def median_of_seeds(values):
    """Median over seeds once every seed has finished, else None. NEVER sorts last, so two
    censored seeds out of three make the median NEVER."""
    if any(v is None for v in values):
        return None
    return sorted(values)[len(values) // 2]


def fmt_epoch(v):
    if v is None:
        return "—"
    return "> 40 000" if v == NEVER else f"{v:,}".replace(",", " ")

"""The two pre-registered sweeps (docs/preregistration.md) and how their runs are named."""
from grokking import curves as C
from grokking import paths

SEEDS = [0, 1, 2]
GRIDS = {
    "wd": ["0", "0.3", "1", "3"],
    "frac": ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"],
}
MAIN = {"wd": "1", "frac": "0.3"}     # with seed 0 this is the main run, runs/seed0


def run_name(grid, value, seed):
    return "seed0" if (value == MAIN[grid] and seed == 0) else f"{grid}{value}_s{seed}"


def finished_curves(name):
    """Curves of a finished run (40 000 epochs or an early stop), else None."""
    d = paths.run_dir(name)
    if not (d / "metrics.csv").exists():
        return None
    c = C.read_csv(d / "metrics.csv")
    return c if (d / "done.json").exists() or c["epoch"][-1] >= 39900 else None


def grid_curves(grid):
    """{(value, seed): curves or None} for every run of the grid."""
    return {(v, s): finished_curves(run_name(grid, v, s)) for v in GRIDS[grid] for s in SEEDS}


# ---------------------------------------------------------------- weight decay sweep
def wd_table(runs):
    """Markdown table: epoch of >99% test accuracy per seed and the median, for each λ."""
    lines = ["| λ | seed 0 | seed 1 | seed 2 | median |", "|---|---|---|---|---|"]
    show = lambda v: "running" if v is None else C.fmt_epoch(v)
    for wd in GRIDS["wd"]:
        vals = [None if runs[(wd, s)] is None else C.grok_epoch(runs[(wd, s)]) for s in SEEDS]
        lines.append(f"| {wd} | " + " | ".join(map(show, vals)) + f" | {show(C.median_of_seeds(vals))} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- training-fraction sweep
def timings(c):
    f = lambda key, cond: C.first_epoch(c, key, cond)
    t_train = f("train_acc", lambda v: v == 1.0)
    t_test = C.grok_epoch(c)
    return {
        "t_train": t_train,
        "t_test": t_test,
        "delay": None if (t_test == C.NEVER or t_train is None) else t_test - t_train,
        "train_loss_1e-6": f("train_loss", lambda v: v < 1e-6),
        "test_loss_1e-6": f("test_loss", lambda v: v < 1e-6),
        "max_test_acc": max(c["test_acc"]),
    }


def _has_delay(m, at_least):
    return m["delay"] is not None and m["delay"] >= at_least


HYPOTHESES = [   # (fractions, statement, holds for one run)
    (["0.1", "0.2"], "1. fractions 0.1 and 0.2: t_test > 40 000 for every seed",
     lambda m: m["t_test"] == C.NEVER),
    (["0.3", "0.4", "0.5"], "2. fractions 0.3-0.5: t_train <= 1 000 and delay >= 1 000 for every seed",
     lambda m: m["t_train"] is not None and m["t_train"] <= 1000 and _has_delay(m, 1000)),
    (["0.6", "0.7", "0.8", "0.9"], "4. fractions 0.6-0.9: delay < 1 000 for every seed",
     lambda m: m["delay"] is not None and m["delay"] < 1000),
]


def hypotheses(M, med):
    """[(statement, confirmed)] for every hypothesis whose runs have all finished."""
    done = lambda frs: all(M[(fr, s)] for fr in frs for s in SEEDS)
    out = [(text, all(holds(M[(fr, s)]) for fr in frs for s in SEEDS))
           for frs, text, holds in HYPOTHESES if done(frs)]
    if done(["0.3", "0.4", "0.5"]):
        trend = " / ".join(C.fmt_epoch(med[fr]) for fr in ["0.3", "0.4", "0.5"])
        out.append((f"3. median t_test decreases 0.3 > 0.4 > 0.5 ({trend})", med["0.3"] > med["0.4"] > med["0.5"]))
    return sorted(out)


def _timing_row(fr, s, m):
    if m is None:
        return f"| {fr} | {s} | not finished | | | | | |"
    cells = [C.fmt_epoch(m[k]) for k in ("t_train", "t_test", "delay", "train_loss_1e-6", "test_loss_1e-6")]
    return f"| {fr} | {s} | " + " | ".join(cells) + f" | {m['max_test_acc']:.1%} |"


def fraction_report(runs):
    """Markdown: timings of every run, medians over seeds and the verdict on each hypothesis."""
    M = {k: (None if c is None else timings(c)) for k, c in runs.items()}
    med = {fr: C.median_of_seeds([M[(fr, s)] and M[(fr, s)]["t_test"] for s in SEEDS]) for fr in GRIDS["frac"]}
    lines = ["| fraction | seed | t_train | t_test (> 99%) | delay | train loss < 1e-6 | test loss < 1e-6 | best test acc |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [_timing_row(fr, s, m) for (fr, s), m in M.items()]
    return ("\n".join(lines)
            + "\n\nMedian t_test over seeds: " + ", ".join(f"{fr}: {C.fmt_epoch(v)}" for fr, v in med.items())
            + "\n\nHypotheses (docs/preregistration.md):\n"
            + "\n".join(f"- {h}: {'confirmed' if ok else 'NOT confirmed'}" for h, ok in hypotheses(M, med)))

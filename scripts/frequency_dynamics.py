"""Exploratory: when do frequencies appear in the embedding, in the output map W_L and in the neurons?

    python scripts/frequency_dynamics.py        # writes results/frequency_dynamics.md

Not pre-registered: these measurements were made after the seed grids, to see how the key frequencies
come about. Per checkpoint of every run with saved weights it records the W_E and W_L norm of each frequency
k = 1..56, how many neurons have k as their best frequency with FVE > 0.85, and the mean neuron FVE.
The per-checkpoint arrays are cached in runs/dynamics/<run>.npz.

Runs: the authors' trajectory and our main run (checkpoints every 100 epochs), early_i0_d0 (the main run
retrained for 400 epochs with checkpoints every 5), and the 21 runs of the seed grids (every 500).
"""
import sys

import numpy as np
import torch

from grokking import curves, fourier as fo, paths
from grokking import weights as W

sys.path.insert(0, str(paths.ROOT / "scripts"))
import seeds_analysis as sa  # noqa: E402

CACHE = paths.run_dir("dynamics")
SEED_RUNS = [n for n, _, _ in sa.RUNS + sa.FOLLOWUP + [sa.SYM] if n != "seed0"]


def measure(name):
    """Per-checkpoint arrays for one run, cached."""
    path = CACHE / f"{name}.npz"
    if path.exists():
        return dict(np.load(path))
    rows = []
    for e, load in W.checkpoints(name):
        if name in ("authors", "seed0") and e > 20000 and e % 1000:
            continue
        r = fo.Run(load())
        freq, fve, _ = r.neuron_fve()
        rows.append((e, r.embed_freq_norms()[1].numpy(), r.W_L_freq_norms()[1].numpy(),
                     torch.bincount(freq[fve > 0.85], minlength=57)[1:].numpy(), fve.mean().item()))
    d = {"epoch": np.array([x[0] for x in rows])}
    for i, key in enumerate(("emb", "wl", "clean", "fve_mean"), start=1):
        d[key] = np.array([x[i] for x in rows])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)
    return d


def rank(v):
    """Rank of each frequency (index k-1), 1 = largest."""
    return np.argsort(np.argsort(-v)) + 1


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def key_sets(d):
    return [tuple(k for k in range(1, 57) if c[k - 1] >= 10) for c in d["clean"]]


def t_test(name):
    if name == "authors":
        return None
    return curves.grok_epoch(curves.our_curves(name))


def history(name, d):
    """Final keys, when the set is first complete and stable, transient extra keys, W_E top-n stability."""
    E, sets = list(d["epoch"]), key_sets(d)
    final = sets[-1]
    complete = next(E[i] for i, s in enumerate(sets) if set(final) <= set(s))
    stable = next(E[i] for i in range(len(E)) if all(s == final for s in sets[i:]))
    first_key = next(E[i] for i, s in enumerate(sets) if s)
    extra = {}
    for i, s in enumerate(sets):
        for k in s:
            if k not in final:
                extra.setdefault(k, []).append(E[i])
    n = len(final)
    tops = [set((np.argsort(-v)[:n] + 1).tolist()) for v in d["emb"]]
    top_stable = next((E[i] for i in range(len(E)) if all(t == set(final) for t in tops[i:])), None)
    return {"final": list(final), "first_key": first_key, "complete": complete, "stable": stable,
            "extra": {k: (v[0], v[-1]) for k, v in extra.items()}, "top_stable": top_stable}


def at(d, epoch):
    return list(d["epoch"]).index(epoch)


def main():
    names = ["authors", "seed0", "early_i0_d0"] + SEED_RUNS
    D = {n: measure(n) for n in names}
    lines = ["# Frequency dynamics (exploratory)", "",
             "Key set at a checkpoint: frequencies with at least 10 neurons whose best frequency it is with FVE > 0.85. "
             "Checkpoints every 100 epochs for authors and seed0 (every 1 000 after epoch 20 000), every 500 for the seed grids. "
             "t_test: first epoch with test accuracy above 99% (authors: not logged; the paper puts the jump at 10 000–14 000).", "",
             "## Key-set history", "",
             "| run | t_test | final keys | first key | all final keys present | set final and stable | transient extra keys (first–last epoch as key) | W_E top-n = final keys from |",
             "|---|---|---|---|---|---|---|---|"]
    H = {}
    for n in ["authors", "seed0"] + SEED_RUNS:
        h = H[n] = history(n, D[n])
        ex = ", ".join(f"{k} ({a}–{b})" for k, (a, b) in sorted(h["extra"].items())) or "none"
        lines.append(f"| {n} | {t_test(n) or '—'} | {h['final']} | {h['first_key']} | {h['complete']} | {h['stable']} | {ex} | {h['top_stable']} |")
    seeds = [n for n in SEED_RUNS]
    with_extra = [n for n in ["authors", "seed0"] + seeds if H[n]["extra"]]
    gaps = [H[n]["complete"] - t_test(n) for n in ["seed0"] + seeds if not H[n]["extra"]]
    lines += ["", f"Runs with a transient extra key: {len(with_extra)} of {len(H)} ({', '.join(with_extra)}).",
              f"Runs without one: all final keys present at t_test {min(gaps):+d} to {max(gaps):+d} epochs "
              f"(median {int(np.median(gaps)):+d}); checkpoint spacing 100 (seed0) or 500 epochs.", ""]

    lines += ["## Neurons commit late: seed0 and authors", "",
              "| run | epoch | mean neuron FVE | clean neurons on each final key | key set |", "|---|---|---|---|---|"]
    for n in ("seed0", "authors"):
        d, final = D[n], H[n]["final"]
        for e in (100, 1000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 14000, 20000, 39000):
            if e in list(d["epoch"]):
                i = at(d, e)
                lines.append(f"| {n} | {e} | {d['fve_mean'][i]:.2f} | {[int(d['clean'][i][k - 1]) for k in final]} | {list(key_sets(d)[i])} |")
    lines += ["", "## The first 400 epochs of the main run (early_i0_d0, checkpoints every 5 epochs)", "",
              "Ranks among the 56 frequencies of the final key frequencies 4, 13, 25, 35.", "",
              "| epoch | W_E ranks | W_E norm / median | W_L ranks | Spearman(W_E, W_L) over 56 frequencies |", "|---|---|---|---|---|"]
    keys = [4, 13, 25, 35]
    d = D["early_i0_d0"]
    for e in (0, 5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400):
        if e in list(d["epoch"]):
            i = at(d, e)
            emb, wl = d["emb"][i], d["wl"][i]
            lines.append(f"| {e} | {[int(rank(emb)[k - 1]) for k in keys]} | {[round(float(emb[k - 1] / np.median(emb)), 2) for k in keys]} | "
                         f"{[int(rank(wl)[k - 1]) for k in keys]} | {spearman(emb, wl):.2f} |")
    lines += ["", "## Agreement of the W_E and W_L spectra", "",
              "Spearman rank correlation of the 56 frequency norms of W_E and W_L.", "",
              "| epoch | " + " | ".join(str(e) for e in (0, 100, 500, 1000, 2000, 4000, 8000, 12000, 20000)) + " |",
              "|---|" + "---|" * 9]
    for n in ("seed0", "authors"):
        d = D[n]
        lines.append(f"| {n} | " + " | ".join(f"{spearman(d['emb'][at(d, e)], d['wl'][at(d, e)]):.2f}"
                                             for e in (0, 100, 500, 1000, 2000, 4000, 8000, 12000, 20000)) + " |")
    text = "\n".join(lines) + "\n"
    (paths.RESULTS / "frequency_dynamics.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

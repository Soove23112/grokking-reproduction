"""Key frequencies across seeds: the analysis of the runs from scripts/seeds.sh and scripts/seeds2.sh.

    python scripts/seeds_analysis.py      # writes results/seeds.md

Pre-registered metrics and hypotheses: docs/preregistration.md, "Key frequencies across seeds"
and "Follow-up: symmetry check and seeds 8–15".
"""
from collections import Counter
from itertools import combinations

from grokking import curves, fourier as fo, paths
from grokking.weights import KEY, ours_at

EPOCH = 20_000
KEY_EPOCH = 19_500  # last checkpoint of a 20 000-epoch run (epochs 0..19 999, saved every 500)
EARLY = [0, 500, 1000]
# (name, init seed, data seed); every run, including the main run seed0, is read at KEY_EPOCH
RUNS = [("seed0", 0, 0)] + [(f"seeds_i{i}_d0", i, 0) for i in range(1, 8)] + [
    ("seeds_i0_d0_rep", 0, 0), ("seeds_i0_d1", 0, 1), ("seeds_i0_d2", 0, 2),
    ("seeds_i1_d1", 1, 1), ("seeds_i1_d2", 1, 2)]
FOLLOWUP = [(f"seeds_i{i}_d0", i, 0) for i in range(8, 16)]
SYM = ("sym_i0_d0_u7", 0, 0)          # the main model with every number renamed x -> 7x mod 113
SYM_MAP = {4: 49, 13: 18, 25: 52, 35: 5}  # k -> 97k mod 113, folded into 1..56
THRESHOLDS = [(t, m) for t in (0.8, 0.85, 0.9) for m in (5, 10, 20)]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def ranks(per_freq_norms):
    """Rank (1 = largest) of each frequency k = 1..56 by its W_E norm -> {k: rank}."""
    order = (-per_freq_norms).argsort().tolist()
    return {k + 1: order.index(k) + 1 for k in range(len(order))}


def grokked(name):
    c = curves.our_curves(name)
    return any(a > 0.99 for e, a in zip(c["epoch"], c["test_acc"]) if e <= EPOCH)


def counts_under_thresholds(freq, fve):
    return [len(fo.key_freqs(freq, fve, t, m)[0]) for t, m in THRESHOLDS]


def measure(name):
    freq, fve, _ = fo.Run(ours_at(KEY_EPOCH, name)).neuron_fve()
    keys, counts = fo.key_freqs(freq, fve)
    early = {e: ranks(fo.Run(ours_at(e, name)).embed_freq_norms()[1]) for e in EARLY}
    others = [int(counts[k]) for k in fo.FREQS if k not in keys]
    return {"keys": keys, "neurons": [int(counts[k]) for k in keys], "robust": counts_under_thresholds(freq, fve),
            "other_max": max(others), "rank7_at_0": early[0][7],
            "grokked": grokked(name), "t_test": curves.grok_epoch(curves.our_curves(name)),
            "early": {e: [early[e][k] for k in keys] for e in EARLY}}


def pair_overlaps(res, same_init):
    """Mean Jaccard over pairs of grokked runs with (same init, different data) or (different init, same data)."""
    vals = []
    for (n1, i1, d1), (n2, i2, d2) in combinations(RUNS, 2):
        if "rep" in n1 or "rep" in n2 or not (res[n1]["grokked"] and res[n2]["grokked"]):
            continue
        if (same_init and i1 == i2 and d1 != d2) or (not same_init and i1 != i2 and d1 == d2):
            vals.append(jaccard(res[n1]["keys"], res[n2]["keys"]))
    return sum(vals) / len(vals) if vals else float("nan"), len(vals)


def fmt_t(t):
    return "> 20 000" if t == curves.NEVER else t


def table(res, runs):
    rows = ["| run | init | data | t_test | key frequencies | neurons | most neurons on another frequency | ranks at epoch 0 | 500 | 1 000 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for n, i, d in runs:
        r = res[n]
        rows.append(f"| {n} | {i} | {d} | {fmt_t(r['t_test'])} | {', '.join(map(str, r['keys']))} | "
                    f"{', '.join(map(str, r['neurons']))} | {r['other_max']} | "
                    + " | ".join(", ".join(map(str, r["early"][e])) for e in EARLY) + " |")
    return rows


def report(res):
    lines = ["# Key frequencies across seeds", "",
             f"Key frequencies at epoch {KEY_EPOCH:,} (the last checkpoint; the plan said 20 000, which is never saved); "
             "neurons: how many neurons have that frequency as their best one with FVE > 0.85; "
             "ranks of each final key frequency among the 56 W_E frequency norms.", "",
             "## First grid", ""] + table(res, RUNS)
    on_split0 = [res[n] for n, _, d in RUNS if d == 0 and "rep" not in n]
    g = [r for r in on_split0 if r["grokked"]]
    same, n_same = pair_overlaps(res, True)
    diff, n_diff = pair_overlaps(res, False)
    top5 = sum(sum(rk <= 5 for rk in r["early"][1000]) >= 3 for r in g)
    pooled = [rk for r in on_split0 for rk in r["early"][0]]
    lines += ["", "## Hypotheses", "",
              f"1. Counts on data seed 0 (grokked runs): {sorted(len(r['keys']) for r in g)}.",
              f"2. Mean Jaccard, same init / different split: {same:.2f} ({n_same} pairs); "
              f"different init / same split: {diff:.2f} ({n_diff} pairs).",
              f"3. Repeat of the main model: {res['seeds_i0_d0_rep']['keys']} (main model at epoch {KEY_EPOCH:,}: {res['seed0']['keys']}).",
              f"4. Runs with at least 3 final key frequencies in the W_E top 5 at epoch 1 000: {top5} of {len(g)}.",
              f"5. Final key frequencies in the W_E top 10 at epoch 0: {sum(rk <= 10 for rk in pooled)} of {len(pooled)} "
              f"({sum(rk <= 10 for rk in pooled) / len(pooled):.0%}; chance 18%)."]
    lines += ["", "## Exploratory (found after looking at the first grid, not pre-registered)", ""]
    for split in (0, 1, 2):
        sets = [res[n]["keys"] for n, _, d in RUNS if d == split and "rep" not in n] + ([KEY["authors"]] if split == 0 else [])
        ks = [k for v in sets for k in v]
        top = ", ".join(f"{k} ({c})" for k, c in Counter(ks).most_common(4))
        lines.append(f"- Data seed {split}, {len(sets)} models{' incl. the authors' if split == 0 else ''}: most frequent key frequencies "
                     f"{top}; multiples of 7: {sum(k % 7 == 0 for k in ks)} of {len(ks)} (chance 8/56 = 14%).")
    return "\n".join(lines + followup(res)) + "\n"


def followup(res):
    sym, main = res[SYM[0]], res["seed0"]
    expected = sorted(SYM_MAP.values())
    mapped = {SYM_MAP[k]: {e: main["early"][e][i] for e in EARLY} for i, k in enumerate(main["keys"])}
    same_ranks = all(sym["early"][e][i] == mapped[k][e] for i, k in enumerate(sym["keys"]) for e in EARLY) \
        if sorted(sym["keys"]) == expected else False
    new = [res[n] for n, _, _ in FOLLOWUP]
    g = [r for r in new if r["grokked"]]
    all0 = [res[n]["keys"] for n, _, d in RUNS + FOLLOWUP if d == 0 and "rep" not in n]
    ks = [k for v in all0 for k in v]
    lines = ["", "## Follow-up: symmetry check and seeds 8–15", "", "### Symmetry check", "",
             f"Run {SYM[0]}: key frequencies {sym['keys']}, expected {expected}; ranks at epochs 0 / 500 / 1 000 "
             f"equal to those of the matching frequencies of the main model: {same_ranks}; t_test {fmt_t(sym['t_test'])} "
             f"(main model {fmt_t(main['t_test'])}, its repeat {fmt_t(res['seeds_i0_d0_rep']['t_test'])}).", "",
             "### Seeds 8–15 on data seed 0", ""] + table(res, FOLLOWUP + [SYM]) + ["",
             f"6. Frequency 7 among the key frequencies: {sum(7 in r['keys'] for r in new)} of {len(new)} new runs "
             f"(prediction: at least 3). Frequency 14: {sum(14 in r['keys'] for r in new)} of {len(new)}.",
             f"7. Runs with exactly 4 key frequencies: {sum(len(r['keys']) == 4 for r in g)} of {len(g)} grokked new runs "
             f"(prediction: at least 5); counts {sorted(len(r['keys']) for r in g)}.",
             "", f"All {len(all0)} of our models on data seed 0: counts "
             f"{dict(sorted(Counter(len(v) for v in all0).items()))}; most frequent key frequencies "
             + ", ".join(f"{k} ({c})" for k, c in Counter(ks).most_common(5)) + ".",
             "", "Rank of frequency 7 among the 56 W_E frequencies at epoch 0, all models on data seed 0 "
             "(* = 7 becomes a key frequency): " + ", ".join(
                 f"{n.replace('seeds_', '')} {res[n]['rank7_at_0']}{'*' if 7 in res[n]['keys'] else ''}"
                 for n, _, d in RUNS + FOLLOWUP if d == 0 and "rep" not in n) + ".",
             "", "### Key-frequency count under other thresholds of the rule", "",
             "| run | " + " | ".join(f"FVE > {t}, ≥ {m}" for t, m in THRESHOLDS) + " |",
             "|---|" + "---|" * len(THRESHOLDS)]
    for n, _, _ in RUNS + FOLLOWUP + [SYM]:
        lines.append(f"| {n} | " + " | ".join(map(str, res[n]["robust"])) + " |")
    return lines


def main():
    res = {n: measure(n) for n, _, _ in RUNS + FOLLOWUP + [SYM]}
    text = report(res)
    (paths.RESULTS / "seeds.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

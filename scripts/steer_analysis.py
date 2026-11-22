"""Analysis of scripts/steer.py: speed-ups, hypotheses 1-6, written to results/steer.md.

    python scripts/steer_analysis.py
"""
import json
from statistics import median

import torch

from grokking import fourier as fo
from grokking import model as gm
from grokking import paths

CONDS = ["none", "scale", "top4_once", "rand4_once", "top4_persist", "rand4_persist"]
NEVER = float("inf")


def load_results():
    rows = [json.loads(l) for l in open(paths.RESULTS / "steer.jsonl")]
    return {(r["init"], r["cond"]): r for r in rows}


def final_keys(init, cond):
    m = gm.Transformer()
    m.load_state_dict(torch.load(paths.run_dir("steer") / f"{init}_{cond}.pt", map_location="cpu", weights_only=True))
    return fo.Run(m.double().eval()).key_freqs()[0]


def t(r):
    return NEVER if r["t_test"] is None else r["t_test"]


def fmt(x):
    return "never" if x == NEVER else f"{x:,}".replace(",", " ")


def main():
    R = load_results()
    inits = sorted({i for i, _ in R})
    keys = {(i, c): final_keys(i, c) for i, c in R}
    lines = ["# Keeping a few frequencies in W_E", "",
             "t_test: first epoch with test accuracy above 99% (checked every 10 epochs); speed-up: t_test(none) / t_test.", "",
             "| init | " + " | ".join(CONDS) + " | top 4 at epoch 100 | final keys of none |",
             "|---|" + "---|" * (len(CONDS) + 2)]
    for i in inits:
        lines.append(f"| {i} | " + " | ".join(fmt(t(R[i, c])) for c in CONDS)
                     + f" | {R[i, 'top4_once']['top4']} | {keys[i, 'none']} |")
    sp = {c: [t(R[i, "none"]) / t(R[i, c]) for i in inits] for c in CONDS}
    lines += ["", "Median speed-up: " + ", ".join(f"{c} {median(sp[c]):.1f}" for c in CONDS[1:]) + ".", ""]
    lines += ["| init | cond | forced / kept frequencies | final key frequencies | final test loss | W_E norm kept |",
              "|---|---|---|---|---|---|"]
    for i in inits:
        for c in CONDS:
            r = R[i, c]
            lines.append(f"| {i} | {c} | {r.get('keys', '')} | {keys[i, c]} | {r['test_loss']:.2e} | "
                         f"{r.get('kept_norm', r.get('factor', '')) if c != 'none' else ''} |")
    n = len(inits)
    h1 = sum(t(R[i, "top4_once"]) < t(R[i, "none"]) for i in inits)
    h3 = sum(t(R[i, "top4_once"]) < t(R[i, "rand4_once"]) for i in inits)
    h4a = sum(R[i, "rand4_persist"]["t_test"] is not None for i in inits)
    h4b = sum(t(R[i, "rand4_persist"]) < t(R[i, "none"]) for i in inits)
    h4c = sum(0.5 <= t(R[i, "rand4_persist"]) / t(R[i, "top4_persist"]) <= 2 for i in inits)
    h5 = []
    for i in inits:
        forced, k = set(R[i, "rand4_persist"]["keys"]), set(keys[i, "rand4_persist"])
        ok = len(k & forced) >= 3 and not (k - forced) and R[i, "rand4_persist"]["test_loss"] <= 10 * R[i, "none"]["test_loss"]
        h5.append(ok)
    h6 = sum(len(set(R[i, "top4_once"]["top4"]) & set(keys[i, "none"])) >= 3 for i in inits)
    lines += ["", "## Hypotheses", "",
              f"1. top4_once earlier than none: {h1} of {n}; median speed-up {median(sp['top4_once']):.1f} (prediction: {n} of {n}, at least 5).",
              f"2. Median speed-up of scale: {median(sp['scale']):.1f} (prediction: at most 1.5).",
              f"3. top4_once earlier than rand4_once: {h3} of {n} (prediction: at least 4).",
              f"4. rand4_persist reaches 99%: {h4a} of {n}; earlier than none: {h4b} of {n}; "
              f"rand4_persist / top4_persist within 0.5-2: {h4c} of {n} (prediction: {n}, {n}, at least 4).",
              f"5. rand4_persist runs the forced frequencies (>= 3 of 4 key, no other key, test loss within 10x of none): "
              f"{sum(h5)} of {n} (prediction: {n} of {n}).",
              f"6. Top 4 at epoch 100 hold >= 3 final keys of none: {h6} of {n} (prediction: at least 4)."]
    text = "\n".join(lines) + "\n"
    (paths.RESULTS / "steer.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

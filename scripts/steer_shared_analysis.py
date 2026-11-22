"""Analysis of scripts/steer_shared.py: per-initialisation ratios, the two hypotheses, the change at the projection.

    python scripts/steer_shared_analysis.py      # writes results/steer_shared.md

Pre-registration: docs/preregistration.md, "Same starting state".
"""
import csv
import json
from math import log
from statistics import median

from grokking import fourier as fo, paths

CONDS = ["none", "top4", "rand4n"]
CHANCE = log(113)


def load():
    rows = [json.loads(line) for line in open(paths.RESULTS / "steer_shared.jsonl")]
    return {(r["init"], r["cond"]): r for r in rows}


def at_projection(init, cond):
    """The 'before' and 'after' rows logged at epoch 100."""
    rows = list(csv.DictReader(open(paths.run_dir("steer_shared") / f"{init}_{cond}.csv")))
    return {r["when"]: {k: float(v) for k, v in r.items() if k not in ("when",)} for r in rows if r["when"]}


def final_keys(init, cond):
    return fo.Run(fo.load(path=paths.run_dir("steer_shared") / f"{init}_{cond}" / "final.pt")).key_freqs()[0]


def ratios(R, inits, num, den):
    return [R[i, num]["t_test"] / R[i, den]["t_test"] for i in inits]


def fmt(xs):
    return f"median {median(xs):.2f}, range {min(xs):.2f}–{max(xs):.2f}"


def main():
    R = load()
    inits = sorted({i for i, _ in R})
    assert all(R[i, c]["t_test"] is not None for i in inits for c in CONDS), "a run did not reach 99%"
    lines = ["# Same starting state: top-4 against random 4 frequencies of equal norm", "",
             "t_test: first epoch with test accuracy above 99% (checked every 10 epochs). "
             "All continuations of one initialisation start from the same saved state at epoch 100.", "",
             "| init | none | top4 | rand4n | top4 frequencies | rand4n frequencies | rand4n factor | final key frequencies none / top4 / rand4n | final test loss none / top4 / rand4n |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i in inits:
        r = {c: R[i, c] for c in CONDS}
        lines.append(f"| {i} | {r['none']['t_test']} | {r['top4']['t_test']} | {r['rand4n']['t_test']} | "
                     f"{r['top4']['keys']} | {r['rand4n']['keys']} | {r['rand4n']['factor']:.3f} | "
                     + " / ".join(str(final_keys(i, c)) for c in CONDS) + " | "
                     + " / ".join(f"{r[c]['test_loss']:.1e}" for c in CONDS) + " |")
    r_tr = ratios(R, inits, "rand4n", "top4")
    r_nr = ratios(R, inits, "none", "rand4n")
    r_nt = ratios(R, inits, "none", "top4")
    after = [R[i, "rand4n"]["t_test"] - 100 for i in inits], [R[i, "top4"]["t_test"] - 100 for i in inits]
    h1 = sum(R[i, "top4"]["t_test"] < R[i, "rand4n"]["t_test"] for i in inits)
    h2 = sum(R[i, "rand4n"]["t_test"] < R[i, "none"]["t_test"] for i in inits)
    n = len(inits)
    lines += ["", "## Hypotheses", "",
              f"1. top4 earlier than rand4n: {h1} of {n}; t_test(rand4n) / t_test(top4): {fmt(r_tr)} "
              f"(prediction: at least 4 of {n}, median at least 1.5).",
              f"2. rand4n earlier than none: {h2} of {n} (prediction: {n} of {n}).", "",
              "## Speed-ups", "",
              f"- none / top4: {fmt(r_nt)}.",
              f"- none / rand4n: {fmt(r_nr)}.",
              f"- epochs after the change, rand4n / top4: {fmt([a / b for a, b in zip(*after)])}.",
              "", "## At the projection (epoch 100, just before and just after)", "",
              f"Chance level of the loss: ln 113 = {CHANCE:.2f}.", "",
              "| init | cond | train acc before → after | train loss before → after | test acc before → after | "
              "test loss before → after | W_E norm before → after |", "|---|---|---|---|---|---|---|"]
    for i in inits:
        for c in ("top4", "rand4n"):
            p = at_projection(i, c)
            b, a = p["before"], p["after"]
            lines.append(f"| {i} | {c} | {b['train_acc']:.1%} → {a['train_acc']:.1%} | {b['train_loss']:.2f} → {a['train_loss']:.2f} | "
                         f"{b['test_acc']:.1%} → {a['test_acc']:.1%} | {b['test_loss']:.2f} → {a['test_loss']:.2f} | "
                         f"{b['W_E_norm']:.2f} → {a['W_E_norm']:.2f} |")
    text = "\n".join(lines) + "\n"
    (paths.RESULTS / "steer_shared.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

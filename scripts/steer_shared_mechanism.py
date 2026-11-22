"""Restricted and excluded loss on the saved weights of scripts/steer_shared.py.

    python scripts/steer_shared_mechanism.py    # writes results/steer_shared_mechanism.md and .csv

Planned in docs/preregistration.md, "Same starting state": computed afterwards from the saved weights,
for the projected frequencies and for the final key frequencies of each run, and reported only after
checking that they remain meaningful when the final key frequencies differ from the projected ones.
No hypothesis was attached to them.

Definitions as in section 5.1 of the paper and grokking/fourier.py: the restricted loss keeps only the
constant and cos/sin(w_k(a+b)) of the given frequencies in the 2D Fourier transform of the logits (loss
over all 12 769 pairs); the excluded loss removes exactly those terms (loss on the training pairs).
For `none` the "projected" set is the top-4 set of the same initialisation, the one `top4` projects onto.
"""
import csv
import json
from math import log

from grokking import fourier as fo, paths

CONDS = ["none", "top4", "rand4n"]
CHANCE = log(113)
OUT = paths.run_dir("steer_shared")
SHOW = [100, 110, 150, 200, 300, 400]


def checkpoints(init, cond):
    d = OUT / f"{init}_{cond}"
    return sorted((int(f.stem), f) for f in d.glob("[0-9]*.pt")) + [("final", d / "final.pt")]


def measure(path, proj, final):
    r = fo.Run(fo.load(path=path))
    return {"train_loss": r.loss(mask=r.is_train), "test_loss": r.loss(mask=~r.is_train),
            "restr_proj": r.restricted_loss(proj), "restr_final": r.restricted_loss(final),
            "excl_proj": r.excluded_loss(proj), "excl_final": r.excluded_loss(final)}


def first(rows, key, cond):
    return next((r["epoch"] for r in rows if r["epoch"] != "final" and cond(r[key])), None)


def main():
    R = {(r["init"], r["cond"]): r for r in map(json.loads, open(paths.RESULTS / "steer_shared.jsonl"))}
    inits = sorted({i for i, _ in R})
    rows = []
    for i in inits:
        for c in CONDS:
            proj = R[i, "top4"]["keys"] if c == "none" else R[i, c]["keys"]
            final = fo.Run(fo.load(path=OUT / f"{i}_{c}" / "final.pt")).key_freqs()[0]
            for e, f in checkpoints(i, c):
                rows.append({"init": i, "cond": c, "epoch": e, "proj": proj, "final_keys": final,
                             "t_test": R[i, c]["t_test"], **measure(f, proj, final)})
            print(i, c, "done", flush=True)
    with open(paths.RESULTS / "steer_shared_mechanism.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def run(i, c):
        return [r for r in rows if r["init"] == i and r["cond"] == c]

    lines = ["# Restricted and excluded loss after the projection (shared starting state)", "",
             "Restricted loss: only the constant and cos/sin(w_k(a+b)) of the frequency set kept in the logits, all pairs. "
             "Excluded loss: those terms removed, training pairs. 'proj': the projected set (for `none`, the top-4 set of the "
             "same initialisation); 'final': the key frequencies of the run's final weights. "
             f"Chance level ln 113 = {CHANCE:.2f}. Epoch 100 is just after the projection.", "",
             "## Meaningfulness check on the final weights", "",
             "If the measures mean what they should, at the end the restricted loss on the final keys is close to the loss "
             "of the full model, and removing those terms makes the training loss large.", "",
             "| init | cond | projected | final keys | full loss train / test | restricted final | restricted proj | excluded final | excluded proj |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i in inits:
        for c in CONDS:
            f = run(i, c)[-1]
            lines.append(f"| {i} | {c} | {f['proj']} | {f['final_keys']} | {f['train_loss']:.1e} / {f['test_loss']:.1e} | "
                         f"{f['restr_final']:.1e} | {f['restr_proj']:.1e} | {f['excl_final']:.1f} | {f['excl_proj']:.1f} |")
    lines += ["", "## Timing", "",
              "First epoch (checkpoints every 10 epochs up to 400, then every 100) at which the restricted loss on the final "
              "keys falls below 0.1 and below 10⁻³, the test loss below 0.1, and t_test (first test accuracy > 99%, checked every 10 epochs).", "",
              "| init | cond | restricted < 0.1 | test loss < 0.1 | restricted < 1e-3 | t_test | peak excluded (final keys), epoch |",
              "|---|---|---|---|---|---|---|"]
    for i in inits:
        for c in CONDS:
            rs = run(i, c)
            pk = max((r for r in rs if r["epoch"] != "final"), key=lambda r: r["excl_final"])
            lines.append(f"| {i} | {c} | {first(rs, 'restr_final', lambda v: v < 0.1)} | {first(rs, 'test_loss', lambda v: v < 0.1)} | "
                         f"{first(rs, 'restr_final', lambda v: v < 1e-3)} | {rs[0]['t_test']} | {pk['excl_final']:.1f} at {pk['epoch']} |")
    lines += ["", "## Early epochs", "",
              "| init | cond | epoch | train loss | test loss | restricted final | excluded final |", "|---|---|---|---|---|---|---|"]
    for i in inits:
        for c in CONDS:
            for r in run(i, c):
                if r["epoch"] in SHOW:
                    lines.append(f"| {i} | {c} | {r['epoch']} | {r['train_loss']:.2f} | {r['test_loss']:.2f} | "
                                 f"{r['restr_final']:.2e} | {r['excl_final']:.2f} |")
    text = "\n".join(lines) + "\n"
    (paths.RESULTS / "steer_shared_mechanism.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

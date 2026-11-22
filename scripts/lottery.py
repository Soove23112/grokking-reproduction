"""Exploratory: does the initial state predict which frequencies win? A test of the lottery of He et al. (2026).

    python scripts/lottery.py        # writes results/lottery.md

He et al. (arXiv 2602.16849) study a two-layer MLP on modular addition: inside each neuron the frequencies
compete, and the winner is the one with the largest initial input and output magnitudes and the smallest
phase misalignment. We adapt their score to our transformer at epoch 0. For neuron n, theta_a / theta_b are
the parts of its pre-activation at '=' that depend only on a / only on b (marginal means over all pairs),
and xi(c) = W_L[n, c]. With 1D Fourier coefficients Fa, Fb, Fx for k = 1..56 and D = arg Fa + arg Fb - arg Fx,
the score is |Fa| |Fb| |Fx| cos D, the growth rate of the k(a+b-c) term.

Not pre-registered. Only the "full" variant follows He et al.; the other six were tried afterwards.
"""
import math
import sys

import numpy as np
import torch

from grokking import fourier as fo, paths
from grokking import model as gm
from grokking import weights as W

sys.path.insert(0, str(paths.ROOT / "scripts"))
import seeds_analysis as sa  # noqa: E402

P = fo.P
# one run per distinct (init, data) pair: the repeat and the relabelled copy of seed0 are left out
RUNS = [("seed0", 39900)] + [(n, sa.KEY_EPOCH) for n, _, _ in sa.RUNS + sa.FOLLOWUP
                             if n not in ("seed0", "seeds_i0_d0_rep")] + [("authors", 40000)]


def models(name, final_epoch):
    if name == "authors":
        return fo.load(W.authors_state_dicts()[0], authors=True), fo.load(W.authors_state(), authors=True)
    return W.ours_at(0, name), W.ours_at(final_epoch, name)


def fourier1d(v):
    """(P, N) -> complex coefficients (56, N) for k = 1..56."""
    j = torch.arange(P).double()
    E = torch.exp(-2j * math.pi * torch.outer(torch.arange(1, 57).double(), j) / P)
    return E @ v.to(torch.complex128)


def init_terms(m):
    x, _ = gm.all_pairs()
    with torch.no_grad():
        _, c = m(x, return_cache=True)
    pre = c["pre"][:, -1].reshape(P, P, -1)
    ta, tb = pre.mean(1) - pre.mean((0, 1)), pre.mean(0) - pre.mean((0, 1))
    WL = (m.W_out.detach() @ m.W_U.detach()[:, :P]).T
    Fa, Fb, Fx = fourier1d(ta), fourier1d(tb), fourier1d(WL)
    cosD = torch.cos(Fa.angle() + Fb.angle() - Fx.angle()).numpy()
    return Fa.abs().numpy(), Fb.abs().numpy(), Fx.abs().numpy(), cosD


def rank(v):
    return np.argsort(np.argsort(-v)) + 1


def init_group(name):
    """Runs that share an initialisation seed are resampled together in the bootstrap."""
    if name == "authors":
        return "authors"
    return "i0" if name == "seed0" or name.startswith("seeds_i0_") else name.split("_")[1]


def main():
    rng = np.random.default_rng(0)
    per_neuron, variants = [], {}
    for name, fe in RUNS:
        m0, mf = models(name, fe)
        a, b, x, cosD = init_terms(m0)
        mag = a * b * x
        freq, fve, _ = fo.Run(mf).neuron_fve()
        keys, _ = fo.key_freqs(freq, fve)
        freq, clean = freq.numpy(), (fve > 0.85).numpy()
        pred = (mag * cosD).argmax(0) + 1
        exact = (pred[clean] == freq[clean]).mean()
        shuffle = np.mean([(rng.permutation(pred)[clean] == freq[clean]).mean() for _ in range(200)])
        per_neuron.append((name, int(clean.sum()), exact, shuffle))
        scores = {"full (He et al.): sum over neurons of max(0, A_a A_b A_x cos D)": np.clip(mag * cosD, 0, None).sum(1),
                  "signed: sum of A_a A_b A_x cos D": (mag * cosD).sum(1),
                  "magnitude only: sum of A_a A_b A_x": mag.sum(1),
                  "phase only: sum of cos D": cosD.sum(1),
                  "input side only: sum of A_a A_b": (a * b).sum(1),
                  "output side only: sum of A_x": x.sum(1),
                  "W_E frequency norm": fo.Run(m0).embed_freq_norms()[1].numpy()}
        for v, s in scores.items():
            r = rank(s)
            variants.setdefault(v, []).append(([int(r[k - 1]) for k in keys], len(keys), name))
        print(name, "done", flush=True)

    lines = ["# Does the initial state predict the winning frequencies? (exploratory)", "",
             f"{len(RUNS)} runs, one per distinct (initialisation, data) pair, the authors' included. "
             "Neurons counted: FVE > 0.85 in the final model. "
             "A_a, A_b, A_x: magnitudes of the Fourier coefficients Fa, Fb, Fx of a neuron at epoch 0; "
             "D: its phase misalignment (see the docstring of scripts/lottery.py).", "",
             "## Per neuron: lottery winner at epoch 0 = the neuron's final frequency", "",
             "| run | clean neurons | exact match | with neurons shuffled |", "|---|---|---|---|"]
    lines += [f"| {n} | {c} | {e:.1%} | {s:.1%} |" for n, c, e, s in per_neuron]
    groups = sorted({init_group(n) for _, _, n in variants["W_E frequency norm"]})
    lines += ["", "## Over the whole model: rank of each final key frequency by a per-frequency score at epoch 0", "",
              f"Chance: top-n about {np.mean([n / 56 for _, n, _ in variants['W_E frequency norm']]):.0%}, "
              "top 10 18%, median rank 28.5. 95% intervals: bootstrap over "
              f"{len(groups)} initialisation groups (runs sharing an initialisation resampled together).", "",
              "| score | keys in top-n | keys in top 10 | 95% interval | median rank |", "|---|---|---|---|---|"]

    def top10(v, sel):
        r = np.array([x for g in sel for rs, _, n in variants[v] if init_group(n) == g for x in rs])
        return np.mean(r <= 10)

    boots = {v: [] for v in variants}
    for _ in range(5000):
        sel = rng.choice(groups, len(groups))
        for v in variants:
            boots[v].append(top10(v, sel))
    for v, lst in variants.items():
        ranks = np.array([x for rs, _, _ in lst for x in rs])
        hits, tot = sum(sum(r <= n for r in rs) for rs, n, _ in lst), sum(n for _, n, _ in lst)
        lines.append(f"| {v} | {hits}/{tot} ({hits / tot:.0%}) | {np.mean(ranks <= 10):.0%} | "
                     f"{np.percentile(boots[v], 2.5):.0%}–{np.percentile(boots[v], 97.5):.0%} | {np.median(ranks):.0f} |")
    text = "\n".join(lines) + "\n"
    (paths.RESULTS / "lottery.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

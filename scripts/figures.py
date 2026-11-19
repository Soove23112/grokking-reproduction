"""Report figures in figures/. Numbers follow the matching figures of the paper.

    python scripts/figures.py           # figures 2-7
    python scripts/figures.py sweeps    # weight decay and training-fraction sweeps
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from grokking import curves as C
from grokking import fourier as fo
from grokking import paths, sweeps, weights
from grokking import phases as ph

FIG, RES = paths.FIGURES, paths.RESULTS
FIG.mkdir(exist_ok=True)
MODELS = [("authors", "Authors, epoch 40 000"), ("seed0", "Ours, epoch 39 900")]


def snap(name):
    return json.load(open(RES / f"{name}_snapshot.json"))


def final_run(name):
    return fo.Run(weights.final(name)[1]())


def fig2_curves():
    ours, auth = C.our_curves("seed0"), C.authors_curves()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ours["epoch"], ours["train_loss"], label="ours: train")
    ax[0].plot(ours["epoch"], ours["test_loss"], label="ours: test")
    ax[0].plot(auth["epoch"], auth["train_loss"], "--", color="gray", label="authors: train")
    ax[0].plot(auth["epoch"], auth["test_loss"], ":", color="black", label="authors: test")
    ax[0].set_yscale("log"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(ours["epoch"], ours["train_acc"], label="train")
    ax[1].plot(ours["epoch"], ours["test_acc"], label="test")
    ax[1].set_title("Accuracy, our model"); ax[1].legend()
    for a in ax:
        a.set_xlabel("epoch"); a.set_xlim(0, 30000)
    fig.tight_layout(); fig.savefig(FIG / "fig2_curves.png", dpi=130); plt.close(fig)


def fig3_fourier_norms():
    fig, ax = plt.subplots(2, 2, figsize=(13, 7))
    for col, (name, title) in enumerate(MODELS):
        r = final_run(name)
        key = snap(name)["key_freqs"]
        for row, (comp, what) in enumerate([(r.embed_freq_norms()[0], "embedding W_E"),
                                             (r.W_L_freq_norms()[0], "neuron→logit W_L")]):
            ks = torch.arange(1, fo.N_FREQ + 1)
            a = ax[row, col]
            a.bar(ks - 0.2, comp[1::2], width=0.4, label="cos")
            a.bar(ks + 0.2, comp[2::2], width=0.4, label="sin")
            a.set_title(f"{title}: {what}\nkey frequencies {key}")
            a.set_xlabel("frequency k"); a.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig3_fourier_norms.png", dpi=130); plt.close(fig)


def fig4_periodicity():
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for row, (name, title) in enumerate(MODELS):
        r = final_run(name)
        s = snap(name)
        hk, _ = s["head_freqs"][0]
        nk, _ = s["neuron0"]
        im = ax[row, 0].imshow(r.attn[:, 0, 2, 0].reshape(fo.P, fo.P), cmap="viridis")
        ax[row, 0].set_title(f"{title}\nhead 0 attention from '=' to a (frequency {hk})")
        fig.colorbar(im, ax=ax[row, 0], fraction=0.046)
        im = ax[row, 1].imshow(r.post[:, 0].reshape(fo.P, fo.P), cmap="viridis")
        ax[row, 1].set_title(f"neuron 0 activation (frequency {nk})")
        fig.colorbar(im, ax=ax[row, 1], fraction=0.046)
        im = ax[row, 2].imshow(r.logit_fourier_norms(), cmap="Blues")
        ax[row, 2].set_title("logit norm in 2D Fourier space\n(row: component along a, column: along b)")
        fig.colorbar(im, ax=ax[row, 2], fraction=0.046)
        for a in ax[row, :2]:
            a.set_xlabel("b"); a.set_ylabel("a")
    fig.tight_layout(); fig.savefig(FIG / "fig4_periodicity.png", dpi=130); plt.close(fig)


def fig5_neuron_fve():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, (name, title) in zip(ax, MODELS):
        _, fve, _ = final_run(name).neuron_fve()
        n = int((fve > 0.85).sum())
        a.hist(fve.numpy(), bins=40)
        a.axvline(0.85, color="red")
        a.set_title(f"{title}\n{n} of 512 neurons > 85%")
        a.set_xlabel("fraction of neuron variance explained by one frequency")
    fig.tight_layout(); fig.savefig(FIG / "fig5_neuron_fve.png", dpi=130); plt.close(fig)


def fig6_ablations():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for a, (name, title) in zip(ax, MODELS):
        r = final_run(name)
        s = snap(name)
        ks = list(range(1, fo.N_FREQ + 1))
        losses = [r.ablate_freq_loss(k) for k in ks]
        colors = ["C3" if k in s["key_freqs"] else "C0" for k in ks]
        a.bar(ks, losses, color=colors)
        a.axhline(s["loss_all"], color="black", lw=1, label="no ablation")
        a.axhline(s["restricted_loss"], color="green", lw=1, label="key frequencies only (restricted)")
        a.set_yscale("log"); a.set_xlabel("removed frequency k (red: key)")
        a.set_title(f"{title}: loss after removing a frequency"); a.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig6_ablations.png", dpi=130); plt.close(fig)


def mark_phases(x, bounds):
    for v in (v for v in bounds.values() if v is not None):
        x.axvline(v, color="gray", lw=0.8, ls="--")
    x.set_xlim(0, 30000); x.set_xlabel("epoch")
    if x.get_legend_handles_labels()[0]:
        x.legend(fontsize=8)


def fig7_progress():
    fig, ax = plt.subplots(2, 4, figsize=(20, 8))
    for row, (name, title) in enumerate(MODELS):
        t = C.read_csv(RES / f"{name}_trajectory.csv")
        e = t["epoch"]
        b = ph.boundaries(t)
        a = ax[row]
        a[0].plot(e, t["train_loss"], label="train"); a[0].plot(e, t["test_loss"], label="test")
        a[0].plot(e, t["excluded_loss"], label="excluded (train)")
        a[0].set_yscale("log"); a[0].set_title(f"{title}\nlosses and excluded loss")
        a[1].plot(e, t["test_loss"], label="test"); a[1].plot(e, t["restricted_loss"], label="restricted")
        a[1].set_yscale("log"); a[1].set_title("restricted loss")
        a[2].plot(e, t["gini_embed"], label="W_E"); a[2].plot(e, t["gini_W_L"], label="W_L")
        a[2].set_title("Gini coefficient of Fourier norms")
        a[3].plot(e, t["sum_sq_weights"]); a[3].set_title("sum of squared weights")
        for x in a:
            mark_phases(x, b)
    fig.tight_layout(); fig.savefig(FIG / "fig7_progress.png", dpi=130); plt.close(fig)


def test_acc_grid(grid, title, out, ncols):
    vals = sweeps.GRIDS[grid]
    nrows = -(-len(vals) // ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), sharey=True, squeeze=False)
    runs = sweeps.grid_curves(grid)
    for a, v in zip(ax.flat, vals):
        for s in sweeps.SEEDS:
            c = runs[(v, s)]
            if c is None:
                continue
            a.plot(c["epoch"], c["test_acc"], label=f"test, seed {s}")
            a.plot(c["epoch"], c["train_acc"], ":", color="gray", lw=0.8)
        a.axhline(0.99, color="red", lw=0.6)
        a.set_xscale("log"); a.set_xlim(100, 40000); a.set_title(title(v))
        a.set_xlabel("epoch (log); dotted: train")
    ax.flat[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / out, dpi=120); plt.close(fig)


def fig_sweeps():
    test_acc_grid("wd", lambda v: f"weight decay λ = {v}", "fig_wd_sweep.png", ncols=4)
    test_acc_grid("frac", lambda v: f"training fraction {v}", "fig_data_fraction.png", ncols=3)


if __name__ == "__main__":
    if sys.argv[1:] == ["sweeps"]:
        fig_sweeps()
    else:
        for f in [fig2_curves, fig3_fourier_norms, fig4_periodicity, fig5_neuron_fve, fig6_ablations, fig7_progress]:
            f(); print("ok", f.__name__)

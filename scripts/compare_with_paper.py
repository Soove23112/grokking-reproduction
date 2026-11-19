"""Comparison with the paper on the criteria fixed in docs/preregistration.md before our model was trained.

    python scripts/compare_with_paper.py    -> results/comparison.md

Reads results/{authors,seed0}_snapshot.json, results/*_trajectory.csv and runs/seed0/metrics.csv.
"""
import json
import math

from grokking import curves as C
from grokking import paths
from grokking import phases as ph

RES = paths.RESULTS
A = json.load(open(RES / "authors_snapshot.json"))
S = json.load(open(RES / "seed0_snapshot.json"))
ours_curves = C.our_curves("seed0")
ta = C.read_csv(RES / "authors_trajectory.csv")
ts = C.read_csv(RES / "seed0_trajectory.csv")
pa, ps = ph.boundaries(ta), ph.boundaries(ts)
LN113 = math.log(113)


def rel(x, base):
    return f"{(x / base - 1) * 100:+.0f}%"


def ok(b):
    return "yes" if b else "NO"


rows = []   # (what, paper, authors, ours, authors match, ours match)


def row(what, paper, a, s, a_ok, s_ok):
    rows.append((what, paper, a, s, a_ok, s_ok))


row("loss over all pairs, final checkpoint", "2.41e-7",
    f"{A['loss_all']:.3g}", f"{S['loss_all']:.3g}", abs(A["loss_all"] / 2.41e-7 - 1) < 0.01, None)
row("100% train accuracy, epoch", "memorised by ~1 400", "—",
    str(C.first_epoch(ours_curves, "train_acc", lambda v: v == 1.0)), None,
    C.first_epoch(ours_curves, "train_acc", lambda v: v == 1.0) <= 2000)
t99 = C.first_epoch(ours_curves, "test_acc", lambda v: v > 0.99)
row(">99% test accuracy, epoch", "~10 000", "—", str(t99), None, t99 is not None and t99 < 40000)
row("key frequencies (rule: >=10 neurons with FVE > 85%)", "14, 35, 41, 42, 52",
    ", ".join(map(str, A["key_freqs"])), ", ".join(map(str, S["key_freqs"])),
    A["key_freqs"] == [14, 35, 41, 42, 52], 3 <= len(S["key_freqs"]) <= 6)
emb_a = sum(1 for v, _ in A["embed_freq_norms_top8"] if v > 10 * A["embed_freq_norm_median"])
emb_s = sum(1 for v, _ in S["embed_freq_norms_top8"] if v > 10 * S["embed_freq_norm_median"])
row("frequencies with a large norm in W_E (>10 medians)", "6", str(emb_a), str(emb_s), emb_a == 6, emb_s == len(S["key_freqs"]) + 1)
row("neuron 0: frequency", "42", str(A["neuron0"][0]), str(S["neuron0"][0]), A["neuron0"][0] == 42, None)
row("head 0: attention frequency", "35", str(A["head_freqs"][0][0]), str(S["head_freqs"][0][0]), A["head_freqs"][0][0] == 35, None)
row("neurons with single-frequency FVE > 85%", "433 / 512",
    f"{A['neurons_fve_gt_085']} / 512", f"{S['neurons_fve_gt_085']} / 512",
    abs(A["neurons_fve_gt_085"] - 433) <= 5, S["neurons_fve_gt_085"] >= 0.7 * 512)
row("neurons of frequency 14", "44", str(A["neurons_per_key_freq"]["14"]), "—", A["neurons_per_key_freq"]["14"] == 44, None)
row("W_L outside the cos/sin directions of key frequencies (share of squared norm)", "< 0.55%",
    f"{A['W_L_residual_sq_ratio'] * 100:.2f}%", f"{S['W_L_residual_sq_ratio'] * 100:.2f}%",
    A["W_L_residual_sq_ratio"] < 0.006, S["W_L_residual_sq_ratio"] < 0.02)
n_big_a = sum(1 for v in A["logit_2d_norms_top24"] if v > 0.3 * A["logit_2d_norms_top24"][0])
n_big_s = sum(1 for v in S["logit_2d_norms_top24"] if v > 0.3 * S["logit_2d_norms_top24"][0])
row("large 2D Fourier components of the logits (>30% of max)", "20 (4 per frequency)", str(n_big_a), str(n_big_s),
    n_big_a == 20, n_big_s == 4 * len(S["key_freqs"]))
t1a = min(f for *_, f in A["table1"]); t1s = min(f for *_, f in S["table1"])
row("table 1: FVE of u_k, v_k projections by one cos/sin(w(a+b)) term, minimum", "93.2%",
    f"{t1a * 100:.1f}%", f"{t1s * 100:.1f}%", abs(t1a - 0.932) < 0.005, t1s > 0.9)
row("logits ~ sum alpha_k cos(w_k(a+b-c)): fraction of variance", "95%",
    f"{A['cos_fit_fve'] * 100:.1f}%", f"{S['cos_fit_fve'] * 100:.1f}%",
    0.94 <= A["cos_fit_fve"] <= 0.96, S["cos_fit_fve"] >= 0.90)
row("  ... loss of this approximation", "4.7e-8", f"{A['cos_fit_loss']:.2g}", f"{S['cos_fit_loss']:.2g}", None, None)
row("ablation: removing any key frequency from the logits raises the loss", "yes",
    f"min {min(A['ablate_key_loss'].values()):.1e}", f"min {min(S['ablate_key_loss'].values()):.1e}",
    min(A["ablate_key_loss"].values()) > 10 * A["loss_all"], min(S["ablate_key_loss"].values()) > 10 * S["loss_all"])
row("ablation: removing any non-key frequency barely changes the loss", "yes",
    f"max {A['ablate_nonkey_max_loss']:.2g}", f"max {S['ablate_nonkey_max_loss']:.2g}",
    A["ablate_nonkey_max_loss"] < 1.1 * A["loss_all"], S["ablate_nonkey_max_loss"] < 1.1 * S["loss_all"])
row("logits: keep only the 40 key-frequency components (4.4)", "−70% (7.24e-8)",
    f"{rel(A['keep_blocks_loss'], A['loss_all'])} ({A['keep_blocks_loss']:.3g})",
    f"{rel(S['keep_blocks_loss'], S['loss_all'])} ({S['keep_blocks_loss']:.3g})",
    abs(A["keep_blocks_loss"] / 7.24e-8 - 1) < 0.01, S["keep_blocks_loss"] < S["loss_all"])
row("restricted loss (5.1): constant + cos/sin(w_k(a+b))", "— (fig. 7 only)",
    f"{rel(A['restricted_loss'], A['loss_all'])} ({A['restricted_loss']:.3g})",
    f"{rel(S['restricted_loss'], S['loss_all'])} ({S['restricted_loss']:.3g})", None, None)
row("neurons with FVE > 85% replaced by their frequency polynomial", "+3% (2.48e-7)",
    f"{rel(A['neuron_poly_loss'], A['loss_all'])}", f"{rel(S['neuron_poly_loss'], S['loss_all'])}", None, None)
row("all neurons: keep only cos/sin(w_k(a+b)) of key frequencies", "−77% (5.54e-8)",
    f"{rel(A['neuron_strict_loss'], A['loss_all'])}", f"{rel(S['neuron_strict_loss'], S['loss_all'])}", None, None)
row("MLP activations projected onto the 10 W_L directions", "−50% (1.19e-7)",
    f"{rel(A['proj10_loss'], A['loss_all'])}", f"{rel(S['proj10_loss'], S['loss_all'])}", None, None)
row("... onto their null space: loss", "5.27 (worse than chance, ln 113 = 4.73)",
    f"{A['null10_loss']:.2f}", f"{S['null10_loss']:.2f}",
    abs(A["null10_loss"] - 5.27) < 0.1, S["null10_loss"] > LN113)
row("Gini W_E / W_L at the end", "~0.8 on fig. 7; 0.62–0.80 / 0.73–0.91 in table 5",
    f"{A['gini_embed']:.2f} / {A['gini_W_L']:.2f}", f"{S['gini_embed']:.2f} / {S['gini_W_L']:.2f}", None, None)
row("phases t1 / t2 / t3, epochs", "1 400 / 9 400 / 14 000",
    f"{pa['t1']} / {pa['t2']} / {pa['t3']}", f"{ps['t1']} / {ps['t2']} / {ps['t3']}",
    abs(pa["t1"] - 1400) <= 1000 and abs(pa["t2"] - 9400) <= 1000 and abs(pa["t3"] - 14000) <= 1000,
    ps["t1"] < ps["t2"] < ps["t3"])


def cell(v):
    return "" if v is None else ok(v)


lines = ["| what | paper | authors' weights | our model | authors match | ours match |",
         "|---|---|---|---|---|---|"]
for what, paper, a, s, ao, so in rows:
    lines.append(f"| {what} | {paper} | {a} | {s} | {cell(ao)} | {cell(so)} |")
md = "\n".join(lines)
(RES / "comparison.md").write_text(md + "\n")
print(md)
fails = [r[0] for r in rows if r[4] is False or r[5] is False]
print("\nmismatches:", fails if fails else "none")

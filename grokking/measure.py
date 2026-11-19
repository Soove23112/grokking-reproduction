"""Section 4 measurements on one model (a snapshot) and section 5 progress measures over a trajectory.

Key frequencies are found on the final checkpoint by the rule fourier.Run.key_freqs and then
applied to every checkpoint of the trajectory.
"""
from grokking import fourier as fo
from grokking import weights


def _top8(norms):
    return sorted([(round(v.item(), 3), k + 1) for k, v in enumerate(norms)], reverse=True)[:8]


def snapshot(name):
    """Every number the comparison table needs, measured on the final model of `name`."""
    epoch, loader = weights.final(name)
    r = fo.Run(loader())
    key, counts = r.key_freqs()
    freq, fve, _ = r.neuron_fve()
    _, embed = r.embed_freq_norms()
    _, wl = r.W_L_freq_norms()
    logit_norms = r.logit_fourier_norms().flatten().sort(descending=True).values
    cos_fve, alpha, cos_loss = r.logits_cos_fit(key)
    poly_loss, n_poly = r.ablate_neurons_poly()
    non_key = [k for k in fo.FREQS if k not in key]
    w_l_res = r.W_L_residual(key)
    return {
        "epoch": epoch,
        "loss_all": r.loss(), "loss_train": r.loss(mask=r.is_train), "loss_test": r.loss(mask=~r.is_train),
        "key_freqs": key,
        "neurons_per_key_freq": {k: int(counts[k]) for k in key},
        "neurons_fve_gt_085": int((fve > 0.85).sum()),
        "neuron0": [int(freq[0]), fve[0].item()],
        "head_freqs": [r.head_freq(h) for h in range(4)],
        "embed_freq_norms_top8": _top8(embed),
        "embed_freq_norm_median": embed.median().item(),
        "W_L_freq_norms_top8": _top8(wl),
        "W_L_residual_norm_ratio": w_l_res,
        "W_L_residual_sq_ratio": w_l_res ** 2,
        "logit_2d_norms_top24": [round(v, 1) for v in logit_norms[:24].tolist()],
        # table 1 coefficients in the paper's units: the paper's waves are not normalised, ours are
        # (the norm of cos(w(a+b)) over 113*113 points is 113/sqrt(2)), hence the division
        "table1": [(k, t, c / (fo.P / 2 ** 0.5), f) for k, t, c, f in r.table1(key)],
        "cos_fit_fve": cos_fve, "cos_fit_alpha": alpha.tolist(), "cos_fit_loss": cos_loss,
        "neuron_poly_loss": poly_loss, "neuron_poly_n": n_poly,
        "neuron_strict_loss": r.ablate_neurons_strict(key),
        "proj10_loss": r.project_post(key), "null10_loss": r.project_post(key, null=True),
        "keep_blocks_loss": r.keep_blocks_loss(key),
        "restricted_loss": r.restricted_loss(key), "excluded_loss": r.excluded_loss(key),
        "ablate_key_loss": {k: r.ablate_freq_loss(k) for k in key},
        "ablate_nonkey_max_loss": max(r.ablate_freq_loss(k) for k in non_key),
        "gini_embed": r.gini_embed(), "gini_W_L": r.gini_W_L(),
    }


def progress_row(epoch, r, key):
    return {
        "epoch": epoch,
        "train_loss": r.loss(mask=r.is_train), "test_loss": r.loss(mask=~r.is_train),
        "restricted_loss": r.restricted_loss(key), "excluded_loss": r.excluded_loss(key),
        "gini_embed": r.gini_embed(), "gini_W_L": r.gini_W_L(),
        "sum_sq_weights": sum(p.pow(2).sum().item() for p in r.m.parameters()),
    }


def trajectory(name, on_row=None):
    """Progress measures at every checkpoint of `name` -> (rows, key frequencies)."""
    _, loader = weights.final(name)
    key, _ = fo.Run(loader()).key_freqs()
    rows = []
    for epoch, load in weights.checkpoints(name):
        rows.append(progress_row(epoch, fo.Run(load()), key))
        if on_row:
            on_row(rows[-1])
    return rows, key

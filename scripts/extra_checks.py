"""Numbers quoted in the README that are not part of the snapshots or the comparison table.

    python scripts/extra_checks.py

1. The pre-registered phase rules (t_excl, t_restr, t_test) with k = 1.1 and q = 1.6e-3.
2. Test accuracy at t2 for both models.
3. Our 30% model on the 10% training pairs (a subset of the 30% ones), against the models trained on 10%.
"""

import torch

from grokking import curves as C
from grokking import fourier as fo
from grokking import model as gm
from grokking import paths, weights
from grokking import phases as ph


def preregistered_phases(t, k=1.1, q=1.6e-3):
    """t_excl: first epoch after the minimum where the excluded loss exceeds k times it;
    t_restr: first epoch where the restricted loss is below q times its initial value;
    t_test: first epoch with test loss below 1e-3."""
    e, excl, restr = t["epoch"], t["excluded_loss"], t["restricted_loss"]
    i_min = min(range(len(e)), key=lambda i: excl[i])
    t_excl = next((int(e[i]) for i in range(i_min, len(e)) if excl[i] > k * excl[i_min]), None)
    t_restr = next((int(x) for x, v in zip(e, restr) if v < q * restr[0]), None)
    t_test = next((int(x) for x, v in zip(e, t["test_loss"]) if v < 1e-3), None)
    return t_excl, t_restr, t_test


def test_acc(model):
    _, _, vx, vy = gm.make_data()
    with torch.no_grad():
        return gm.accuracy(model(vx), vy)


def main():
    traj = {n: C.read_csv(paths.RESULTS / f"{n}_trajectory.csv") for n in ("authors", "seed0")}
    for n, t in traj.items():
        print(f"{n}: pre-registered phase rules t_excl / t_restr / t_test = {preregistered_phases(t)}")

    for n, t in traj.items():
        t2 = ph.boundaries(t)["t2"]
        m = weights.checkpoints("authors")[t2 // 100][1]() if n == "authors" else weights.ours_at(t2)
        print(f"{n}: test accuracy at t2 = {t2}: {test_acc(m):.1%}")

    r = weights.run("ours")
    tx10, _, _, _ = gm.make_data(frac=0.1)
    sub = torch.zeros(gm.P * gm.P, dtype=torch.bool)
    sub[tx10[:, 0] * gm.P + tx10[:, 1]] = True
    assert (sub & ~r.is_train).sum() == 0, "the 10% pairs should be a subset of the 30% ones"
    print(f"ours (30%): sum of squared weights {gm.sum_sq_weights(r.m):.0f}, "
          f"loss on the {int(sub.sum())} pairs of the 10% split {r.loss(mask=sub):.2g}")
    for s in (0, 1, 2):
        c = C.our_curves(f"frac0.1_s{s}")
        print(f"frac0.1_s{s} at epoch {c['epoch'][-1]:.0f}: sum of squared weights {c['sum_sq_weights'][-1]:.0f}, "
              f"training loss {c['train_loss'][-1]:.2g}")


if __name__ == "__main__":
    main()

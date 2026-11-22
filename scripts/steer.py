"""Does keeping only a few frequencies in the number embeddings speed up grokking?

    python scripts/steer.py --init 0 --cond top4_once          # one run, appends to results/steer.jsonl
    python scripts/steer.py --list                             # print every (init, condition) of the plan

At epoch START the embeddings of the numbers, W_E[:113] (113 x 128), are changed in token space:

    none          nothing (baseline)
    scale         multiplied by the factor that top4_once would shrink them by (norm control)
    top4_once     projected once onto {const, cos k, sin k} of the 4 largest W_E frequencies at START
    rand4_once    the same with 4 random frequencies outside those 4 (seeded by the init seed)
    top4_persist  top4 projection re-applied after every step
    rand4_persist rand4 projection re-applied after every step

t_test is the first epoch (evaluated every 10) with test accuracy above 0.99. Training goes on for
EXTRA epochs after t_test (or stops at MAX_EPOCHS) and the final weights go to runs/steer/<init>_<cond>.pt.
Pre-registered hypotheses: docs/preregistration.md, "Keeping a few frequencies in W_E".
"""
import argparse
import json
import time

import numpy as np
import torch

from grokking import fourier as fo
from grokking import model as gm
from grokking import paths

START, EVAL_EVERY, EXTRA, MAX_EPOCHS = 100, 10, 1000, 15_000
INITS = [0, 1, 2, 3, 4]
CONDS = ["none", "scale", "top4_once", "rand4_once", "top4_persist", "rand4_persist"]
P = gm.P


def projector(keys):
    """(113, 113) orthogonal projector onto the constant and the cos/sin of the given frequencies."""
    idx = [0] + [i for k in keys for i in (fo.cos_idx(k), fo.sin_idx(k))]
    Bs = fo.B[idx].float()
    return Bs.T @ Bs


def top_freqs(W_E, n=4):
    norms = fo.per_freq_norm((fo.B @ W_E.detach()[:P].double().cpu()).norm(dim=1))
    return (norms.argsort(descending=True)[:n] + 1).tolist()


def random_freqs(init, exclude, n=4):
    rng = np.random.default_rng(1000 + init)
    pool = [k for k in fo.FREQS if k not in exclude]
    return sorted(int(k) for k in rng.choice(pool, n, replace=False))


def intervention(cond, init, W_E):
    """-> (matrix M applied as W_E[:113] = M @ W_E[:113], persistent?, info)."""
    top = top_freqs(W_E)
    E = W_E.detach()[:P].float().cpu()
    kept = ((projector(top) @ E).norm() / E.norm()).item()
    if cond == "scale":
        return kept * torch.eye(P), False, {"top4": top, "factor": kept}
    keys = top if cond.startswith("top4") else random_freqs(init, top)
    return projector(keys), cond.endswith("persist"), {"top4": top, "keys": keys, "kept_norm": kept}


def run(init, cond, device="cpu"):
    torch.manual_seed(0)
    model = gm.init_weights(gm.Transformer(), init).to(device)
    tx, ty, vx, vy = gm.make_data(P, 0.3, 0)
    tx, vx = tx.to(device), vx.to(device)
    opt, sched = gm.make_optimizer(model)
    M, persist, info, t_test, end = None, False, {}, None, MAX_EPOCHS
    t0 = time.time()
    for epoch in range(MAX_EPOCHS + EXTRA):
        if epoch == START and cond != "none":
            M, persist, info = intervention(cond, init, model.W_E)
            M = M.to(device)
        if M is not None and (persist or epoch == START):
            with torch.no_grad():
                model.W_E[:P] = M @ model.W_E[:P]
        if t_test is None and epoch % EVAL_EVERY == 0:
            with torch.no_grad():
                if gm.accuracy(model(vx), vy) > 0.99:
                    t_test, end = epoch, epoch + EXTRA
        if epoch >= end:
            break
        loss = gm.loss_fn(model(tx), ty)
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()
    with torch.no_grad():
        tl, vl = model(tx), model(vx)
    out = paths.run_dir("steer")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / f"{init}_{cond}.pt")
    return {"init": init, "cond": cond, "t_test": t_test, "final_epoch": epoch, **info,
            "train_loss": gm.loss_fn(tl, ty).item(), "test_loss": gm.loss_fn(vl, vy).item(),
            "test_acc": gm.accuracy(vl, vy), "minutes": round((time.time() - t0) / 60, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=int)
    ap.add_argument("--cond", choices=CONDS)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for i in INITS:
            for c in CONDS:
                print(i, c)
        return
    res = run(a.init, a.cond)
    with open(paths.RESULTS / "steer.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()

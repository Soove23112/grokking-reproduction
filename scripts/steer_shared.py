"""Top-4 against random 4 frequencies of equal norm, both continued from one saved state at epoch 100.

    python scripts/steer_shared.py --init 0 --cond top4     # none | top4 | rand4n
    python scripts/steer_shared.py --list

The state at the start of epoch 100 (weights, AdamW, schedule) is trained once per initialisation and cached
in runs/steer_shared/<init>_e100.pt. Every continuation logs train/test loss and accuracy just before and
just after the change and then every 10 epochs (runs/steer_shared/<init>_<cond>.csv), saves weights every
10 epochs up to epoch 400 and every 100 after (runs/steer_shared/<init>_<cond>/), and appends a summary to
results/steer_shared.jsonl. Pre-registration: docs/preregistration.md, "Same starting state".
"""
import argparse
import csv
import fcntl
import json
import time

import torch

from grokking import model as gm
from grokking import paths
from steer import projector, random_freqs, top_freqs

START, EVERY, EXTRA, MAX_EPOCHS = 100, 10, 1000, 15_000
INITS, CONDS = [0, 1, 2, 3, 4], ["none", "top4", "rand4n"]
P = gm.P
OUT = paths.run_dir("steer_shared")


def fresh(init):
    model = gm.init_weights(gm.Transformer(), init)
    opt, sched = gm.make_optimizer(model)
    return model, opt, sched


def start_state(init, data):
    """Train epochs 0..START-1 once and cache the state (lock: several continuations may ask at once)."""
    path = OUT / f"{init}_e{START}.pt"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{init}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            model, opt, sched = fresh(init)
            tx, ty = data[0], data[1]
            for _ in range(START):
                gm.loss_fn(model(tx), ty).backward()
                opt.step(); sched.step(); opt.zero_grad()
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict()}, path)
    model, opt, sched = fresh(init)
    s = torch.load(path, weights_only=False)
    model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
    return model, opt, sched


def change(cond, init, W_E):
    """Apply the change in place; return what was done."""
    if cond == "none":
        return {}
    E = W_E.data[:P]
    top = top_freqs(W_E)
    target = (projector(top) @ E).norm()
    keys = top if cond == "top4" else random_freqs(init, top)
    new = projector(keys) @ E
    factor = (target / new.norm()).item() if cond == "rand4n" else 1.0
    E.copy_(new * factor)
    return {"top4": top, "keys": keys, "factor": factor}


def evaluate(model, data):
    tx, ty, vx, vy = data
    with torch.no_grad():
        tl, vl = model(tx), model(vx)
    return {"train_loss": gm.loss_fn(tl, ty).item(), "test_loss": gm.loss_fn(vl, vy).item(),
            "train_acc": gm.accuracy(tl, ty), "test_acc": gm.accuracy(vl, vy),
            "W_E_norm": model.W_E.detach()[:P].norm().item()}


def run(init, cond):
    data = gm.make_data(P, 0.3, 0)
    model, opt, sched = start_state(init, data)
    tx, ty = data[0], data[1]
    ck = OUT / f"{init}_{cond}"
    ck.mkdir(exist_ok=True)
    f = open(OUT / f"{init}_{cond}.csv", "w", newline="")
    w = None
    t0 = time.time()

    def log(epoch, when):
        nonlocal w
        row = {"epoch": epoch, "when": when, **evaluate(model, data)}
        if w is None:
            w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader()
        w.writerow(row); f.flush()
        return row

    log(START, "before")
    norm_before = model.W_E.detach()[:P].norm().item()
    with torch.no_grad():
        info = change(cond, init, model.W_E)
    info["norm_before"] = norm_before
    row = log(START, "after")
    torch.save(model.state_dict(), ck / f"{START:05d}.pt")
    t_test, end = None, MAX_EPOCHS
    for epoch in range(START, MAX_EPOCHS + EXTRA):
        if epoch % EVERY == 0 and epoch > START:
            row = log(epoch, "")
            if epoch <= 400 or epoch % 100 == 0:
                torch.save(model.state_dict(), ck / f"{epoch:05d}.pt")
        if t_test is None and row["test_acc"] > 0.99:
            t_test, end = epoch, epoch + EXTRA
        if epoch >= end:
            break
        gm.loss_fn(model(tx), ty).backward()
        opt.step(); sched.step(); opt.zero_grad()
    torch.save(model.state_dict(), ck / "final.pt")
    return {"init": init, "cond": cond, "t_test": t_test, "final_epoch": epoch, **info, **evaluate(model, data),
            "minutes": round((time.time() - t0) / 60, 1)}


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
    with open(paths.RESULTS / "steer_shared.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()

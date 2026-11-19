"""Train one model.

    python scripts/train.py --name seed0                                   # main run, checkpoints every 100 epochs
    python scripts/train.py --name wd0.3_s1 --wd 0.3 --init-seed 1 --save-every 0
    python scripts/train.py --name frac0.6_s1 --frac 0.6 --init-seed 1 --save-every 0 --stop-test-loss 1e-6

Writes runs/<name>/config.json, metrics.csv (every --eval-every epochs), done.json and,
if --save-every > 0, runs/<name>/ckpt/<epoch>.pt.
"""
import argparse
import csv
import json
import time

import torch

from grokking import model as gm
from grokking import paths


class StopEarly(Exception):
    pass


class MetricsLog:
    def __init__(self, path, stop_test_loss=0.0):
        self.f = open(path, "w", newline="")
        self.w = None
        self.stop = stop_test_loss
        self.t0 = time.time()

    def __call__(self, epoch, d):
        row = {"epoch": epoch, **d}
        if self.w is None:
            self.w = csv.DictWriter(self.f, fieldnames=list(row))
            self.w.writeheader()
        self.w.writerow(row)
        self.f.flush()
        if epoch % 1000 == 0:
            print(f"{epoch:>6}  train {d['train_loss']:.2e}  test {d['test_loss']:.2e}  "
                  f"acc {d['train_acc']:.3f}/{d['test_acc']:.3f}  {time.time() - self.t0:.0f}s", flush=True)
        if self.stop and d["test_loss"] < self.stop:
            raise StopEarly(epoch)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=40_000)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=100, help="0 disables checkpoints")
    ap.add_argument("--stop-test-loss", type=float, default=0.0, help="stop once test loss is below this (0: never)")
    return ap.parse_args()


def main():
    args = parse_args()
    out = paths.run_dir(args.name)
    if out.exists():
        raise SystemExit(f"{out} already exists; runs are never overwritten")
    (out / "ckpt").mkdir(parents=True)
    json.dump({**vars(args), "torch": torch.__version__}, open(out / "config.json", "w"), indent=2)

    cfg = dict(epochs=args.epochs, lr=args.lr, weight_decay=args.wd, frac=args.frac,
               data_seed=args.data_seed, device=args.device, eval_every=args.eval_every)
    model = gm.init_weights(gm.Transformer(), args.init_seed)
    log = MetricsLog(out / "metrics.csv", args.stop_test_loss)

    def save(epoch, sd):
        if args.save_every and epoch % args.save_every == 0:
            torch.save(sd, out / "ckpt" / f"{epoch:05d}.pt")

    try:
        gm.train(model, cfg, log=log, save=save)
        stopped = None
    except StopEarly as e:
        stopped = e.args[0]
    minutes = round((time.time() - log.t0) / 60, 1)
    json.dump({"stopped_early_at": stopped, "minutes": minutes}, open(out / "done.json", "w"))
    print(f"done in {minutes} min" + (f", stopped at epoch {stopped}" if stopped is not None else "") + f" -> {out}")


if __name__ == "__main__":
    main()

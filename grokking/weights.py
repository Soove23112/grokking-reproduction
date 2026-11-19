"""Where trained weights come from: the authors' trajectory and our own runs.

    from grokking import weights as W
    r = W.run("authors")       # authors' model at epoch 40 000, run over all 12 769 pairs
    r = W.run("ours")          # our model at epoch 39 900
    m = W.ours_at(5600)        # our model at epoch 5 600

The authors' trajectory file (456 MB) holds state_dicts every 100 epochs; the final model,
state_dicts[400], is cached in data/authors_final.pt (0.9 MB).
"""
from functools import lru_cache

import torch

from grokking import fourier as fo
from grokking import paths

OURS = "seed0"                     # our main run: runs/seed0
OURS_LAST = 39900
AUTHORS_FINAL_INDEX = 400          # epoch 40 000, the model of the paper

# key frequencies found by the rule fourier.Run.key_freqs on each final model
KEY = {"authors": [14, 35, 41, 42, 52], "ours": [4, 13, 25, 35]}


@lru_cache(maxsize=1)
def authors_run():
    """The authors' whole trajectory file: state_dicts, train_losses, test_losses, config."""
    return torch.load(paths.AUTHORS_RUN, map_location="cpu", weights_only=True)


def authors_state_dicts():
    """All 500 authors' state_dicts, one every 100 epochs."""
    return authors_run()["state_dicts"]


def authors_state():
    """The authors' final state_dict, extracted once from the full trajectory."""
    if not paths.AUTHORS_FINAL.exists():
        torch.save(authors_state_dicts()[AUTHORS_FINAL_INDEX], paths.AUTHORS_FINAL)
    return torch.load(paths.AUTHORS_FINAL, map_location="cpu", weights_only=True)


def ckpt_dir(name=OURS):
    return paths.run_dir(name) / "ckpt"


def ours_at(epoch, name=OURS):
    return fo.load(path=ckpt_dir(name) / f"{epoch:05d}.pt")


def available_epochs(name=OURS):
    return sorted(int(f.stem) for f in ckpt_dir(name).glob("*.pt"))


def model(who="authors"):
    """Final model in float64, eval mode."""
    return fo.load(authors_state(), authors=True) if who == "authors" else ours_at(OURS_LAST)


@lru_cache(maxsize=4)
def run(who="authors"):
    return fo.Run(model(who))


def checkpoints(name):
    """[(epoch, loader)] for the authors' trajectory up to epoch 40 000, or for runs/<name>/ckpt."""
    if name == "authors":
        sds = authors_state_dicts()[:AUTHORS_FINAL_INDEX + 1]
        return [(100 * i, (lambda sd=sd: fo.load(sd, authors=True))) for i, sd in enumerate(sds)]
    return [(e, (lambda e=e: ours_at(e, name))) for e in available_epochs(name)]


def final(name):
    """(epoch, loader) of the last checkpoint: epoch 40 000 for the authors, the last saved one for ours."""
    return checkpoints(name)[-1]

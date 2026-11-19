"""Every file location the project uses, in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
AUTHORS_RUN = DATA / "authors" / "grokking_addition_full_run.pth"   # authors' trajectory, 456 MB
AUTHORS_FINAL = DATA / "authors_final.pt"                            # its state_dicts[400], 0.9 MB
AUTHORS_CURVES = DATA / "authors_curves.csv"                         # its losses every 100 epochs

RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def run_dir(name):
    return RUNS / name

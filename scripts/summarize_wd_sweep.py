"""Weight decay sweep: the first epoch with test accuracy above 99%, per seed and median.

    python scripts/summarize_wd_sweep.py    -> results/sweep.md
"""
from grokking import paths, sweeps

if __name__ == "__main__":
    md = sweeps.wd_table(sweeps.grid_curves("wd"))
    paths.RESULTS.mkdir(exist_ok=True)
    (paths.RESULTS / "sweep.md").write_text(md + "\n")
    print(md)

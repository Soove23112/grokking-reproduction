"""Training-fraction sweep: timings per run and the four pre-registered hypotheses.

    python scripts/summarize_data_fraction.py    -> results/data_fraction.md
"""
from grokking import paths, sweeps

if __name__ == "__main__":
    md = sweeps.fraction_report(sweeps.grid_curves("frac"))
    paths.RESULTS.mkdir(exist_ok=True)
    (paths.RESULTS / "data_fraction.md").write_text(md + "\n")
    print(md)

"""Boundaries of the three grokking phases (section 5.2), calibrated on the authors' trajectory only.
They replace the pre-registered rules; the README explains why.

The paper places the boundaries by eye on fig. 7: 1.4k / 9.4k / 14k epochs.
The rules and what they give on the authors' trajectory (results/authors_trajectory.csv, every 100 epochs):

  t1, end of memorisation = epoch of the minimum of the excluded loss
                            (paper: "excluded loss falls during memorisation and rises during circuit formation")
                            -> 1 400
  t2, start of cleanup    = first epoch with test loss < ln 113 (better than chance)
                            (paper: "test loss falls sharply during cleanup")
                            -> 9 400 (4.85 at 9 300, 4.15 at 9 400)
  t3, end of cleanup      = first epoch where the sum of squared weights <= 1.05 * its final value (epoch 40 000)
                            (paper: "the sum of squared weights falls sharply during cleanup", then a stable phase)
                            -> 13 800

One parameter: the 5% in t3; t1 and t2 have none.
"""
import math


def boundaries(t):
    """t: trajectory columns (epoch, test_loss, excluded_loss, sum_sq_weights) -> {'t1', 't2', 't3'}."""
    e = t["epoch"]
    excl = t["excluded_loss"]
    t1 = int(e[min(range(len(e)), key=lambda i: excl[i])])
    t2 = next((int(x) for x, v in zip(e, t["test_loss"]) if v < math.log(113)), None)
    final = t["sum_sq_weights"][-1]
    t3 = next((int(x) for x, v in zip(e, t["sum_sq_weights"]) if v <= 1.05 * final), None)
    return {"t1": t1, "t2": t2, "t3": t3}

from grokking import curves as C
from grokking import phases as ph
from conftest import ROOT, needs_authors_run


def test_phase_rules_on_authors_trajectory():
    t = C.read_csv(ROOT / "results/authors_trajectory.csv")
    assert ph.boundaries(t) == {"t1": 1400, "t2": 9400, "t3": 13800}


def test_phase_rules_on_a_toy_trajectory():
    t = {"epoch": [0, 100, 200, 300],
         "excluded_loss": [3.0, 1.0, 2.0, 4.0],
         "test_loss": [5.0, 4.9, 1.0, 0.1],
         "sum_sq_weights": [100.0, 50.0, 10.4, 10.0]}
    assert ph.boundaries(t) == {"t1": 100, "t2": 200, "t3": 200}


def test_first_epoch_returns_none_when_never_true():
    assert C.first_epoch({"epoch": [0, 100], "x": [1, 2]}, "x", lambda v: v > 5) is None


def test_authors_curve_rows_align_losses_with_checkpoints():
    import torch
    sd = {"embed.W_E": torch.full((2, 2), 2.0), "blocks.0.attn.mask": torch.ones(3, 3), "IGNORE": torch.ones(1)}
    run = {"state_dicts": [sd, sd],
           "train_losses": [float(e) for e in range(201)],
           "test_losses": [float(-e) for e in range(21)]}
    assert C.authors_curve_rows(run) == [[0, 0.0, -0.0, 16.0], [100, 100.0, -10.0, 16.0]]


@needs_authors_run
def test_cached_authors_curves_match_the_trajectory_file():
    import pytest
    from grokking import weights
    c = C.authors_curves()
    rows = C.authors_curve_rows(weights.authors_run())
    assert c["epoch"] == [r[0] for r in rows]
    assert c["test_loss"] == pytest.approx([r[2] for r in rows], rel=1e-12)


def test_phase_rules_at_their_thresholds():
    import math
    t = {"epoch": [0, 100, 200],
         "excluded_loss": [2.0, 1.0, 3.0],
         "test_loss": [5.0, math.log(113), 1.0],       # exactly chance is not better than chance
         "sum_sq_weights": [20.0, 10.5, 10.0]}         # exactly 1.05 * final counts as settled
    assert ph.boundaries(t) == {"t1": 100, "t2": 200, "t3": 100}


def test_grok_epoch_needs_strictly_more_than_99_percent():
    assert C.grok_epoch({"epoch": [0, 100], "test_acc": [0.99, 0.995]}) == 100

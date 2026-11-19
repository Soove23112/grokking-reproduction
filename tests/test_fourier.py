import json
import math

import pytest
import torch

from grokking import fourier as fo
from grokking import measure
from conftest import ROOT, needs_authors_run

P = 113


def test_basis_is_orthonormal():
    assert torch.allclose(fo.B @ fo.B.T, torch.eye(P, dtype=torch.float64), atol=1e-12)


def test_2d_transform_round_trips():
    t = torch.randn(P * P, 3, dtype=torch.float64)
    assert torch.allclose(fo.from_fourier_2d(fo.to_fourier_2d(t)), t, atol=1e-10)


def _cos_sum(k):
    a = torch.arange(P).repeat_interleave(P).double()
    b = torch.arange(P).repeat(P).double()
    return torch.cos(2 * math.pi * k * (a + b) / P)[:, None]


def test_cos_of_sum_lives_in_its_xpy_directions():
    f = fo.to_fourier_2d(_cos_sum(14))
    assert torch.allclose(fo.keep_xpy(f, [14], keep_const=False), f, atol=1e-9)
    assert fo.remove_xpy(f, [14]).abs().max() < 1e-9


def test_keep_and_remove_split_the_signal():
    f = fo.to_fourier_2d(torch.randn(P * P, 2, dtype=torch.float64))
    kept = fo.keep_xpy(f, [3, 50], keep_const=False)
    assert torch.allclose(kept + fo.remove_xpy(f, [3, 50]), f, atol=1e-10)


def test_gini_extremes():
    assert fo.gini(torch.ones(100, dtype=torch.float64)) == pytest.approx(0, abs=1e-9)
    one_hot = torch.zeros(100, dtype=torch.float64)
    one_hot[7] = 1
    assert fo.gini(one_hot) == pytest.approx(1 - 1 / 100, abs=1e-12)  # Hurley & Rickard: 1 - 1/n


@needs_authors_run
def test_authors_snapshot_matches_committed_results():
    fresh = json.loads(json.dumps(measure.snapshot("authors")))
    committed = json.load(open(ROOT / "results/authors_snapshot.json"))
    assert fresh.keys() == committed.keys()
    for k in committed:
        assert _close(fresh[k], committed[k]), k


def _close(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return a == pytest.approx(b, rel=1e-9, abs=1e-15)
    return a == b


def test_key_frequency_rule_counts_exactly_min_neurons():
    freq = torch.tensor([14] * 10 + [35] * 9 + [42] * 10)
    fve = torch.tensor([0.9] * 29)
    fve[20] = 0.85                                     # exactly at the threshold: not counted
    key, counts = fo.key_freqs(freq, fve, thresh=0.85, min_neurons=10)
    assert key == [14] and counts[35] == 9 and counts[42] == 9

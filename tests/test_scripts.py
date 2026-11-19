"""The command-line scripts, run the way a user runs them, against a temporary runs/ directory."""
import csv
import importlib.util
import json
import sys

import pytest

from conftest import ROOT
from grokking import paths


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "RESULTS", tmp_path / "results")
    return tmp_path


def train(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["train.py", "--device", "cpu", *args])
    load_script("train").main()


def test_train_writes_metrics_checkpoints_and_done(runs, monkeypatch):
    train(monkeypatch, "--name", "tiny", "--epochs", "21", "--eval-every", "10", "--save-every", "20")
    out = paths.run_dir("tiny")
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert [r["epoch"] for r in rows] == ["0", "10", "20"]
    assert float(rows[0]["train_loss"]) == pytest.approx(4.7766063, rel=1e-6)
    assert sorted(f.name for f in (out / "ckpt").iterdir()) == ["00000.pt", "00020.pt"]
    assert json.load(open(out / "done.json"))["stopped_early_at"] is None


def test_train_stops_early_and_records_it(runs, monkeypatch):
    train(monkeypatch, "--name", "stop", "--epochs", "50", "--eval-every", "10", "--save-every", "0",
          "--stop-test-loss", "100")
    assert json.load(open(paths.run_dir("stop") / "done.json"))["stopped_early_at"] == 0


def test_train_refuses_to_overwrite_a_run(runs, monkeypatch):
    train(monkeypatch, "--name", "once", "--epochs", "1", "--save-every", "0")
    with pytest.raises(SystemExit):
        train(monkeypatch, "--name", "once", "--epochs", "1", "--save-every", "0")


def test_trajectory_of_a_short_run(runs, monkeypatch):
    train(monkeypatch, "--name", "short", "--epochs", "201", "--save-every", "100")
    load_script("analyze").write_trajectory("short")
    rows = list(csv.DictReader(open(paths.RESULTS / "short_trajectory.csv")))
    assert [r["epoch"] for r in rows] == ["0", "100", "200"]
    assert {"restricted_loss", "excluded_loss", "gini_embed", "gini_W_L", "sum_sq_weights"} <= rows[0].keys()


def test_sweep_sees_only_finished_runs(runs, monkeypatch):
    from grokking import sweeps
    train(monkeypatch, "--name", "wd3_s1", "--epochs", "1", "--save-every", "0")
    (paths.run_dir("wd3_s2") / "ckpt").mkdir(parents=True)                     # started, nothing logged
    grid = sweeps.grid_curves("wd")
    assert grid[("3", 1)]["epoch"] == [0.0]
    assert grid[("3", 2)] is None and grid[("0", 0)] is None
    (paths.run_dir("wd3_s1") / "done.json").unlink()                            # still running
    assert sweeps.finished_curves("wd3_s1") is None


def test_a_run_logged_to_epoch_39900_counts_as_finished(runs, monkeypatch):
    from grokking import curves as C
    from grokking import sweeps
    d = paths.run_dir("wd0_s1")
    d.mkdir(parents=True)
    (d / "metrics.csv").write_text("epoch,test_acc\n0,0.1\n39900,0.1\n")
    assert sweeps.finished_curves("wd0_s1")["epoch"] == [0.0, 39900.0]
    assert C.our_curves("wd0_s1")["test_acc"] == [0.1, 0.1]

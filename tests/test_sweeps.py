from grokking import curves as C
from grokking import sweeps


def curve(t_train, t_test, last=39900):
    """Metrics every 100 epochs: train accuracy reaches 1 at t_train, test accuracy passes 99% at t_test."""
    epochs = list(range(0, last + 1, 100))
    return {
        "epoch": epochs,
        "train_acc": [1.0 if e >= t_train else 0.5 for e in epochs],
        "test_acc": [1.0 if t_test is not None and e >= t_test else 0.05 for e in epochs],
        "train_loss": [1e-7 if e >= t_train else 1.0 for e in epochs],
        "test_loss": [1e-7 if t_test is not None and e >= t_test else 5.0 for e in epochs],
    }


def test_run_names_map_the_main_run():
    assert sweeps.run_name("wd", "1", 0) == "seed0"
    assert sweeps.run_name("wd", "1", 1) == "wd1_s1"
    assert sweeps.run_name("frac", "0.3", 0) == "seed0"
    assert sweeps.run_name("frac", "0.6", 2) == "frac0.6_s2"


def test_grok_epoch_is_censored_when_test_never_passes():
    assert C.grok_epoch(curve(300, 6800)) == 6800
    assert C.grok_epoch(curve(300, None)) == C.NEVER


def test_median_of_seeds_is_honest_about_censoring():
    assert C.median_of_seeds([5200, 6800, 8400]) == 6800
    assert C.median_of_seeds([C.NEVER, 2000, C.NEVER]) == C.NEVER
    assert C.median_of_seeds([1, None, 3]) is None


def test_fmt_epoch():
    assert C.fmt_epoch(21500) == "21 500"
    assert C.fmt_epoch(C.NEVER) == "> 40 000"
    assert C.fmt_epoch(None) == "—"


def test_timings_of_a_grokking_run():
    t = sweeps.timings(curve(300, 6800))
    assert (t["t_train"], t["t_test"], t["delay"]) == (300, 6800, 6500)
    assert t["max_test_acc"] == 1.0


def test_timings_without_generalisation_have_no_delay():
    t = sweeps.timings(curve(300, None))
    assert t["t_test"] == C.NEVER and t["delay"] is None


def _fraction_grid(make):
    return {(v, s): make(v, s) for v in sweeps.GRIDS["frac"] for s in sweeps.SEEDS}


def paper_like(v, s):
    f = float(v)
    if f <= 0.2:
        return curve(300, None)
    if f <= 0.5:
        return curve(300, int(round(6000 * 0.3 / f, -2)) + 100 * s)
    return curve(100, 600)


def test_hypotheses_on_paper_like_runs_are_all_confirmed():
    report = sweeps.fraction_report(_fraction_grid(paper_like))
    assert report.count("NOT confirmed") == 0
    assert report.count(": confirmed") == 4


def test_hypothesis_two_fails_when_one_run_has_no_delay():
    def fast_mid(v, s):
        return curve(300, 500) if (v, s) == ("0.5", 1) else paper_like(v, s)
    report = sweeps.fraction_report(_fraction_grid(fast_mid))
    assert "2. fractions 0.3-0.5" in report.split("NOT confirmed")[0].splitlines()[-1]


def test_hypotheses_wait_for_unfinished_runs():
    def missing(v, s):
        return None if v == "0.9" else paper_like(v, s)
    report = sweeps.fraction_report(_fraction_grid(missing))
    assert "4. fractions" not in report and "| 0.9 | 0 | not finished |" in report


def test_wd_table_reports_median_and_running():
    runs = {(v, s): curve(300, 1000 * (i + 1) + 100 * s) for i, v in enumerate(sweeps.GRIDS["wd"]) for s in sweeps.SEEDS}
    runs[("0", 2)] = None
    table = sweeps.wd_table(runs)
    assert "| 0.3 | 2 000 | 2 100 | 2 200 | 2 100 |" in table
    assert "| 0 | 1 000 | 1 100 | running | running |" in table


def test_timings_thresholds_are_strict():
    c = curve(300, 6800)
    c["train_loss"] = [1e-6] * len(c["epoch"])        # exactly 1e-6 is not below it
    c["test_loss"] = [1e-6] * len(c["epoch"])
    t = sweeps.timings(c)
    assert t["train_loss_1e-6"] is None and t["test_loss_1e-6"] is None


def test_boundary_delays_of_the_hypotheses():
    def edge(v, s):
        f = float(v)
        if f <= 0.2:
            return curve(300, None)
        if f <= 0.5:                                    # t_train exactly 1 000, delay exactly 1 000: holds
            return curve(1000, 2000 + 1000 * (5 - int(f * 10)))
        return curve(100, 1100)                         # delay exactly 1 000: hypothesis 4 fails
    report = sweeps.fraction_report(_fraction_grid(edge))
    assert "2. fractions 0.3-0.5: t_train <= 1 000 and delay >= 1 000 for every seed: confirmed" in report
    assert "4. fractions 0.6-0.9: delay < 1 000 for every seed: NOT confirmed" in report


def test_equal_medians_do_not_count_as_decreasing():
    def flat(v, s):
        return curve(300, None) if float(v) <= 0.2 else curve(300, 5000 if float(v) <= 0.4 else 600)
    assert "3. median t_test decreases 0.3 > 0.4 > 0.5 (5 000 / 5 000 / 600): NOT confirmed" in \
        sweeps.fraction_report(_fraction_grid(flat))


def test_a_high_fraction_run_that_never_generalises_fails_hypothesis_four():
    def stuck(v, s):
        return curve(300, None) if (v, s) == ("0.9", 0) else paper_like(v, s)
    assert "4. fractions 0.6-0.9: delay < 1 000 for every seed: NOT confirmed" in \
        sweeps.fraction_report(_fraction_grid(stuck))

import numpy as np

from peaks import match_peaks, rr_from_peaks
from reject_rules import classify_ref_rr, filter_rr_stream, rr_cv, short_rr_fraction


def test_match_peaks_unique_within_tolerance():
    ref = np.array([1.0, 2.0, 3.0])
    det = np.array([1.05, 1.12, 3.14])
    m = match_peaks(ref, det, tol_s=0.15)
    assert m["tp"] == 2
    assert m["fn"] == 1
    assert m["fp"] == 1


def test_rr_from_peaks_sorted_unique():
    rr, te = rr_from_peaks([0.0, 0.8, 0.8, 1.6])
    assert np.allclose(rr, [0.8, 0.8])


def test_short_abs_rule():
    assert classify_ref_rr(0.40, [0.8, 0.82, 0.79]) == "short_abs"
    assert classify_ref_rr(0.55, [0.8, 0.82, 0.79]) == "ok"


def test_relative_median_rule():
    acc = [0.90] * 5
    # 0.50 < 0.65*0.90 = 0.585
    assert classify_ref_rr(0.50, acc) == "short_rel_median"
    assert classify_ref_rr(0.60, acc) == "ok"


def test_amplitude_rule_only_on_shortish_rr():
    acc = [0.70] * 5
    assert classify_ref_rr(0.70, acc, peak=0.4, prev_peak=1.0) == "low_amp_prev"
    # long RR skips amplitude gate (AMP_FRAC_PREV_RR_MAX=0.9)
    assert classify_ref_rr(1.0, acc, peak=0.4, prev_peak=1.0) == "ok"


def test_filtering_short_rr_merges_intervals_and_lowers_cv():
    """Dropping true short AF RR must not just delete them: time is merged."""
    rng = np.random.default_rng(2)
    # Alternate 0.35 / 0.85 so mean ~0.6, high CV; abs gate drops 0.35
    rr = np.array([0.35 if i % 2 == 0 else 0.85 for i in range(80)], dtype=float)
    t = np.concatenate([[0.0], np.cumsum(rr)])
    raw_cv = rr_cv(rr)
    out = filter_rr_stream(t)
    kept_cv = rr_cv(out["kept_rr"])
    assert raw_cv > 0.2
    assert kept_cv < raw_cv
    # merged RR around 1.20 s, much more regular
    assert np.median(out["kept_rr"]) > 1.0
    assert short_rr_fraction(rr, 0.48) > 0.4
    assert short_rr_fraction(out["kept_rr"], 0.48) == 0.0


def test_relative_gate_cascade_uses_accepted_median_not_reference():
    # After dropping shorts, median stays long, so 0.50 vs 0.80 ref median
    # is still rejected relative to accepted median.
    t = [0.0]
    for _ in range(10):
        t.append(t[-1] + 0.80)
        t.append(t[-1] + 0.40)
    t = np.asarray(t)
    out = filter_rr_stream(t)
    assert out["n_kept"] < out["n_in"]
    assert np.all(out["kept_rr"] >= 0.48)

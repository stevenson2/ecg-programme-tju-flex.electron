import numpy as np

from af_backend import AFWindow
from events import analyzable_stats, predicted_alarm_episodes, reference_episodes
from labels import (
    Interval,
    parse_rhythm_code,
    rhythm_intervals_from_ann,
    second_true_label,
)
from metrics import bootstrap_ci, pool_records, record_eval


def test_parse_af_and_afl_distinct():
    assert parse_rhythm_code("(AFIB") == "AFIB"
    assert parse_rhythm_code("(AFL") == "AFL"
    assert parse_rhythm_code("(AFLT") == "AFL"
    assert parse_rhythm_code("(N") == "N"
    assert parse_rhythm_code("N") is None or parse_rhythm_code("N") == "N"


def test_rhythm_intervals_leading_gap_is_n():
    iv = rhythm_intervals_from_ann(
        samples=[1000, 5000],
        aux_notes=["(AFIB", "(N"],
        fs=100.0,
        duration_s=80.0,
    )
    assert iv[0].code == "N"
    assert iv[0].start_s == 0.0
    af = [x for x in iv if x.code == "AFIB"][0]
    assert abs(af.start_s - 10.0) < 1e-9


def test_af_only_excludes_afl_from_pos_and_fp_bucket():
    ivs = [
        Interval(0, 30, "AFIB"),
        Interval(30, 60, "AFL"),
        Interval(60, 120, "N"),
    ]
    y = second_true_label(ivs, 120, scheme="af_only")
    assert y[10] == 1
    assert y[40] == -1
    assert y[80] == 0
    y2 = second_true_label(ivs, 120, scheme="af_or_afl")
    assert y2[40] == 1


def test_missed_af_because_undetermined_still_in_denominator():
    y = np.zeros(100, dtype=np.int8)
    y[20:80] = 1
    pred = np.full(100, 2, dtype=np.int8)  # never decides
    ev = record_eval(y, pred, windows=[])
    assert ev["event_fn"] == 1
    assert ev["event_tp"] == 0
    assert ev["event_fn_undetermined"] == 1
    assert ev["event_se"] == 0.0
    assert ev["undetermined_frac"] == 1.0
    assert ev["hours_total"] == 100 / 3600.0
    # denominator for FP/h is total time, not analyzable time
    assert ev["fp_per_hour_total"] == 0.0


def test_called_normal_is_distinct_from_undetermined_miss():
    y = np.zeros(80, dtype=np.int8)
    y[10:70] = 1
    pred = np.zeros(80, dtype=np.int8)
    ev = record_eval(y, pred, windows=[])
    assert ev["event_fn_called_normal"] == 1
    assert ev["event_fn_undetermined"] == 0


def test_false_alarm_uses_total_hours_including_undetermined():
    y = np.zeros(3600, dtype=np.int8)
    pred = np.full(3600, 2, dtype=np.int8)
    pred[100:130] = 1  # 30 s alarm + rest undetermined
    ev = record_eval(y, pred, windows=[])
    assert ev["event_fp"] == 1
    assert abs(ev["fp_per_hour_total"] - 1.0) < 1e-9
    assert ev["undetermined_frac"] > 0.9


def test_detection_delay():
    y = np.zeros(100, dtype=np.int8)
    y[10:80] = 1
    pred = np.zeros(100, dtype=np.int8)
    pred[25:40] = 1
    ev = record_eval(y, pred, windows=[])
    assert ev["event_tp"] == 1
    assert ev["mean_delay_s"] == 15.0


def test_afl_called_af_is_not_fp_in_af_only():
    y = np.zeros(100, dtype=np.int8)
    y[:] = -1  # entire record AFL
    pred = np.zeros(100, dtype=np.int8)
    pred[10:40] = 1
    ev = record_eval(y, pred, windows=[])
    assert ev["event_fp"] == 0
    assert ev["n_ref_events"] == 0
    assert ev["afl_called_af_s"] == 30.0


def test_window_undetermined_on_af_counts_as_fn():
    y = np.ones(20, dtype=np.int8)
    pred = np.full(20, 2, dtype=np.int8)
    w = AFWindow(
        t_end=20, t_start=10, label=2, cv=0, entropy=0, score=0, n_rr=3,
        window_ready=True, reason="min_rr",
    )
    ev = record_eval(y, pred, windows=[w])
    assert ev["window_fn"] == 1
    assert ev["window_undet_pos"] == 1
    assert ev["window_se"] == 0.0


def test_bootstrap_runs_and_is_deterministic():
    rows = []
    for i in range(6):
        y = np.zeros(200, dtype=np.int8)
        y[20:80] = 1 if i % 2 == 0 else 0
        pred = np.zeros(200, dtype=np.int8)
        if i % 2 == 0:
            pred[30:50] = 1
        rows.append(record_eval(y, pred, windows=[]))
    a = bootstrap_ci(rows, n_boot=50, seed=1)
    b = bootstrap_ci(rows, n_boot=50, seed=1)
    assert a["event_se"] == b["event_se"]
    pooled = pool_records(rows)
    assert pooled["n_records"] == 6
    assert 0.0 <= pooled["event_se"] <= 1.0


def test_pooled_duration_se_uses_reference_af_seconds():
    """Predicted AF seconds must not replace the reference AF denominator."""
    y = np.zeros(100, dtype=np.int8)
    y[:50] = 1
    pred = np.zeros(100, dtype=np.int8)
    pred[:10] = 1
    ev = record_eval(y, pred, windows=[])
    assert ev["n_af_s"] == 50
    assert abs(ev["duration_se"] - 0.2) < 1e-9
    pooled = pool_records([ev, ev])
    assert pooled["n_af_s"] == 100
    assert abs(pooled["duration_se"] - 0.2) < 1e-9


def test_analyzable_longest_run():
    pred = np.zeros(50, dtype=np.int8)
    pred[5:8] = 2
    pred[10:20] = 2
    pred[21:24] = 2
    st = analyzable_stats(pred)
    assert st["longest_undetermined_s"] == 10.0
    assert st["undetermined_s"] == 3 + 10 + 3

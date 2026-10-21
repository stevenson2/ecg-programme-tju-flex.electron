import numpy as np

from af_backend import FirmwareAF, shannon_entropy_fw, windows_to_second_labels
from constants import AF_CV_THR, AF_ENT_THR


def test_entropy_uniform_high():
    rng = np.random.default_rng(0)
    rr = rng.uniform(0.3, 1.5, size=16)
    ent = shannon_entropy_fw(rr)
    # 16 samples in 16 bins, not necessarily one each, but should be well above 1.2
    assert ent > 1.2


def test_entropy_constant_zero():
    assert shannon_entropy_fw([0.8] * 12) == 0.0


def test_entropy_clip_and_last_bin():
    # 1.5 clips to hi then bin 16 -> 15; 0.3 -> bin 0
    ent = shannon_entropy_fw([0.3, 1.5, 0.3, 1.5, 0.3, 1.5])
    assert ent > 0.0
    # two occupied bins -> ln(2)
    assert abs(ent - np.log(2)) < 1e-9


def test_regular_sinus_not_af():
    rr = np.full(40, 60.0 / 70.0)
    af = FirmwareAF()
    wins = af.evaluate_rr(rr)
    decided = [w for w in wins if w.label in (0, 1)]
    assert decided, "window should fill on ~10 s of 70 bpm"
    assert all(w.label == 0 for w in decided)
    assert all(w.cv < AF_CV_THR for w in decided)


def test_irregular_af_like_is_af():
    rng = np.random.default_rng(1)
    t = 0.0
    rr = []
    while t < 60:
        r = float(rng.uniform(0.32, 0.90))
        rr.append(r)
        t += r
    af = FirmwareAF()
    wins = af.evaluate_rr(rr)
    decided = [w for w in wins if w.label in (0, 1)]
    assert decided
    assert any(w.label == 1 for w in decided)
    assert decided[0].cv > AF_CV_THR
    assert decided[0].entropy > AF_ENT_THR


def test_af_backend_accepts_short_rr_that_hrprocess_would_drop():
    """Hypothesis hinge: AF backend itself does not reject 320 ms RR."""
    rr = [0.32, 0.40, 0.36, 0.50, 0.33, 0.70, 0.38, 0.55, 0.34, 0.80] * 3
    af = FirmwareAF()
    wins = af.evaluate_rr(rr)
    assert any(w.n_rr >= 6 and w.window_ready for w in wins)
    # short RR must actually enter the window, not be clipped by AF
    assert min(rr) < 0.48


def test_too_few_rr_is_undetermined_not_dropped():
    rr = [2.0, 2.0, 2.0, 2.0, 2.0]  # 10 s, only 5 RR
    af = FirmwareAF()
    wins = af.evaluate_rr(rr)
    assert wins
    assert all(w.label == 2 for w in wins)
    assert any(w.reason == "min_rr" for w in wins)


def test_sqi_gate_undetermined():
    rr = np.full(20, 0.7)
    sqi = np.full(20, 0.1)
    af = FirmwareAF()
    wins = af.evaluate_rr(rr, sqi)
    assert wins
    assert all(w.label == 2 for w in wins)
    assert any(w.reason == "sqi" for w in wins)


def test_slide_is_half_window():
    rr = np.full(40, 0.5)  # 20 s
    af = FirmwareAF()
    wins = [w for w in af.evaluate_rr(rr) if w.label in (0, 1)]
    assert len(wins) >= 2
    dt = wins[1].t_end - wins[0].t_end
    assert 4.0 <= dt <= 6.5  # ~5 s, allow compact/elapsed drift


def test_relabel_windows_keeps_geometry():
    from af_backend import relabel_windows, windows_to_array
    rng = np.random.default_rng(3)
    t = 0.0
    rr = []
    while t < 40:
        r = float(rng.uniform(0.35, 0.9))
        rr.append(r)
        t += r
    scored = FirmwareAF(cv_thr=-1.0, ent_thr=-1.0).evaluate_rr(rr)
    tight = relabel_windows(scored, cv_thr=0.5, ent_thr=3.0)  # almost never AF
    loose = FirmwareAF(cv_thr=0.5, ent_thr=3.0).evaluate_rr(rr)
    assert [w.t_end for w in tight] == [w.t_end for w in loose]
    assert [w.label for w in tight] == [w.label for w in loose]
    tight_a = relabel_windows(windows_to_array(scored), cv_thr=0.5, ent_thr=3.0)
    assert list(tight_a["t_end"]) == [w.t_end for w in loose]
    assert list(tight_a["label"]) == [w.label for w in loose]


def test_second_labels_keep_unpainted_undetermined():
    rr = np.full(30, 0.5)  # 15 s of beats
    af = FirmwareAF()
    wins = af.evaluate_rr(rr)
    lab = windows_to_second_labels(wins, duration_s=25.0)
    assert np.any(lab == 0)
    assert lab[-1] == 2  # seconds after last window stay undetermined
    assert np.mean(lab == 2) > 0

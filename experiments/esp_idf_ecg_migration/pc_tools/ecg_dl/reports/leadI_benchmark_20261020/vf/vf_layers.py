#!/usr/bin/env python3
"""Score the PR #2 VF chain as four layers, without retuning it.

L0 is the logistic score and the current-window suspect bit (``p >= 0.15``).
L1 is ``vfAlarm``: the current window and the previous window are both
suspect. L2 is the gate already in ``main.cc``: a confirmed window is
released only after a QRS has been seen, the gap back to that QRS is
strictly more than 2500 ms, and the completing window's RMS is at least
0.05 mV. L3 is the user-visible alarm, BLE column 8. A VF hit by itself
does not latch that bit. It is telemetry unless another source triggers
in the same second or the latch is already held.

``event_ge15`` is the PR #2 count: shockable episodes of duration >= 15 s
that overlap any positive window, at any delay. Se15 uses that same
denominator and counts a hit only when the PR #2 first-hit delay is
<= 15 s. The two are not aliases.

Nothing here changes ``VF_THETA``, the RMS floor, the beat gate, or the
latch. No database is read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval_vf_rules import first_hit_delay
from vf_metrics import apply_profile, beat_gate_open, confirm_from_scores, window_end_frame

# Shipped constants. Names match main.cc / vf_detect.h / ecg_protocol_generated.h.
THETA = 0.15
RMS_MIN = 0.05
GATE_MS = 2500
HOLD_S = 12
AI_DENSITY_TH = 8
AI_SQI_MIN = 0.80
REL_DENSITY_TH = 1
REL_SQI_MIN = 0.50
COOLDOWN_S = 25
WIN_S = 5.0
EVENT_GE15_MIN_DUR_S = 15.0
SE15_MAX_DELAY_S = 15.0

ASRC_AI = 0x01
ASRC_RS = 0x02
ASRC_FLAT = 0x04
ASRC_VF = 0x08
ASRC_LEADOFF = 0x10


def l0_suspect(scores, theta: float = THETA) -> np.ndarray:
    """Current window suspect. This is a score decision, not an alarm."""
    return np.asarray(scores, dtype=np.float64) >= float(theta)


def l1_vf_alarm(scores, theta: float = THETA) -> np.ndarray:
    """``vfAlarm`` from two consecutive suspect windows. The first window is never on."""
    return confirm_from_scores(np.asarray(scores, dtype=np.float64), float(theta))


def l2_gate(n_win: int, beat_frames, gate_ms: int = GATE_MS, require_arm: bool = True) -> np.ndarray:
    beats = np.asarray(beat_frames, dtype=np.int32)
    return np.array(
        [beat_gate_open(window_end_frame(i), beats, gate_ms, require_arm) for i in range(n_win)],
        dtype=bool,
    )


def l2_vf_hit(vf_alarm, rms, beat_frames, gate_ms: int = GATE_MS, rms_min: float = RMS_MIN) -> np.ndarray:
    """L1 after the existing RMS floor and beat gate. Not the user alarm."""
    alarm = np.asarray(vf_alarm, dtype=bool)
    gate = l2_gate(alarm.size, beat_frames, gate_ms, True)
    return apply_profile(alarm, rms, gate, True, True, rms_min)


class UserAlarm:
    """``alarmSecondTick`` from main.cc, including the VF corroboration rule.

    ``ai_density`` is ``alarmAiDensity(ALARM_AI_WIN)``. ``ai_rel_density`` is
    ``alarmAiDensity(ALARM_REL_WIN)``; it defaults to ``ai_density`` when the
    caller only has one number. Both thresholds stay the shipped ones.
    """

    def __init__(self) -> None:
        self.latched = False
        self.src = 0
        self.last_trig_sec = 0
        self.cooldown_until = 0

    def tick(
        self,
        now_sec: int,
        rule_hits: int = 0,
        sqi_min: float = 1.0,
        vf_hit: bool = False,
        beat_recent: bool = False,
        ai_density: int = 0,
        ai_rel_density: int | None = None,
    ) -> dict:
        if ai_rel_density is None:
            ai_rel_density = ai_density
        was_latched = self.latched
        # Call site keeps VF and AI out of ruleHits. VF is applied below.
        trig = int(rule_hits) & (ASRC_RS | ASRC_FLAT | ASRC_LEADOFF)
        if ai_density >= AI_DENSITY_TH and sqi_min >= AI_SQI_MIN:
            if self.latched or now_sec >= self.cooldown_until:
                trig |= ASRC_AI
        trig_before_vf = trig
        vf_added = False
        if vf_hit and (trig != 0 or self.latched):
            trig |= ASRC_VF
            vf_added = True
        if trig != 0:
            self.latched = True
            self.src |= trig
            self.last_trig_sec = int(now_sec)
        elif (
            self.latched
            and (int(now_sec) - self.last_trig_sec) >= HOLD_S
            and sqi_min >= REL_SQI_MIN
            and beat_recent
            and ai_rel_density <= REL_DENSITY_TH
        ):
            self.latched = False
            self.src = 0
            self.cooldown_until = int(now_sec) + COOLDOWN_S
        # VF is added only when another source is already in ``trig`` or the
        # latch is already held, so a rising edge cannot be credited to VF.
        vf_started = (not was_latched) and self.latched and vf_added and trig_before_vf == 0
        return {
            "latched": self.latched,
            "ble_alarm": 1 if self.latched else 0,
            "ble_asrc": int(self.src) if self.latched else 0,
            "vf_in_ble_asrc": bool(self.latched and (self.src & ASRC_VF)),
            "vf_telemetry_only": bool(vf_hit and not vf_added),
            "vf_started_latch": bool(vf_started),
        }


def _ratio(hits: int, n: int) -> dict:
    return {
        "hits": int(hits),
        "n": int(n),
        "est": None if n <= 0 else hits / n,
    }


def score_episodes(flags, episodes) -> dict:
    """Split the legacy duration filter from the 15 s latency budget.

    Both denominators are shockable episodes with duration >= 15 s.
    ``event_ge15`` hits on any overlap. ``se15`` hits only when
    ``first_hit_delay`` is <= 15 s. Episodes shorter than 15 s are in
    neither denominator.
    """
    flags = np.asarray(flags, dtype=bool)
    ge_hits = se_hits = n = 0
    for ep in episodes:
        a = float(ep[0])
        b = float(ep[1])
        if (b - a) < EVENT_GE15_MIN_DUR_S:
            continue
        n += 1
        dly = first_hit_delay(flags, a, b)
        if dly is None:
            continue
        ge_hits += 1
        if float(dly) <= SE15_MAX_DELAY_S:
            se_hits += 1
    return {
        "event_ge15": _ratio(ge_hits, n),
        "se15": _ratio(se_hits, n),
        "definition": {
            "event_ge15": "duration >= 15 s and any overlapping positive window",
            "se15": "same episodes, first-hit delay <= 15 s using the PR #2 window-end offset",
        },
    }


def score_user_alarm_seconds(latched, episodes) -> dict:
    """Episode scores for BLE column 8.

    Second ``s`` covers ``[s, s+1)``. The decision is known at the start of
    that second, so the delay is ``s - onset``, not the window-end formula
    used for L1 and L2.
    """
    latched = np.asarray(latched, dtype=bool)
    ge_hits = se_hits = n = 0
    for ep in episodes:
        a = float(ep[0])
        b = float(ep[1])
        if (b - a) < EVENT_GE15_MIN_DUR_S:
            continue
        n += 1
        dly = None
        for s, flag in enumerate(latched):
            if not flag:
                continue
            t0 = float(s)
            t1 = float(s + 1)
            if min(t1, b) > max(t0, a):
                dly = t0 - a
                break
        if dly is None:
            continue
        ge_hits += 1
        if dly <= SE15_MAX_DELAY_S:
            se_hits += 1
    return {"event_ge15": _ratio(ge_hits, n), "se15": _ratio(se_hits, n)}


def pool_episode_scores(parts: list[dict]) -> dict:
    out = {}
    for key in ("event_ge15", "se15"):
        hits = sum(int(p[key]["hits"]) for p in parts)
        n = sum(int(p[key]["n"]) for p in parts)
        out[key] = _ratio(hits, n)
    return out


def _vf_hit_tick(l2_flags: np.ndarray, n_sec: int) -> np.ndarray:
    """Map a completing L2 window onto the second tick that consumes it.

    The window's last sample is 4 ms before the nominal end. ``s_vfHitSec``
    is consumed on the next ``nowSec`` change, which is ``floor(t) + 1``.
    """
    ticks = np.zeros(n_sec, dtype=bool)
    for i, flag in enumerate(np.asarray(l2_flags, dtype=bool)):
        if not flag:
            continue
        complete = WIN_S * (i + 1) - 0.004
        tick = int(np.floor(complete)) + 1
        if 0 <= tick < n_sec:
            ticks[tick] = True
    return ticks


def trace_record(
    scores,
    rms,
    beat_frames,
    episodes,
    rule_hits_by_sec: dict | None = None,
    ai_density_by_sec: dict | None = None,
    sqi_min: float = 1.0,
    beat_recent: bool = False,
) -> dict:
    """One synthetic record, four layers, no waveform replay."""
    scores = np.asarray(scores, dtype=np.float64)
    rms = np.asarray(rms, dtype=np.float64)
    suspect = l0_suspect(scores)
    vf_alarm = l1_vf_alarm(scores)
    vf_hit = l2_vf_hit(vf_alarm, rms, beat_frames)
    n_sec = int(scores.size) * int(WIN_S) + 1
    tick_hit = _vf_hit_tick(vf_hit, n_sec)
    rules = rule_hits_by_sec or {}
    densities = ai_density_by_sec or {}
    user = UserAlarm()
    ble = np.zeros(n_sec, dtype=np.int64)
    asrc = np.zeros(n_sec, dtype=np.int64)
    for s in range(n_sec):
        out = user.tick(
            s,
            rule_hits=int(rules.get(s, 0)),
            sqi_min=sqi_min,
            vf_hit=bool(tick_hit[s]),
            beat_recent=beat_recent,
            ai_density=int(densities.get(s, 0)),
        )
        ble[s] = out["ble_alarm"]
        asrc[s] = out["ble_asrc"]
    l1_scores = score_episodes(vf_alarm, episodes)
    l2_scores = score_episodes(vf_hit, episodes)
    l3_scores = score_user_alarm_seconds(ble, episodes)
    return {
        "L0_suspect": suspect.astype(np.int64),
        "L1_vfAlarm": vf_alarm.astype(np.int64),
        "L2_vfHit": vf_hit.astype(np.int64),
        "L2_tick_sec": tick_hit.astype(np.int64),
        "L3_user_alarm_sec": ble,
        "L3_ble_asrc_sec": asrc,
        "by_layer": {
            "L0": {
                "n_windows": int(scores.size),
                "n_suspect": int(np.count_nonzero(suspect)),
                "scores": [float(v) for v in scores],
            },
            "L1": l1_scores,
            "L2": l2_scores,
            "L3": l3_scores,
        },
    }


def _ticks(alarm: UserAlarm, steps: list[dict]) -> list[dict]:
    out = []
    for step in steps:
        sec = int(step["sec"])
        row = alarm.tick(
            sec,
            rule_hits=int(step.get("rule_hits", 0)),
            sqi_min=float(step.get("sqi_min", 1.0)),
            vf_hit=bool(step.get("vf_hit", False)),
            beat_recent=bool(step.get("beat_recent", False)),
            ai_density=int(step.get("ai_density", 0)),
        )
        out.append({"sec": sec, **row})
    return out


def build_v01_evidence() -> dict:
    """Synthetic counterexamples only. No database, no new gate, no retune."""
    from eval_vf_rules import event_counts

    late = np.zeros(8, dtype=bool)
    late[6] = True
    early = np.zeros(8, dtype=bool)
    early[1] = True
    on_budget = np.zeros(8, dtype=bool)
    on_budget[2] = True
    short = np.zeros(8, dtype=bool)
    short[1] = True
    ep40 = [(0.0, 40.0, "VF")]
    late_score = score_episodes(late, ep40)
    early_score = score_episodes(early, ep40)
    budget_score = score_episodes(on_budget, ep40)
    short_score = score_episodes(short, [(0.0, 12.0, "VF")])
    pooled = pool_episode_scores([early_score, late_score])
    legacy_late = event_counts(late, ep40, 15.0)

    scores = np.array([0.01, 0.01, 0.01, 0.01, 0.80, 0.80, 0.01, 0.01])
    traced = trace_record(
        scores,
        np.full(8, 0.40),
        beat_frames=np.array([0], dtype=np.int32),
        episodes=ep40,
    )
    consumed = [int(i) for i, flag in enumerate(traced["L2_tick_sec"]) if flag]

    vf_only = _ticks(UserAlarm(), [
        {"sec": s, "vf_hit": True, "beat_recent": False} for s in range(30)
    ])
    rs_only = _ticks(UserAlarm(), [
        {"sec": 0, "rule_hits": ASRC_RS, "beat_recent": True},
        {"sec": 11, "beat_recent": True},
    ])
    extended = _ticks(UserAlarm(), [
        {"sec": 0, "rule_hits": ASRC_RS},
        {"sec": 5, "vf_hit": True},
        {"sec": 16, "beat_recent": True},
        {"sec": 17, "beat_recent": True},
    ])
    together = _ticks(UserAlarm(), [
        {"sec": 3, "rule_hits": ASRC_RS, "vf_hit": True},
    ])
    ai_only = _ticks(UserAlarm(), [
        {"sec": 0, "ai_density": 8, "sqi_min": 0.80},
    ])
    ai_and_vf = _ticks(UserAlarm(), [
        {"sec": 0, "ai_density": 8, "sqi_min": 0.80, "vf_hit": True},
    ])
    cooled = UserAlarm()
    cool_rows = _ticks(cooled, [
        {"sec": 0, "ai_density": 8, "sqi_min": 0.90, "beat_recent": True},
        {"sec": 12, "beat_recent": True, "sqi_min": 1.0, "ai_density": 0},
    ])
    after = cooled.tick(13, rule_hits=0, sqi_min=0.90, vf_hit=False, beat_recent=True, ai_density=8)

    stored_block = json.loads(
        (Path(__file__).resolve().parent / "results" / "summary.json").read_text(encoding="utf-8")
    )["profiles"]["firmware"]["vfdb"]

    def slim(rows: list[dict]) -> list[dict]:
        keep = ("sec", "ble_alarm", "ble_asrc", "vf_in_ble_asrc", "vf_telemetry_only", "vf_started_latch")
        return [{k: row[k] for k in keep} for row in rows]

    return {
        "input_commit": "6913749f90db15fb9e0acc983dfc5d02e0c1d5a5",
        "pr": 2,
        "packet_path": r"C:\ecg_data\round_20261021\next_packets\V01_PR2契约适配与小复现_CursorGrok.md",
        "packet_readable_on_runner": False,
        "e02_contract_received": False,
        "heavy_slot_records": 0,
        "heavy_slot_limit": 3,
        "full_library_replay": False,
        "thresholds_changed": False,
        "new_gates": False,
        "product_branch_switched": False,
        "flashed": False,
        "stored_pr2_summary": {
            "profile": "firmware",
            "database": "vfdb",
            "event_ge15": stored_block["event_ge15"],
            "has_se15_key": "se15" in stored_block,
            "reinterpreted": False,
        },
        "definitions": {
            "L0": "logistic score and current-window suspect, score >= 0.15",
            "L1": "vfAlarm, two consecutive suspect windows",
            "L2": "L1 and RMS >= 0.05 mV and armed beat gap > 2500 ms",
            "L3": "BLE column 8, alarmSecondTick latch; VF alone does not latch",
            "event_ge15": "shockable episodes with duration >= 15 s and any overlap",
            "se15": "same denominator as event_ge15, first-hit delay <= 15 s",
        },
        "delays_s": {
            "late_window_6": first_hit_delay(late, 0.0, 40.0),
            "early_window_1": first_hit_delay(early, 0.0, 40.0),
            "budget_window_2": first_hit_delay(on_budget, 0.0, 40.0),
            "trace_l1_window_5": first_hit_delay(traced["L1_vfAlarm"].astype(bool), 0.0, 40.0),
        },
        "cases": {
            "late_overlap": {
                "event_ge15": late_score["event_ge15"],
                "se15": late_score["se15"],
                "legacy_event_counts_hits": int(legacy_late["hits"]),
                "legacy_matches_event_ge15": int(legacy_late["hits"]) == late_score["event_ge15"]["hits"],
                "legacy_matches_se15": int(legacy_late["hits"]) == late_score["se15"]["hits"],
            },
            "pool_early_and_late": pooled,
            "window_end_15s": {
                "event_ge15": budget_score["event_ge15"],
                "se15": budget_score["se15"],
            },
            "duration_12s_in_neither_denominator": {
                "event_ge15": short_score["event_ge15"],
                "se15": short_score["se15"],
            },
            "one_record_four_layers": {
                "L0_suspect": traced["L0_suspect"].astype(int).tolist(),
                "L1_vfAlarm": traced["L1_vfAlarm"].astype(int).tolist(),
                "L2_vfHit": traced["L2_vfHit"].astype(int).tolist(),
                "L2_consumed_at_sec": consumed,
                "L3_user_alarm_any": int(np.count_nonzero(traced["L3_user_alarm_sec"])),
                "L0_n_suspect": traced["by_layer"]["L0"]["n_suspect"],
                "L0_has_se15": "se15" in traced["by_layer"]["L0"],
                "L1": {
                    "event_ge15": traced["by_layer"]["L1"]["event_ge15"],
                    "se15": traced["by_layer"]["L1"]["se15"],
                },
                "L2": {
                    "event_ge15": traced["by_layer"]["L2"]["event_ge15"],
                    "se15": traced["by_layer"]["L2"]["se15"],
                },
                "L3": {
                    "event_ge15": traced["by_layer"]["L3"]["event_ge15"],
                    "se15": traced["by_layer"]["L3"]["se15"],
                },
            },
            "vf_hit_alone_30s": {
                "n": len(vf_only),
                "ble_alarm_sum": int(sum(r["ble_alarm"] for r in vf_only)),
                "ble_asrc_sum": int(sum(r["ble_asrc"] for r in vf_only)),
                "telemetry_only_all": all(r["vf_telemetry_only"] for r in vf_only),
                "vf_started_latch_any": any(r["vf_started_latch"] for r in vf_only),
            },
            "rs_latch_without_vf": slim(rs_only),
            "rs_then_vf_extends_hold": slim(extended),
            "same_second_rs_and_vf": slim(together),
            "ai_latch_without_vf": slim(ai_only),
            "ai_and_vf_same_second": slim(ai_and_vf),
            "ai_cooldown_blocks_relatch": {
                "rows": slim(cool_rows),
                "sec_13_ai_again": {
                    "ble_alarm": after["ble_alarm"],
                    "ble_asrc": after["ble_asrc"],
                },
            },
        },
    }

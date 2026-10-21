#!/usr/bin/env python3
"""Synthetic counterexamples for the PR #2 VF scoring contract.

These traces are constructed. They do not replay VFDB, CUDB, or MIT-BIH.
Thresholds and gates are the ones already compiled into PR #2; the tests
only check that the four layers stay distinct and that the legacy
``event_ge15`` count is not reported as Se15.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from eval_vf_rules import event_counts
from vf_metrics import beat_gate_open, confirm_from_scores, window_end_frame

HERE = Path(__file__).resolve().parent
ROOT = HERE
for _ in range(12):
    if (ROOT / "experiments/esp_idf_ecg_migration/main/main.cc").exists():
        break
    ROOT = ROOT.parent
MAIN = ROOT / "experiments/esp_idf_ecg_migration/main/main.cc"
HEADER = ROOT / "experiments/esp_idf_ecg_migration/components/ecg_core/include/vf_detect/vf_detect.h"


class LayerContractTest(unittest.TestCase):
    def test_l0_is_score_compared_with_shipped_theta(self):
        from vf_layers import THETA, l0_suspect

        self.assertEqual(THETA, 0.15)
        scores = np.array([0.14, 0.15, 0.149, 0.90], dtype=np.float64)
        suspect = l0_suspect(scores)
        np.testing.assert_array_equal(suspect, [False, True, False, True])

    def test_l1_needs_two_consecutive_suspect_windows(self):
        from vf_layers import l1_vf_alarm

        scores = np.array([0.90, 0.90, 0.10, 0.90], dtype=np.float64)
        alarm = l1_vf_alarm(scores)
        np.testing.assert_array_equal(alarm, [False, True, False, False])
        np.testing.assert_array_equal(alarm, confirm_from_scores(scores, 0.15))
        alone = l1_vf_alarm(np.array([0.99]))
        self.assertFalse(bool(alone[0]))

    def test_l1_does_not_open_l2_when_an_existing_gate_is_closed(self):
        from vf_layers import l1_vf_alarm, l2_vf_hit

        scores = np.array([0.80, 0.80], dtype=np.float64)
        alarm = l1_vf_alarm(scores)
        self.assertTrue(bool(alarm[1]))

        # RMS floor already in main.cc. Below 0.05 mV the VF bit is not released.
        low_rms = l2_vf_hit(alarm, np.array([0.40, 0.04]), np.array([0], dtype=np.int32))
        self.assertFalse(bool(low_rms[1]))

        # Gap of exactly 2500 ms stays closed (`>` in main.cc).
        end = window_end_frame(1)
        beat_exact = np.array([end - 1250], dtype=np.int32)
        self.assertFalse(beat_gate_open(end, beat_exact, 2500, True))
        closed = l2_vf_hit(alarm, np.array([0.40, 0.40]), beat_exact)
        self.assertFalse(bool(closed[1]))

        # No QRS in the session: s_beatEverSeen stays false.
        unarmed = l2_vf_hit(alarm, np.array([0.40, 0.40]), np.zeros(0, dtype=np.int32))
        self.assertFalse(bool(unarmed[1]))

        beat_open = np.array([end - 1251], dtype=np.int32)
        self.assertTrue(beat_gate_open(end, beat_open, 2500, True))
        opened = l2_vf_hit(alarm, np.array([0.40, 0.40]), beat_open)
        np.testing.assert_array_equal(opened, [False, True])

    def test_vf_bit_in_rule_hits_does_not_latch(self):
        from vf_layers import ASRC_VF, UserAlarm

        out = UserAlarm().tick(0, rule_hits=ASRC_VF, vf_hit=False)
        self.assertEqual(out["ble_alarm"], 0)
        self.assertEqual(out["ble_asrc"], 0)

    def test_vf_hit_alone_is_not_the_user_alarm(self):
        from vf_layers import UserAlarm

        alarm = UserAlarm()
        for sec in range(30):
            out = alarm.tick(
                sec,
                rule_hits=0,
                sqi_min=1.0,
                vf_hit=True,
                beat_recent=False,
                ai_density=0,
            )
            self.assertEqual(out["ble_alarm"], 0)
            self.assertEqual(out["ble_asrc"], 0)
            self.assertTrue(out["vf_telemetry_only"])
            self.assertFalse(out["vf_started_latch"])

        # AI just under the shipped 8/10 density, and SQI just under 0.80,
        # still do not let VF start the latch.
        blocked = UserAlarm()
        for density, sqi in ((7, 1.0), (10, 0.79)):
            out = blocked.tick(0, rule_hits=0, sqi_min=sqi, vf_hit=True, beat_recent=False, ai_density=density)
            self.assertEqual(out["ble_alarm"], 0, msg=f"density={density} sqi={sqi}")

    def test_another_source_can_latch_without_vf(self):
        from vf_layers import ASRC_RS, UserAlarm

        alarm = UserAlarm()
        out = alarm.tick(0, rule_hits=ASRC_RS, sqi_min=1.0, vf_hit=False, beat_recent=True, ai_density=0)
        self.assertEqual(out["ble_alarm"], 1)
        self.assertEqual(out["ble_asrc"], ASRC_RS)
        self.assertFalse(out["vf_in_ble_asrc"])
        # Hold is 12 s. VF is still false, so this is a user alarm with no vfAlarm.
        held = alarm.tick(11, rule_hits=0, sqi_min=1.0, vf_hit=False, beat_recent=True, ai_density=0)
        self.assertEqual(held["ble_alarm"], 1)
        self.assertEqual(held["ble_asrc"], ASRC_RS)

    def test_vf_only_joins_asrc_after_another_source_has_latched(self):
        from vf_layers import ASRC_RS, ASRC_VF, UserAlarm

        alarm = UserAlarm()
        first = alarm.tick(0, rule_hits=ASRC_RS, vf_hit=False, beat_recent=False, ai_density=0)
        self.assertEqual(first["ble_asrc"], ASRC_RS)
        extended = alarm.tick(5, rule_hits=0, vf_hit=True, beat_recent=False, ai_density=0, sqi_min=1.0)
        self.assertEqual(extended["ble_alarm"], 1)
        self.assertEqual(extended["ble_asrc"], ASRC_RS | ASRC_VF)
        self.assertFalse(extended["vf_started_latch"])
        # The VF hit moved the hold anchor from t=0 to t=5, which is the
        # existing extend path, not a new gate.
        still = alarm.tick(16, rule_hits=0, vf_hit=False, beat_recent=True, ai_density=0, sqi_min=1.0)
        self.assertEqual(still["ble_alarm"], 1)
        cleared = alarm.tick(17, rule_hits=0, vf_hit=False, beat_recent=True, ai_density=0, sqi_min=1.0)
        self.assertEqual(cleared["ble_alarm"], 0)
        self.assertEqual(cleared["ble_asrc"], 0)

    def test_same_second_rule_and_vf_latch_together(self):
        from vf_layers import ASRC_RS, ASRC_VF, UserAlarm

        alarm = UserAlarm()
        out = alarm.tick(3, rule_hits=ASRC_RS, vf_hit=True, beat_recent=False, ai_density=0, sqi_min=1.0)
        self.assertEqual(out["ble_alarm"], 1)
        self.assertEqual(out["ble_asrc"], ASRC_RS | ASRC_VF)
        self.assertFalse(out["vf_started_latch"])
        self.assertFalse(out["vf_telemetry_only"])

    def test_event_ge15_is_not_se15(self):
        from vf_layers import score_episodes

        # Eight 5 s windows. Episode lasts 40 s, so it is inside the old
        # duration >= 15 s denominator. The only alarm window ends at 35 s.
        late = np.zeros(8, dtype=bool)
        late[6] = True
        late_score = score_episodes(late, [(0.0, 40.0, "VF")])
        self.assertEqual(late_score["event_ge15"]["hits"], 1)
        self.assertEqual(late_score["event_ge15"]["n"], 1)
        self.assertEqual(late_score["se15"]["hits"], 0)
        self.assertEqual(late_score["se15"]["n"], 1)
        self.assertNotEqual(late_score["event_ge15"]["est"], late_score["se15"]["est"])

        legacy = event_counts(late, [(0.0, 40.0, "VF")], 15.0)
        self.assertEqual(legacy["hits"], late_score["event_ge15"]["hits"])
        self.assertNotEqual(legacy["hits"], late_score["se15"]["hits"])

        # Two episodes, same denominator. Old overlap sensitivity is 2/2.
        # Latency <= 15 s keeps only the early one, so Se15 is 1/2.
        early = np.zeros(8, dtype=bool)
        early[1] = True
        pooled_flags = [early, late]
        pooled_eps = [[(0.0, 40.0, "VF")], [(0.0, 40.0, "VFIB")]]
        # Score them independently and pool, matching the PR2 record sum.
        from vf_layers import pool_episode_scores

        pooled = pool_episode_scores(
            [score_episodes(flags, eps) for flags, eps in zip(pooled_flags, pooled_eps)]
        )
        self.assertEqual(pooled["event_ge15"], {"hits": 2, "n": 2, "est": 1.0})
        self.assertEqual(pooled["se15"]["hits"], 1)
        self.assertEqual(pooled["se15"]["n"], 2)
        self.assertEqual(pooled["se15"]["est"], 0.5)

        # A window that ends at 15 s is inside the 15 s budget. Delay uses the
        # PR2 completing-sample offset (window_end - 0.004 s).
        on_budget = np.zeros(8, dtype=bool)
        on_budget[2] = True
        on_score = score_episodes(on_budget, [(0.0, 40.0, "VT")])
        self.assertEqual(on_score["event_ge15"]["hits"], 1)
        self.assertEqual(on_score["se15"]["hits"], 1)

        # Duration under 15 s is outside both denominators. An early alarm
        # must not be renamed into Se15 by shrinking the old filter.
        short = np.zeros(8, dtype=bool)
        short[1] = True
        short_score = score_episodes(short, [(0.0, 12.0, "VF")])
        self.assertEqual(short_score["event_ge15"]["n"], 0)
        self.assertEqual(short_score["se15"]["n"], 0)
        self.assertIsNone(short_score["event_ge15"]["est"])
        self.assertIsNone(short_score["se15"]["est"])

    def test_one_trace_keeps_l0_l1_l2_l3_apart(self):
        from vf_layers import trace_record

        # Suspect only on windows 4 and 5 (20-30 s). Episode [0, 40).
        # One QRS at frame 0 arms the beat gate and leaves it open.
        scores = np.array([0.01, 0.01, 0.01, 0.01, 0.80, 0.80, 0.01, 0.01])
        rms = np.full(8, 0.40)
        traced = trace_record(
            scores,
            rms,
            beat_frames=np.array([0], dtype=np.int32),
            episodes=[(0.0, 40.0, "VF")],
            # No RS / FLAT / LEADOFF / AI. L3 must stay dark.
            rule_hits_by_sec={},
            ai_density_by_sec={},
        )
        np.testing.assert_array_equal(traced["L0_suspect"], [0, 0, 0, 0, 1, 1, 0, 0])
        np.testing.assert_array_equal(traced["L1_vfAlarm"], [0, 0, 0, 0, 0, 1, 0, 0])
        np.testing.assert_array_equal(traced["L2_vfHit"], [0, 0, 0, 0, 0, 1, 0, 0])
        self.assertTrue(np.all(traced["L3_user_alarm_sec"] == 0))
        self.assertEqual(traced["by_layer"]["L1"]["event_ge15"]["hits"], 1)
        self.assertEqual(traced["by_layer"]["L1"]["se15"]["hits"], 0)
        self.assertEqual(traced["by_layer"]["L2"]["event_ge15"]["hits"], 1)
        self.assertEqual(traced["by_layer"]["L2"]["se15"]["hits"], 0)
        self.assertEqual(traced["by_layer"]["L3"]["event_ge15"]["hits"], 0)
        self.assertEqual(traced["by_layer"]["L3"]["se15"]["hits"], 0)
        self.assertNotIn("se15", traced["by_layer"]["L0"])

    def test_constants_match_pr2_firmware_text(self):
        from vf_layers import (
            AI_DENSITY_TH,
            AI_SQI_MIN,
            ASRC_AI,
            ASRC_FLAT,
            ASRC_LEADOFF,
            ASRC_RS,
            ASRC_VF,
            COOLDOWN_S,
            GATE_MS,
            HOLD_S,
            REL_DENSITY_TH,
            REL_SQI_MIN,
            RMS_MIN,
            THETA,
        )

        header = HEADER.read_text(encoding="utf-8")
        main = MAIN.read_text(encoding="utf-8")
        self.assertIn("#define VF_THETA  0.15f", header)
        self.assertIn("#define ALARM_VF_RMS_MIN     0.05f", main)
        self.assertIn("#define ALARM_VF_GATE_MS     2500u", main)
        self.assertIn("#define ALARM_MIN_HOLD_S     12", main)
        self.assertIn("#define ALARM_AI_TH          8", main)
        self.assertIn("#define ALARM_AI_SQI_MIN     0.80f", main)
        self.assertIn("#define ALARM_REL_TH         1", main)
        self.assertIn("#define ALARM_REL_SQI_MIN    0.50f", main)
        self.assertIn("#define ALARM_COOLDOWN_S     25", main)
        self.assertEqual(THETA, 0.15)
        self.assertEqual(RMS_MIN, 0.05)
        self.assertEqual(GATE_MS, 2500)
        self.assertEqual(HOLD_S, 12)
        self.assertEqual(AI_DENSITY_TH, 8)
        self.assertEqual(AI_SQI_MIN, 0.80)
        self.assertEqual(REL_DENSITY_TH, 1)
        self.assertEqual(REL_SQI_MIN, 0.50)
        self.assertEqual(COOLDOWN_S, 25)
        self.assertEqual(ASRC_AI, 0x01)
        self.assertEqual(ASRC_RS, 0x02)
        self.assertEqual(ASRC_FLAT, 0x04)
        self.assertEqual(ASRC_VF, 0x08)
        self.assertEqual(ASRC_LEADOFF, 0x10)

    def test_ai_can_latch_and_cooldown_blocks_a_new_ai_latch(self):
        from vf_layers import ASRC_AI, ASRC_VF, UserAlarm

        alarm = UserAlarm()
        entered = alarm.tick(0, ai_density=8, sqi_min=0.80, vf_hit=False, beat_recent=True)
        self.assertEqual(entered["ble_alarm"], 1)
        self.assertEqual(entered["ble_asrc"], ASRC_AI)
        self.assertFalse(entered["vf_started_latch"])

        both = UserAlarm()
        joined = both.tick(0, ai_density=8, sqi_min=0.80, vf_hit=True, beat_recent=False)
        self.assertEqual(joined["ble_alarm"], 1)
        self.assertEqual(joined["ble_asrc"], ASRC_AI | ASRC_VF)
        self.assertFalse(joined["vf_started_latch"])
        self.assertFalse(joined["vf_telemetry_only"])

        cooled = UserAlarm()
        cooled.tick(0, ai_density=8, sqi_min=0.90, beat_recent=True)
        released = cooled.tick(12, ai_density=0, sqi_min=1.0, vf_hit=False, beat_recent=True)
        self.assertEqual(released["ble_alarm"], 0)
        blocked = cooled.tick(13, ai_density=8, sqi_min=0.90, vf_hit=True, beat_recent=True)
        self.assertEqual(blocked["ble_alarm"], 0)
        self.assertTrue(blocked["vf_telemetry_only"])
        rules = cooled.tick(14, rule_hits=0x02, vf_hit=False, beat_recent=True)
        self.assertEqual(rules["ble_alarm"], 1)
        self.assertEqual(rules["ble_asrc"], 0x02)

    def test_old_summary_has_event_ge15_and_no_se15_key(self):
        summary = json.loads((HERE / "results" / "summary.json").read_text(encoding="utf-8"))
        block = summary["profiles"]["firmware"]["vfdb"]
        self.assertIn("event_ge15", block)
        self.assertNotIn("se15", block)
        # The committed VFDB firmware event_ge15 figure stays a duration filter.
        # This test does not reinterpret it.
        self.assertEqual(block["event_ge15"]["hits"], block["event_ge15"]["se"]["k"])
        self.assertEqual(block["event_ge15"]["n"], block["event_ge15"]["se"]["n"])

    def test_published_v01_artifact_matches_builder(self):
        from vf_layers import build_v01_evidence

        disk = json.loads((HERE / "results" / "v01_score_contract.json").read_text(encoding="utf-8"))
        fresh = json.loads(json.dumps(build_v01_evidence()))
        self.assertEqual(disk, fresh)
        text = (HERE / "REPORT_V01_评分口径.md").read_text(encoding="utf-8")
        self.assertIn("event_ge15 不是 Se15", text)
        self.assertIn("2/2 = 1.000", text)
        self.assertIn("1/2 = 0.500", text)
        self.assertIn("L3", text)
        self.assertNotIn("se15", disk["stored_pr2_summary"]["event_ge15"])
        self.assertFalse(disk["stored_pr2_summary"]["has_se15_key"])
        self.assertFalse(disk["full_library_replay"])
        self.assertEqual(disk["heavy_slot_records"], 0)


if __name__ == "__main__":
    unittest.main()

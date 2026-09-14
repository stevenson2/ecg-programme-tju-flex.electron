# -*- coding: utf-8 -*-
"""summarize_r18_gate.py - aggregate R18 A1 seed evaluations (test-based decision)."""
import json
import statistics
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/devcontainers/meeti/runs/r18_gate")
seeds = [42, 43, 44]
rows = []
for s in seeds:
    p = OUT / f"r18_gate_eval_seed{s}.json"
    if p.exists():
        rows.append(json.loads(p.read_text(encoding="utf-8")))
summary = {"prereg": "runs/R18_A1_GATE_PREREG.md", "seeds": seeds, "runs": rows}
if rows:
    test_ea = [r["test_mit_incart"]["E_A"] for r in rows]
    test_sn = [r["test_mit_incart"]["Sn_A"] for r in rows]
    ptb_sn = [r["test_ptb"]["Sn_A"] for r in rows]
    summary["aggregate"] = {
        "n_runs": len(rows),
        "gate_passed_runs_test_EA_le_0.10": sum(1 for v in test_ea if v <= 0.10),
        "mit_test_E_A": {"mean": round(statistics.mean(test_ea), 4), "min": round(min(test_ea), 4), "max": round(max(test_ea), 4)},
        "mit_test_Sn_A": {"mean": round(statistics.mean(test_sn), 4), "min": round(min(test_sn), 4), "max": round(max(test_sn), 4)},
        "ptb_test_Sn_A": {"mean": round(statistics.mean(ptb_sn), 4), "min": round(min(ptb_sn), 4), "max": round(max(ptb_sn), 4)},
    }
    summary["decision"] = "RELAXED_PASS" if summary["aggregate"]["gate_passed_runs_test_EA_le_0.10"] >= 2 else "FAIL_TO_3CLASS"
else:
    summary["decision"] = "NO_RESULTS"
out = OUT / "r18_gate_summary.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary.get("aggregate", {}), ensure_ascii=False))
print("decision:", summary["decision"])
print("saved", out)

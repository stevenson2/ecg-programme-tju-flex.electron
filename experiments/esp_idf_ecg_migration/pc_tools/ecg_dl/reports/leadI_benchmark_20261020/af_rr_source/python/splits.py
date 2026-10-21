"""Patient-level split. Development = AFDB; test = LTAFDB + MIT-BIH. No leakage."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set

from constants import MIT_PATIENT_MAP


def patient_id(db: str, rec: str) -> str:
    rec = str(rec)
    db = str(db)
    if db in ("mitdb", "mit-bih", "mitbih"):
        pid = MIT_PATIENT_MAP.get(rec, rec)
        return f"mitdb/{pid}"
    if db in ("afdb", "mitdb_afdb"):
        return f"afdb/{rec}"
    if db in ("ltafdb",):
        return f"ltafdb/{rec}"
    if db in ("nstdb",):
        return f"nstdb/{rec}"
    return f"{db}/{rec}"


def assign_split(db: str) -> str:
    db = str(db)
    if db in ("afdb",):
        return "dev"
    if db in ("ltafdb",):
        return "test_ltaf"
    if db in ("mitdb", "mit-bih", "mitbih"):
        return "test_mit"
    if db in ("nstdb",):
        return "stress_nstdb"
    return "other"


def assert_no_leakage(records: Sequence[dict]) -> None:
    """records: dicts with keys db, rec, split."""
    by_split: Dict[str, Set[str]] = {}
    rec_to_pid: Dict[str, str] = {}
    for r in records:
        pid = patient_id(r["db"], r["rec"])
        key = f"{r['db']}/{r['rec']}"
        rec_to_pid[key] = pid
        by_split.setdefault(r["split"], set()).add(pid)
    # A patient must never appear in more than one split bucket.
    seen: Dict[str, str] = {}
    for split, pids in by_split.items():
        if split.startswith("stress"):
            continue
        for pid in pids:
            if pid in seen and seen[pid] != split:
                raise AssertionError(
                    f"patient leakage: {pid} in {seen[pid]} and {split}"
                )
            seen[pid] = split
    # Development patients must be AFDB-only.
    for pid in by_split.get("dev", ()):
        if not pid.startswith("afdb/"):
            raise AssertionError(f"dev split contains non-AFDB patient {pid}")
    for pid in by_split.get("test_ltaf", ()):
        if not pid.startswith("ltafdb/"):
            raise AssertionError(f"LTAF test split contains {pid}")
    for pid in by_split.get("test_mit", ()):
        if not pid.startswith("mitdb/"):
            raise AssertionError(f"MIT test split contains {pid}")


def split_table(db_records: Dict[str, Iterable[str]]) -> List[dict]:
    rows = []
    for db, recs in db_records.items():
        for rec in recs:
            rows.append(
                {
                    "db": db,
                    "rec": str(rec),
                    "patient": patient_id(db, rec),
                    "split": assign_split(db),
                }
            )
    assert_no_leakage(rows)
    return rows

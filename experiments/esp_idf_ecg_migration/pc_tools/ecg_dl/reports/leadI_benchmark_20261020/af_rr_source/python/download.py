#!/usr/bin/env python3
"""Parallel download from the PhysioNet open S3 bucket (physionet.org is ~KB/s here)."""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

DB_VERSION = {
    "mitdb": ("mitdb", "1.0.0"),
    "afdb": ("afdb", "1.0.0"),
    "ltafdb": ("ltafdb", "1.0.0"),
    "nstdb": ("nstdb", "1.0.0"),
}
EXTS = (".hea", ".dat", ".atr", ".qrs", ".xws", ".qrsc")
UA = "ecg-af-rr-source/1.0 (research; parallel fetch)"
FULL_MIN = {"mitdb": 40, "afdb": 20, "ltafdb": 50, "nstdb": 3}
S3 = "https://physionet-open.s3.amazonaws.com"
PN = "https://physionet.org/files"


def _n_dat(dest: Path) -> int:
    if not dest.is_dir():
        return 0
    return len(list(dest.rglob("*.dat")))


def has_record(dest: Path, rec: str) -> bool:
    return (dest / f"{rec}.dat").exists() and (dest / f"{rec}.hea").exists()


def urls_for(db: str, version: str, name: str) -> List[str]:
    return [
        f"{S3}/{db}/{version}/{name}",
        f"{PN}/{db}/{version}/{name}",
    ]


def http_open(url: str, timeout: int = 180):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_file(url_list: Sequence[str], dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 64:
        return f"skip {dest.name}"
    last_err = "no-url"
    for url in url_list:
        try:
            with http_open(url) as resp:
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as f:
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
                tmp.replace(dest)
            return f"ok {dest.name} {dest.stat().st_size}"
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code == 404:
                continue
        except Exception as e:
            last_err = str(e)
            continue
    if last_err == "HTTP 404" or "404" in last_err:
        return f"miss {dest.name}"
    return f"err {dest.name} {last_err}"


def list_remote_records(db: str, version: str) -> List[str]:
    last = None
    for url in urls_for(db, version, "RECORDS"):
        try:
            with http_open(url, timeout=60) as resp:
                text = resp.read().decode("utf-8", "replace")
            recs = []
            for line in text.splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                recs.append(s.split("/")[-1])
            if recs:
                return recs
        except Exception as e:
            last = e
            continue
    raise RuntimeError(f"cannot list {db} records: {last}")


def download_one(
    slug: str,
    dest: Path,
    records: Union[str, Sequence[str]] = "all",
    workers: int = 8,
) -> None:
    db, ver = DB_VERSION[slug]
    dest.mkdir(parents=True, exist_ok=True)
    recs = list_remote_records(db, ver) if records == "all" else list(records)
    print(f"[download] {slug} n_records={len(recs)} workers={workers} -> {dest}", flush=True)
    jobs: List[Tuple[List[str], Path]] = []
    for rec in recs:
        for ext in EXTS:
            name = f"{rec}{ext}"
            jobs.append((urls_for(db, ver, name), dest / name))
    jobs.append((urls_for(db, ver, "RECORDS"), dest / "RECORDS"))
    ok = miss = err = skip = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_file, u, p) for u, p in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            msg = fut.result()
            if msg.startswith("ok"):
                ok += 1
            elif msg.startswith("skip"):
                skip += 1
            elif msg.startswith("miss"):
                miss += 1
            else:
                err += 1
                print(f"  {msg}", flush=True)
            if i % 20 == 0 or i == len(futs):
                print(
                    f"  progress {i}/{len(futs)} ok={ok} skip={skip} miss={miss} err={err}",
                    flush=True,
                )
    print(f"[download] {slug} done, {_n_dat(dest)} .dat files", flush=True)


def download_dbs(
    data_dir: Path,
    dbs: str,
    record_map: Optional[Dict[str, List[str]]] = None,
    workers: int = 8,
) -> int:
    wanted = [x.strip() for x in dbs.split(",") if x.strip()]
    for slug in wanted:
        if slug not in DB_VERSION:
            print(f"unknown db {slug}", file=sys.stderr)
            return 2
        dest = data_dir / slug
        recs: Union[str, List[str]] = "all"
        if record_map and slug in record_map:
            recs = record_map[slug]
        if recs != "all":
            missing = [r for r in recs if not has_record(dest, r)]
            if not missing:
                print(f"[download] skip {slug}, requested records present", flush=True)
                continue
            recs = missing
        elif _n_dat(dest) >= FULL_MIN.get(slug, 1):
            print(f"[download] skip {slug}, {_n_dat(dest)} .dat already", flush=True)
            continue
        try:
            download_one(slug, dest, records=recs, workers=workers)
        except Exception as e:
            print(f"[download] {slug} failed: {e}", flush=True)
        if _n_dat(dest) == 0:
            print(f"[download] WARNING {slug} has no .dat yet", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--dbs", default="mitdb,afdb,ltafdb,nstdb")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    return download_dbs(args.data_dir, args.dbs, workers=args.workers)


if __name__ == "__main__":
    raise SystemExit(main())

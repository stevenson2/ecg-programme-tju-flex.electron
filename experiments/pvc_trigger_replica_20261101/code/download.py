"""Download a random patient subset of Icentia11k (open access, CC BY-NC-SA 4.0)."""
import json, random, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = 'https://physionet-open.s3.amazonaws.com/icentia11k-continuous-ecg/1.0'
OUT = Path('/tmp/icentia')
N_PATIENTS, SEGS, SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 140, 4, 20261031


def fetch(url, dest=None, tries=4):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if dest:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            return data
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f'{url}: {err}')


def patient_dir(pid):
    return f'p{pid // 1000:02d}/p{pid:05d}'


def main():
    rng = random.Random(SEED)
    pids = rng.sample(range(11000), N_PATIENTS)
    plan = []

    def choose(pid):
        recs = fetch(f'{BASE}/{patient_dir(pid)}/RECORDS').decode().split()
        rng_p = random.Random(SEED * 100003 + pid)
        return pid, sorted(rng_p.sample(recs, min(SEGS, len(recs))))

    with ThreadPoolExecutor(16) as ex:
        chosen = list(ex.map(choose, pids))
    jobs = []
    for pid, segs in chosen:
        for s in segs:
            for ext in ('.hea', '.atr', '.dat'):
                dest = OUT / patient_dir(pid) / (s + ext)
                if not dest.exists() or dest.stat().st_size == 0:
                    jobs.append((f'{BASE}/{patient_dir(pid)}/{s}{ext}', dest))
        plan.append({'patient': f'p{pid:05d}', 'dir': patient_dir(pid), 'segments': segs})
    print('files to fetch', len(jobs), flush=True)
    def safe(j):
        try:
            fetch(*j); return None
        except RuntimeError as e:
            return str(j[1])
    with ThreadPoolExecutor(24) as ex:
        failed = [f for f in ex.map(safe, jobs) if f]
    bad = {Path(f).with_suffix('').as_posix() for f in failed}
    print('failed files', failed, flush=True)
    for p in plan:
        p['segments'] = [s for s in p['segments'] if (OUT / p['dir'] / s).as_posix() not in bad]
    (OUT / 'plan.json').write_text(json.dumps({'seed': SEED, 'patients': plan}, indent=1))
    print('done', len(plan), 'patients', sum(len(p['segments']) for p in plan), 'segments')


if __name__ == '__main__':
    main()

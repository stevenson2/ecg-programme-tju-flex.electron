#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hardcoded-path audit for the meeti + electron repos (R17-CLEANUP P5).

Scans source files (*.py, *.sh, *.ps1, *.bat) in both repos and reports
violations of the "no hardcoded cross-shell paths" rule (AGENTS.md §1):

  mnt-c       : '/mnt/c/' hardcoded
  win-users   : 'C:\\Users' or 'C:/Users' hardcoded
  meeti-root  : '/home/devcontainers/meeti' hardcoded (use project_paths)
  legacy-data : meeti data dirs without the data/ prefix
                (processed_small / processed_smoke / selected8k / zenodo10k)
  legacy-refs : electron reference data dirs at the repo root
                (PTB-XL_ECG / ECG-Database / st-petersburg-incart / ECG-figs)

Files listed in scripts/check_paths.json (whitelist) are allowed to match
named rules — this records known legacy debt that later phases retire.
Directories that never need the rule (archives, caches, vendored data) are
skipped wholesale.

Usage (from either repo; both carry a copy of this file):
    python3 scripts/check_paths.py                 # scan both repos
    python3 scripts/check_paths.py --repo meeti    # scan one repo
    python3 scripts/check_paths.py --json          # machine-readable output

Exit code 0 = no unwhitelisted violation, 1 = violations found (CI style).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import all_paths  # noqa: E402

RULES = [
    ('legacy-refs', re.compile(r'PTB-XL_ECG|ECG-Database|st-petersburg-incart|ECG-figs')),
    ('legacy-data', re.compile(r'processed_small|processed_smoke|selected8k|zenodo10k')),
    ('mnt-c', re.compile(r'/mnt/c/', re.IGNORECASE)),
    ('win-users', re.compile(r'C:\\+Users|C:/Users', re.IGNORECASE)),
    ('meeti-root', re.compile(r'/home/devcontainers/meeti')),
]

SCAN_SUFFIXES = {'.py', '.sh', '.ps1', '.bat'}

# Noise directories never worth scanning (names, matched on any level).
SKIP_DIR_NAMES = {
    '.git', '__pycache__', 'venv', 'node_modules', 'papers_fulltext',
    'build', 'build_n16r8', 'build_supermini', 'dist', '.dart_tool', '.idea',
    # bulk data trees: no maintainable source inside, huge enumeration cost
    'processed_small', 'processed_smoke', 'selected8k', 'zenodo10k',
    'PTB-XL_ECG', 'ECG-Database', 'ECG-figs',
    'st-petersburg-incart-12-lead-arrhythmia-database-1.0.0',
}

# 'data' is only noise at a repo root (meeti/data after P2); nested data/
# directories such as pc_tools/ecg_dl/data hold maintained scripts.
ROOT_LEVEL_ONLY_SKIPS = {'data'}

# Directories whose files are exempt from every rule (archives keep their
# historical hardcoded paths; they are no longer maintained).
EXEMPT_DIR_PARTS = ('archive', 'legacy', 'docs')


def load_whitelist() -> dict:
    cfg = Path(__file__).resolve().parent / 'check_paths.json'
    if not cfg.exists():
        return {}
    return json.loads(cfg.read_text(encoding='utf-8'))


def repo_roots(which: str) -> list:
    paths = all_paths()
    win = paths['windows']
    if os.name == 'nt':
        roots = {'meeti': Path(win['meeti_root']), 'electron': Path(win['electron_root'])}
    else:
        roots = {k[:-len('_root')]: Path(v) for k, v in paths['wsl'].items()
                 if k.endswith('_root')}
    return [roots[k] for k in roots if which in ('both', k)]


def iter_files(root: Path):
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        parts = rel_dir.parts if str(rel_dir) != '.' else ()
        # prune noise + archives wholesale
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIR_NAMES
                       and not (d in ROOT_LEVEL_ONLY_SKIPS and not parts)
                       and not any(pre in d.lower() for pre in EXEMPT_DIR_PARTS)]
        if any(part in SKIP_DIR_NAMES for part in parts):
            continue
        exempt = any(any(pre in seg.lower() for pre in EXEMPT_DIR_PARTS)
                     for seg in parts)
        for name in filenames:
            if Path(name).suffix.lower() not in SCAN_SUFFIXES:
                continue
            rel = rel_dir / name
            if exempt:
                continue
            yield rel, Path(dirpath) / name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', choices=['both', 'meeti', 'electron'], default='both')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args()

    whitelist = load_whitelist()          # {"electron:pc_tools/x.py": ["rule", ...]}
    self_name = Path(__file__).name
    violations = []

    for root in repo_roots(args.repo):
        key_prefix = f'{root.name}:'
        for rel, path in iter_files(root):
            rel_posix = rel.as_posix()
            key = key_prefix + rel_posix
            # canonical path configs and this auditor are always exempt
            if rel_posix.endswith(('project_paths.json', self_name, 'cross_shell.py')):
                continue
            try:
                text = path.read_text(encoding='utf-8', errors='replace')
            except OSError:
                continue
            allowed = set(whitelist.get(key, []))
            for lineno, line in enumerate(text.splitlines(), 1):
                for rule, rx in RULES:
                    if rule in allowed:
                        continue
                    if rx.search(line):
                        violations.append(
                            {'repo': root.name, 'file': key, 'line': lineno,
                             'rule': rule, 'text': line.strip()[:160]})

    if args.as_json:
        print(json.dumps(violations, indent=2, ensure_ascii=False))
    elif violations:
        by_file = {}
        for v in violations:
            by_file.setdefault(v['file'], []).append(v)
        for f in sorted(by_file):
            for v in by_file[f][:5]:
                print(f"{v['file']}:{v['line']}: [{v['rule']}] {v['text']}")
            if len(by_file[f]) > 5:
                print(f"{f}: ... {len(by_file[f]) - 5} more")
        print(f'\n{len(violations)} violations in {len(by_file)} files '
              f'(whitelist: scripts/check_paths.json)')
    else:
        print('check_paths: 0 violations')
    sys.exit(0 if not violations else 1)


if __name__ == '__main__':
    main()

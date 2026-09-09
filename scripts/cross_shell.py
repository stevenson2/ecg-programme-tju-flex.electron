#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-shell task dispatcher (WSL bash <-> Windows PowerShell).

The project lives in two shells:
  * WSL bash  : training / evaluation / data prep / literature
  * PowerShell: firmware build / flash / serial / esptool / git push

This dispatcher resolves the right shell from the shared path config
(scripts/project_paths.json), injects the path variables, and runs the
command.  Use it from either side:

    python3 scripts/cross_shell.py auto --task flash --cmd "idf.py -B build_n16r8 build"
    python3 scripts/cross_shell.py wsl  --cmd "python3 scripts/r16_rr_shape_prep.py"
    python3 scripts/cross_shell.py win  --cmd "git push github main"

Long jobs must be started and polled in the same shell; do not hand a
long-running process from one shell to the other.
"""
import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import P, shell_for, all_paths  # noqa: E402


def _is_windows() -> bool:
    return os.name == 'nt' or platform.system().lower().startswith('windows')


def _win_path(key: str) -> str:
    return all_paths()['windows'][key]


def _wsl_path(key: str) -> str:
    return all_paths()['wsl'][key]


def build_command(shell: str, cmd: str) -> list:
    if shell == 'wsl':
        inner = f"cd '{_wsl_path('meeti_root')}' && source scripts/project_paths.sh && {cmd}"
        if _is_windows():
            return ['wsl.exe', '-e', 'bash', '-lc', inner]
        return ['bash', '-lc', inner]
    if shell == 'windows':
        el = _win_path('electron_root').replace("'", "''")
        inner = (f"Set-Location '{el}'; "
                 f". '{el}\\scripts\\project_paths.ps1'; {cmd}")
        return ['powershell.exe', '-NoProfile', '-Command', inner]
    raise ValueError(shell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('shell', choices=['wsl', 'win', 'auto'])
    ap.add_argument('--task', default=None,
                    help='routing keyword, e.g. flash / training / git-push-public')
    ap.add_argument('--cmd', required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    shell = a.shell
    if shell == 'auto':
        if not a.task:
            ap.error('--task required with --shell auto')
        shell = shell_for(a.task)
        print(f'[cross_shell] task={a.task!r} -> {shell}')
    shell = 'windows' if shell == 'win' else shell
    argv = build_command(shell, a.cmd)
    print('[cross_shell]', ' '.join(argv))
    if a.dry_run:
        return 0
    return subprocess.call(argv)


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single source of truth for project paths (WSL + Windows).

Usage:
    from project_paths import P, paths_for
    P['meeti_root']          # platform-appropriate path
    paths_for('windows')     # all Windows-form paths
    P['electron_root'] / 'experiments/esp_idf_ecg_migration'

Never hardcode cross-shell absolute paths in scripts; add the path here instead.
"""
import json
import os
import platform
from pathlib import Path

_JSON = Path(__file__).resolve().parent / 'project_paths.json'
_DATA = json.loads(_JSON.read_text(encoding='utf-8'))


def _is_wsl() -> bool:
    if 'WSL_DISTRO_NAME' in os.environ or 'WSL_INTEROP' in os.environ:
        return True
    try:
        return 'microsoft' in Path('/proc/version').read_text().lower()
    except Exception:
        return False


def _is_windows() -> bool:
    return os.name == 'nt' or platform.system().lower().startswith('windows')


def paths_for(flavor: str = None):
    """Return the dict of paths for 'wsl' or 'windows' (auto-detected)."""
    if flavor is None:
        flavor = 'windows' if _is_windows() else 'wsl'
    return dict(_DATA[flavor])


def all_paths():
    return {k: dict(v) for k, v in _DATA.items()
            if isinstance(v, dict) and k in ('wsl', 'windows')}


P = paths_for()
ROUTING = _DATA.get('task_routing', {})
GIT = _DATA.get('git', {})


def shell_for(task: str) -> str:
    """Return 'wsl' or 'windows' for a task keyword."""
    for shell, tasks in ROUTING.items():
        if shell in ('wsl', 'windows') and task in tasks:
            return shell
    raise KeyError(f'unknown task {task!r}; known: '
                   f'{ROUTING.get("wsl", []) + ROUTING.get("windows", [])}')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ('wsl', 'windows'):
        print(json.dumps(paths_for(sys.argv[1]), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(P, indent=2, ensure_ascii=False))

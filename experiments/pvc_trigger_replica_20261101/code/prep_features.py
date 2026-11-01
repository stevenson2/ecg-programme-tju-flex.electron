"""Build the per-split feature caches once, before parallel queues start (avoids a write race)."""
import time
from common import load
from features import trigger_features

for split in ('train', 'dev', 'test'):
    t = time.time()
    S, M = load(split)
    f = trigger_features(split, M, S)
    print(split, f['F'].shape, f'{time.time() - t:.0f}s', flush=True)

#!/usr/bin/env python3
"""Locate and verify recorded best sources before starting an experiment."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORGE_TASKS = {
    'muse_prefill_layer': 'muse_prefill_boundary',
    'muse_attention_decode': 'muse_attention_sliding',
    'muse_bf16_proj': 'muse_projection_o32',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forge', type=Path, help='optional ForgeKernel checkout to verify')
    args = parser.parse_args()
    best = json.loads((ROOT / 'research/best.json').read_text())
    errors = []
    for problem, record in best.items():
        expected = record['sha256']
        canonical = ROOT / 'kernels' / problem / 'submission.py'
        archived = ROOT / 'experiments' / record['experiment'] / 'submission.py'
        matches = [p for p in (archived, canonical) if digest(p) == expected]
        if not matches:
            errors.append(f'{problem}: no recorded-best source found')
        print(f"{problem}: {record['speedup']:.4f}x; best={matches[0].relative_to(ROOT) if matches else 'MISSING'}")
        if digest(canonical) != expected:
            print(f'  working source differs: {canonical.relative_to(ROOT)}')
        if args.forge and problem in FORGE_TASKS:
            ref = args.forge / 'tasks' / FORGE_TASKS[problem] / 'reference/submission.py'
            if digest(ref) != expected:
                errors.append(f'{problem}: Forge reference missing or differs: {ref}')
            else:
                print(f'  Forge reference matches: {ref}')
    for error in errors:
        print('ERROR:', error)
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())

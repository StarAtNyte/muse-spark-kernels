#!/usr/bin/env python3
"""Promote a fully validated improvement and optionally submit it for scoring."""
import argparse
import fcntl
import hashlib
import json
import shutil
import time
from pathlib import Path

from experiment import ARCHIVE, ROOT, cli, save

EXPECTED = {'muse_bf16_proj': (9, 30), 'muse_attention_decode': (16, 12),
            'muse_moe_w4a16': (5, 6), 'muse_decode_layer': (13, 12),
            'muse_prefill_layer': (5, 8)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    # Promotions update shared best/history files and must remain serial even
    # when callers launch separate CLI processes for different problems.
    lock = (ROOT / 'research/.promotion.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    source = ARCHIVE / args.experiment
    manifest = json.loads((source / 'manifest.json').read_text())
    status = json.loads((source / 'status.json').read_text())
    result = status.get('result') or {}
    problem = manifest['problem']
    if status.get('id', status.get('job_id')) != manifest['job_id']:
        parser.error('official result job ID does not match the candidate manifest')
    if not (status['status'] == 'done' and result.get('passed') is True
            and result.get('status') == 'ok' and result.get('checks', {}).get('ok')
            and not result.get('flags') and result.get('speedup', 0) > 1):
        parser.error('official result does not establish a passing speedup')
    for event, count in zip(('test', 'benchmark'), EXPECTED[problem]):
        records = [r for r in result['records'] if r['event'] == event]
        if len(records) != count or any(r.get('status') != 'pass' for r in records):
            parser.error(f'incomplete or failing {event} coverage')
    code = source / 'submission.py'
    if hashlib.sha256(code.read_bytes()).hexdigest() != manifest['sha256']:
        parser.error('archived source hash mismatch')
    best_path = ROOT / 'research/best.json'
    best = json.loads(best_path.read_text())
    previous = best.get(problem)
    if previous and previous['experiment'] != args.experiment and result['speedup'] <= previous['speedup']:
        parser.error('candidate does not improve the current validated best')
    public = previous.get('public_submission') if previous and previous['experiment'] == args.experiment else None
    existing = [json.loads(p.read_text()) for p in ARCHIVE.glob('*/manifest.json')]
    matching = [m for m in existing if m.get('mode') == 'run' and
                m.get('parent_experiment') == args.experiment]
    if public is None and matching:
        public = matching[-1].get('job_id')
    if args.submit:
        if matching:
            parser.error('scored launch already recorded; collect that job instead of resubmitting')
        limits = next(p['limits']['scored'] for p in cli('problems')['problems'] if p['name'] == problem)
        if limits['left'] < 1:
            parser.error('no scored runs remaining today')
        dest = ARCHIVE / f'scored_{args.experiment}'
        dest.mkdir(exist_ok=False)
        shutil.copyfile(code, dest / 'submission.py')
        scored = dict(manifest, name=dest.name, parent_experiment=args.experiment,
                      mode='run', status='prepared', decision='submitted_for_scoring')
        scored.pop('job_id', None)
        save(dest / 'manifest.json', scored)
        # Never retry an ambiguous POST; the prepared archive is retained.
        response = cli('submit', str(dest / 'submission.py'), '--no-wait')
        save(dest / 'launch.json', response)
        public = response['job_id']
        scored.update(job_id=public, status='received')
        save(dest / 'manifest.json', scored)
        history_path = ROOT / 'research/submissions.json'
        history = json.loads(history_path.read_text())
        history.append(dict(problem=problem, job_id=public, experiment=args.experiment,
                            sha256=manifest['sha256'], submitted_at=time.time(), status='received'))
        save(history_path, history)
    shutil.copyfile(code, ROOT / 'kernels' / problem / 'submission.py')
    manifest.update(decision='promote_globally', decision_reason='Full official coverage passed; current best validated aggregate')
    save(source / 'manifest.json', manifest)
    best[problem] = dict(experiment=args.experiment, job_id=manifest['job_id'],
                         sha256=manifest['sha256'], speedup=result['speedup'],
                         speedup_ci=result['speedup_ci'], tests=EXPECTED[problem][0],
                         timed_cases=EXPECTED[problem][1], public_submission=public)
    save(best_path, best)
    print(json.dumps(best[problem], indent=2))


if __name__ == '__main__':
    main()

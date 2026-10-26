#!/usr/bin/env python3
"""Bounded official-job watcher. --submit promotes clear validated improvements.

Only the existing promote.py gate may publish code. Ambiguous POST attempts
are never retried. Poll failures preserve the same live job for later polls.
"""
import argparse
import fcntl
import json
import subprocess
import sys
import time

from experiment import ARCHIVE, ROOT, TERMINAL, cli, save


def event(kind, **fields):
    entry = dict(time=time.time(), event=kind, **fields)
    line = json.dumps(entry)
    print(line, flush=True)
    with (ROOT / 'research/watch-events.jsonl').open('a') as file:
        file.write(line + '\n')


def poll():
    pending = 0
    for path in sorted(ARCHIVE.glob('*/manifest.json')):
        manifest = json.loads(path.read_text())
        if manifest.get('status') in TERMINAL or not manifest.get('job_id'):
            continue
        try:
            status = cli('status', manifest['job_id'])
        except (RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
            event('poll_error', name=manifest['name'], job=manifest['job_id'], error=str(exc)[:300])
            pending += 1
            continue
        save(path.parent / 'status.json', status)
        previous = manifest.get('status')
        manifest.update(status=status.get('status', 'unknown'), updated_at=time.time())
        save(path, manifest)
        if manifest['status'] not in TERMINAL:
            pending += 1
        if manifest['status'] != previous:
            result = status.get('result') or {}
            event('status', name=manifest['name'], job=manifest['job_id'], status=manifest['status'],
                  passed=result.get('passed'), speedup=result.get('speedup'), ci=result.get('speedup_ci'))
            failure = next((r for r in result.get('records', []) if
                            r.get('event') in ('test', 'benchmark') and r.get('status') != 'pass'), None)
            if failure:
                event('failure', name=manifest['name'], case=failure.get('case'),
                      message=str(failure.get('message', failure.get('error')))[:1600])
    return pending


def candidates():
    best = json.loads((ROOT / 'research/best.json').read_text())
    manifests = [json.loads(p.read_text()) for p in ARCHIVE.glob('*/manifest.json')]
    scored = {m.get('parent_experiment') for m in manifests if m.get('mode') == 'run'}
    chosen = {}
    for manifest in manifests:
        name, problem = manifest['name'], manifest['problem']
        if manifest.get('mode') == 'run' or manifest.get('status') != 'done' or name in scored:
            continue
        result = json.loads((ARCHIVE / name / 'status.json').read_text()).get('result') or {}
        ci = result.get('speedup_ci')
        if result.get('passed') is not True or not ci:
            continue
        previous = best.get(problem)
        if previous and previous['experiment'] == name:
            continue
        # Prefer a clear margin to spending scored runs on overlapping noise.
        threshold = previous['speedup_ci'][1] if previous else 1.
        if ci[0] <= threshold:
            continue
        if problem not in chosen or result['speedup'] > chosen[problem][1]:
            chosen[problem] = (name, result['speedup'])
    return chosen


def sync_public():
    with (ROOT / 'research/.promotion.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        best_path, history_path = ROOT / 'research/best.json', ROOT / 'research/submissions.json'
        best, history = json.loads(best_path.read_text()), json.loads(history_path.read_text())
        for path in ARCHIVE.glob('scored_*/manifest.json'):
            manifest = json.loads(path.read_text())
            status_path = path.parent / 'status.json'
            if not status_path.exists():
                continue
            status = json.loads(status_path.read_text())
            result = status.get('result') or {}
            for row in history:
                if row['job_id'] == manifest.get('job_id'):
                    row.update(status=status['status'], passed=result.get('passed'), speedup=result.get('speedup'))
            entry = best.get(manifest['problem'], {})
            if entry.get('public_submission') == manifest.get('job_id'):
                entry['public_status'] = status['status']
                if result.get('passed'):
                    entry.update(public_speedup=result['speedup'], public_speedup_ci=result['speedup_ci'])
        save(best_path, best)
        save(history_path, history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=150)
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if args.seconds < 0 or args.interval < 1:
        parser.error('seconds must be nonnegative and interval positive')
    lock = (ROOT / 'research/.watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic() + args.seconds
    attempted = set()
    while True:
        pending = poll()
        for problem, (name, speedup) in candidates().items():
            if name in attempted:
                continue
            attempted.add(name)
            event('candidate', problem=problem, name=name, speedup=speedup, submit=args.submit)
            if args.submit:
                command = [sys.executable, str(ROOT / 'tools/promote.py'), name, '--submit']
                proc = subprocess.run(command, capture_output=True, text=True)
                event('promotion', name=name, returncode=proc.returncode,
                      output=proc.stdout.strip(), error=proc.stderr.strip()[:1000])
        sync_public()
        event('round', pending=pending)
        remaining = deadline - time.monotonic()
        if pending == 0 or remaining <= 0:
            break
        time.sleep(min(args.interval, remaining))
    event('watch_finished', pending=pending)


if __name__ == '__main__':
    main()

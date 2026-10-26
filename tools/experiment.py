#!/usr/bin/env python3
"""Immutable candidate archival and bounded official private-check management.

Uses the authenticated okc CLI without reading or storing its credentials.
Never submits to the public leaderboard or promotes a candidate automatically.
"""
import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'experiments'
CLI = shutil.which('okc') or str(Path.home() / '.okc/bin/okc')
TERMINAL = {'done', 'failed', 'rejected', 'duplicate'}


def cli(*args):
    proc = subprocess.run([CLI, *args, '--json'], capture_output=True, text=True,
                          timeout=60)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return json.loads(proc.stdout)


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def launch(args):
    if not args.name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('name must contain only letters, numbers, _ and -')
    source = Path(args.file).resolve()
    code = source.read_text()
    ast.parse(code)
    marker = code.splitlines()[0]
    if not marker.startswith('# okc: problem='):
        raise ValueError('missing first-line problem marker')
    problem = marker.split('=', 1)[1].strip()
    dest = ARCHIVE / args.name
    if (dest / 'manifest.json').exists():
        raise ValueError('experiment already exists; collect it or use a new name')
    problems = cli('problems')['problems']
    limits = next(p['limits'] for p in problems if p['name'] == problem)
    if limits['checks']['left'] <= args.reserve:
        raise RuntimeError('check budget would reach the reserved quota')
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / 'submission.py'
    if target.exists() and target.read_text() != code:
        raise ValueError('existing archived source differs')
    if source != target:
        shutil.copyfile(source, target)
    manifest = dict(name=args.name, problem=problem, hypothesis=args.hypothesis,
                    sha256=hashlib.sha256(code.encode()).hexdigest(),
                    created_at=time.time(), decision='archive_only',
                    limits_before=limits, status='prepared')
    save(dest / 'manifest.json', manifest)
    # Persist the job response immediately. Do not retry POST after ambiguous failure.
    reply = cli('check', str(target), '--no-wait')
    save(dest / 'launch.json', reply)
    manifest.update(job_id=reply['job_id'], status='received')
    save(dest / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


def collect(args):
    paths = sorted(ARCHIVE.glob('*/manifest.json'))
    if args.name:
        paths = [p for p in paths if p.parent.name == args.name]
        if not paths:
            raise ValueError('unknown experiment')
    for path in paths:
        manifest = json.loads(path.read_text())
        if args.pending and manifest.get('status') in TERMINAL:
            continue
        if not manifest.get('job_id'):
            print(f"{manifest['name']}: no recorded job; inspect launch.json and okc jobs")
            continue
        result = cli('status', manifest['job_id'])
        save(path.parent / 'status.json', result)
        manifest.update(status=result.get('status', 'unknown'), updated_at=time.time())
        save(path, manifest)
        print(f"{manifest['name']}: {manifest['job_id']} {manifest['status']}")
        if manifest['status'] in TERMINAL:
            details = result.get('result') or {}
            print(json.dumps({k: details.get(k) for k in
                              ('status', 'passed', 'returncode', 'error', 'speedup', 'speedup_ci', 'score_us')},
                             indent=2))
            for record in details.get('records', []):
                if record.get('event') in ('test', 'benchmark') and record.get('status') != 'pass':
                    print('First failed case:', record.get('case'))
                    print(str(record.get('message', record.get('error', '')))[:2000])
                    break
            print(f'Full evidence: {path.parent / "status.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('check')
    run.add_argument('name')
    run.add_argument('file')
    run.add_argument('--hypothesis', required=True)
    run.add_argument('--reserve', type=int, default=5)
    poll = commands.add_parser('collect')
    poll.add_argument('name', nargs='?')
    poll.add_argument('--pending', action='store_true', help='poll only nonterminal recorded jobs')
    args = parser.parse_args()
    try:
        (launch if args.command == 'check' else collect)(args)
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f'error: {exc}\n')


if __name__ == '__main__':
    main()

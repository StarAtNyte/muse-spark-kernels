#!/usr/bin/env python3
"""Compile our attention Triton kernels for SM90 without torch or an H200.

This catches code-generation failures only. It does not prove correctness,
memory safety, runtime resource limits, or performance. Remote checks do that.
Run with .venv/bin/python (Triton 3.6.0).
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.language.extra.cuda import libdevice


def load_jit_functions(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and any(isinstance(d, ast.Attribute) and d.attr == 'jit'
                         for d in node.decorator_list)]
    namespace = {'triton': triton, 'tl': tl, 'libdevice': libdevice,
                 '__name__': 'preflight_attention', '__file__': str(path)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    path = args.candidate.resolve()
    funcs = load_jit_functions(path)
    target = GPUTarget('cuda', 90, 32)
    results = []

    def check(name, signature, constants):
        compiled = triton.compile(ASTSource(funcs[name], signature, constexprs=constants),
                                  target=target, options={'num_warps': 4, 'num_stages': 2})
        metadata = compiled.metadata._asdict()
        results.append(dict(kernel=name, constants=constants, metadata=metadata))
        print(f'{name}: compiled SM90; shared={metadata.get("shared")} bytes', flush=True)

    for theta, scaled in [(0., False), (500000., True)]:
        check('_prepare_qkv',
              dict(X='*bf16', POS='*i64', LOC='*i64', KS='*fp32', VS='*fp32',
                   Q='*fp8e4nv', KC='*fp8e4nv', VC='*fp8e4nv'),
              dict(HQ=16, HKV=2, D=64, STRIDE=2304, THETA=theta,
                   SCALE=scaled, HEADS=4))
    if '_sliding_partials' in funcs:
        for length, window, page, scaled, splits in [
            (1, 2048, 64, False, 16), (8, 2048, 1, True, 4),
            (1, 0, 1, False, 128), (8, 0, 64, True, 16),
        ]:
            bm, bn = max(16, triton.next_power_of_2(length * 8)), 64 if window else 128
            span = window + length - 1 if window else 100000
            chunk = triton.cdiv(triton.cdiv(span, splits), bn) * bn
            check('_sliding_partials',
                  dict(Q='*fp8e4nv', KC='*fp8e4nv', VC='*fp8e4nv', TABLE='*i32',
                       SEQLEN='*i32', KS='*fp32', VS='*fp32', PART='*fp32', LSE='*fp32'),
                  dict(HQ=16, HKV=2, D=64, LENGTH=length, PAGE=page, TABLE_N=100000,
                       WINDOW=window, SPLITS=splits, CHUNK=chunk, BM=bm, BN=bn,
                       SM_SCALE=5.473/8, SCALED=scaled))
            check('_merge_norm_gate',
                  dict(PART='*fp32', LSE='*fp32', X='*bf16', OUT='*bf16'),
                  dict(HQ=16, HKV=2, D=64, LENGTH=length, STRIDE=2304,
                       SPLITS=splits, BM=bm))
    report = dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  triton=triton.__version__, target='SM90', compilations=results,
                  validated='compilation only; no GPU execution')
    out = args.output or path.parent / 'compile-sm90.json'
    out.write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(f'{len(results)} compilations passed; {out}')


if __name__ == '__main__':
    main()

import sys
base = open("../moe_wm/base_v5.py").read()
base = base.replace("def prepare(weights):", "def _prepare_marlin(weights):", 1)
base = base.replace("def kernel_fn(data, state):", "def _marlin_kernel_fn(data, state):", 1)
SRV = "../../serving/sglang/python/sglang/srt/layers/"
tpl = open(__import__("os").environ.get("LABTPL","lab_conv_template.py")).read().replace("__WMX__", repr(open(SRV + "muse_spark_wm_experts.py").read()))
tail = '''

import time


def prepare(weights):
    state = _prepare_marlin(weights)
    try:
        (_conv_lab if '_conv_lab' in globals() else _dbg_lab)(state, weights)
    except Exception:
        import traceback
        print("LABV FAILED", traceback.format_exc(), flush=True)
    return state


def kernel_fn(data, state):
    return _marlin_kernel_fn(data, state)
'''
open(sys.argv[1], "w").write(base + tpl + tail)

import sys
sys.path.insert(0, "../moe_wm")
base = open("../moe_wm/base_v5.py").read()
base = base.replace("def prepare(weights):", "def _prepare_marlin(weights):", 1)
base = base.replace("def kernel_fn(data, state):", "def _marlin_kernel_fn(data, state):", 1)
SRV = "../../serving/sglang/python/sglang/srt/layers/"
tpl = open("lab_layer_template.py").read()
tpl = tpl.replace("r'''__WMX__'''", repr(open(SRV + "muse_spark_wm_experts.py").read()))
tpl = tpl.replace("r'''__QMOE__'''", repr(open(SRV + "quantization/muse_spark_nvfp4_moe.py").read()))
tpl = tpl.replace("r'''__EXPERTS__'''", repr(open(SRV + "muse_spark_fp4_moe.py").read()))
tail = '''

def prepare(weights):
    state = _prepare_marlin(weights)
    try:
        _srv_lab(state, weights)
    except Exception:
        import traceback
        print("LABS FAILED", traceback.format_exc(), flush=True)
    return state


def kernel_fn(data, state):
    return _marlin_kernel_fn(data, state)
'''
open(sys.argv[1], "w").write(base + tpl + tail)
print("wrote")

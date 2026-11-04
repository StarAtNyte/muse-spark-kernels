

# proj_big dispatch: (layer, m) -> (mode, stages, bft, split); every other cell keeps the parent's route
PB_ROUTES = @@ROUTES@@
_pb_prev_prepare = prepare


def prepare(weights):
    st = _pb_prev_prepare(weights)
    try:
        st["pb"] = pb_prepare(weights)
    except Exception as exc:  # noqa: BLE001  (any failure keeps the parent's routes)
        st["pb"] = {"cells": {}, "log": ["pb prepare failed %r" % (exc,)]}
    print("\n".join(st["pb"].get("log", [])), flush=True)
    return st


_pb_prev_kernel = kernel_fn


def kernel_fn(data, state):
    cfg = data["config"]
    cell = state.get("pb", {}).get("cells", {}).get((cfg["layer"], cfg["m"]))
    x = data["x"]
    if cell is not None and cfg["n"] == cell["n"] and x.is_contiguous() and x.shape[1] == cell["k"]:
        _ctx_bind(x)
        return pb_run(x, cell)
    return _pb_prev_kernel(data, state)

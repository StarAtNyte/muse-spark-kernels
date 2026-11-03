# okc: problem=muse_bf16_proj
"""Candidate projection hybrid v1:
Based on canonical scored 1.5395x source (experiments/scored_projection_vjeux_hybrid_v1/submission.py).
Extends vjeux routing to:
- (32, "pooler"): already in base
- (128, "router"): already in base
- (256, "router"): replaces dual torch.mm launches with fused rk_kernel (TMA+WGMMA)
- (32, "qkv_gate"): replaces fallback with balanced split-K sk2 (qkv32_r208)
- (32, "unpooler"): replaces fallback with split-K sk2 (unpooler32_r144)
- (1, "unpooler"): replaces fallback with 12-bit compressed cgemv_kernel
- (1, "router"): replaces fallback with FP32 vector rgemv_kernel
"""
import sys
from pathlib import Path

# Load canonical base
BASE_PATH = Path('/home/nitiz/Projects/Self/Challenges/muse-spark/experiments/scored_projection_vjeux_hybrid_v1/submission.py')
if not BASE_PATH.exists():
    BASE_PATH = Path(__file__).resolve().parents[5] / 'experiments/scored_projection_vjeux_hybrid_v1/submission.py'

base_code = BASE_PATH.read_text()
# We want to replace the dispatch condition at the bottom of the file
target = 'if (cfg["m"], cfg["layer"]) in ((32, "pooler"), (128, "router")):'
replacement = 'if (cfg["m"], cfg["layer"]) in ((32, "pooler"), (128, "router"), (256, "router"), (32, "qkv_gate"), (32, "unpooler"), (1, "unpooler"), (1, "router")):'

if target not in base_code:
    raise ValueError(f"Target dispatch pattern not found in {BASE_PATH}")

candidate_code = base_code.replace(target, replacement, 1)

# Execute candidate_code in this module's namespace
exec(compile(candidate_code, "<candidate_v1>", "exec"))

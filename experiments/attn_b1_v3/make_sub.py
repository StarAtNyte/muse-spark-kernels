"""submission.py = parent (byte-identical up to its kernel_fn) + the batch-1 module (b1_part.py)."""
import pathlib
H = pathlib.Path(__file__).resolve().parent
ROOT = H.parent.parent
par = (ROOT / 'experiments/attention_readable_fp32_path_v1/submission.py').read_text()
cut = par.index('def kernel_fn(data: Inputs) -> torch.Tensor:')
(H / 'parent.py').write_text(par)
(H / 'submission.py').write_text(par[:cut].rstrip('\n') + '\n' + (H / 'b1_part.py').read_text())
print('ok', len(par[:cut]))

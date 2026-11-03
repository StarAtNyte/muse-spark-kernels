import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x + 1.0, mask=mask)

if __name__ == '__main__':
    x = torch.zeros(64, device='cuda', dtype=torch.float32)
    y = torch.zeros(64, device='cuda', dtype=torch.float32)
    add_kernel[(1,)](x, y, 64, BLOCK_SIZE=64)
    print('Triton SUCCESS! sum =', y.sum().item())

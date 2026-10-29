"""CPU check: _sk1_pack_down is an exact permutation (decode words via _POS back to codes; scales by index)."""
import importlib.util, sys, torch
from unittest.mock import MagicMock
sys.modules.setdefault('torch.utils.cpp_extension', MagicMock())
spec = importlib.util.spec_from_file_location('sub', sys.argv[1]); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
g = torch.Generator().manual_seed(0)
codes = torch.randint(0, 256, (4096, 2048), generator=g, dtype=torch.int32).to(torch.uint8)
sf = torch.randint(0, 256, (4096, 256), generator=g, dtype=torch.int32).to(torch.uint8)
out = S._sk1_pack_down(codes, sf)
nchunk = 256
blk = out.view(nchunk, 36864)
words = blk[:, :32768].contiguous().view(torch.int32).view(nchunk, 256, 8, 4).to(torch.int64) & 0xFFFFFFFF  # c, rb, g, t
nib = torch.stack((codes & 0xF, codes >> 4), -1).reshape(4096, 4096).to(torch.int64)
ok = True
for Rg, (h, j0) in enumerate(S._REG_HJ):
    for up in range(2):
        val = torch.zeros_like(words)
        for b in range(4):
            val |= ((words >> S._POS[(Rg, up, b)]) & 1) << b
        rows = (torch.arange(256)[:, None, None] * 16 + h * 8 + torch.arange(8)[None, :, None])  # rb, g, 1
        ks = torch.arange(4)[None, None, :] * 4 + j0 + up  # t
        for c in (0, 1, 77, 255):
            ref = nib[rows.expand(256, 8, 4), (c * 16 + ks).expand(256, 8, 4)]
            ok &= bool((val[c] == ref).all())
sc = blk[:, 32768:].contiguous().view(nchunk, 256, 8, 2)  # c, rb, g, h
for c in (0, 5, 255):
    ref = torch.stack([sf[torch.arange(256)[:, None] * 16 + h * 8 + torch.arange(8)[None, :], c] for h in range(2)], -1)
    ok &= bool((sc[c] == ref).all())
print('PACK_OK' if ok else 'PACK_FAIL')

# Exact public-reference BF16 exponent statistics

H200 public-image app `ap-OAs4mBzjdeJmmYOvO928A1` called `kernels/muse_bf16_proj/reference.py:generate_weights` at its documented seed 5151 and held-out seed 7919. It counted BF16 exponent bytes in the four dense tensors, selected the 16 most frequent per tensor, and counted contiguous 64-value row segments with at least one exponent outside that palette. Return code 0; exact counts, palettes, shapes and seeds are in `screen.log`.

| Tensor | Exponent entropy, bits | Values outside top 16 | 64-value segments with an exception |
|---|---:|---:|---:|
| QKV | 2.5435–2.5436 | 0.0071–0.0072% | 0.456–0.457% |
| Output | 2.5443–2.5451 | 0.0099–0.0105% | 0.630–0.671% |
| Pooler | 2.5432–2.5435 | 0.0069–0.0071% | 0.438–0.451% |
| Unpooler | 2.5449–2.5451 | 0.0098–0.0100% | 0.628–0.635% |

These are statistics on generated weights, not a full codec output: they omit metadata, padding, palette storage, Huffman bitstream and exception representation. The contiguous-64 segmentation is a model of row segments and has not been checked against Unweight's physical layout or kernel's verbatim decisions. Nothing was encoded or reconstructed, and no speed, preparation time or official score was measured. The result supports testing lossless palette reconstruction, but our scored implementation already has compressed routes and must be the direct baseline. Unusual values, held-out inputs, actual implementation packing, exact round trips and full judge validation remain necessary.

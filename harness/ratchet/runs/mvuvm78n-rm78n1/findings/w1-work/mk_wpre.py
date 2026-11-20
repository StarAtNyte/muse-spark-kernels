"""cand_wpre.py: OKC_WPRE=1 => WGS weight producer starts filling the weight ring as soon as the x producer has the
stage ranges (named barrier 8), instead of after this CTA's x gather (WS_JOIN)."""
import sys
src = open(sys.argv[1]).read()
def rep(old, new, cnt=1):
    global src
    n = src.count(old); assert n == cnt, (n, old[:80]); src = src.replace(old, new)
rep("""__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready""",
"""#ifndef OKC_WPRE
#define OKC_WPRE 0
#endif
#if OKC_WPRE
#define WGS_JOIN() asm volatile("bar.sync 2, %0;\\n" ::"n"(NTL - 32) : "memory")  // consumers + x producer
#else
#define WGS_JOIN() WS_JOIN()
#endif
__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
#if OKC_WPRE
  asm volatile("bar.sync 8, 64;\\n" ::: "memory");  // the x producer has the stage ranges: fill the ring during the gather
#else
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
#endif""")
rep("""  if ((threadIdx.x & 31) == 0) wgs_ranges(p, S);
  __syncwarp();
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready""",
"""  if ((threadIdx.x & 31) == 0) wgs_ranges(p, S);
  __syncwarp();
#if OKC_WPRE
  __threadfence_block();
  asm volatile("bar.arrive 8, 64;\\n" ::: "memory");
#endif
  WGS_JOIN();  // routing tables and this CTA's stage ranges are ready""")
rep("""  wgs_gather(p, S);
  DBG(7);
  WS_JOIN();
  DBG(6);""", """  wgs_gather(p, S);
  DBG(7);
  WGS_JOIN();
  DBG(6);""")
src = src.replace('name=f"okc_moe_w4a16_v216x_', 'name=f"okc_moe_w4a16_v216wp_')
open(sys.argv[2], 'w').write(src); print('ok')

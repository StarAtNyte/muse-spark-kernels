"""Stamp probe of the merge/ticket chain (diagnostic only, never submitted). Env B1_PROBE at build:
0 = stamps only, 1 = no workspace reds (wrong output), 3 = fence.acq_rel.gpu then a relaxed ticket atomic."""
import pathlib, sys
H = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(H.parents[1] / 'experiments/attention_b1lat_v1'))
import make_variants as MV
from make_cand import rep
s = MV.stamps((H / 'parent.py').read_text())
A = 'asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"('
for tgt in ('c', 'p.ws_cnt + bp'):
    old = A + tgt + '), "r"(mine) : "memory");'
    new = ('STAMP(9);\n#if PROBE == 3\n asm volatile("fence.acq_rel.gpu;" ::: "memory"); STAMP(12);\n'
           ' asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(' + tgt + '), "r"(mine) : "memory");\n#else\n'
           + old + '\n#endif\n STAMP(10);')
    s = rep(s, old, new)
s = rep(s, 'if (do_red) {   // (one branch', 'if (do_red && PROBE != 1) {   // (one branch')
s = rep(s, '''          __nanosleep(64);
        }
      }
    } else if (tid == 0) {''', '''          __nanosleep(64);
        }
      }
      STAMP(11);
    } else if (tid == 0) {''')
s = rep(s, '#define SPIN_CHECK', '#ifndef PROBE\n#define PROBE 0\n#endif\n#define SPIN_CHECK')
s = s.replace('name=f"okc_attn_v298', 'name=f"okc_attn_b1pr{_PROBE}')
s = rep(s, '"--expt-relaxed-constexpr", *flags],', '"--expt-relaxed-constexpr", f"-DPROBE={_PROBE}", *flags],')
s = rep(s, '_MOD = None\n', '_MOD = None\nimport os as _os\n_PROBE = int(_os.environ.get("B1_PROBE", "0"))\n')
(H / 'probe.py').write_text(s); print('ok')

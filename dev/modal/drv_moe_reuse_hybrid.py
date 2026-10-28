"""Same independent checks and paired timings, with preserved small-expert tiles."""
code=open('/w/modal/drv_moe_reuse64.py').read()
a=code.index('for name,cuda in ');b=code.index('\nsub._wm_module=',a)
code=code[:a]+'''for name,cuda in [('base',bundle.CUDA_BASE),('hybrid',bundle.CUDA_HYBRID)]:
 mods[name]=build('muse_reuse_hybrid_'+name,bundle.CPP,cuda,functions,flags);mods[name].wm_set_pdl_t(1);print('COMPILED',name,flush=True)
'''+code[b:]
code=code.replace("for name in ('control','stream','paired'):","for name in ('hybrid',):")
code=code.replace('MOE_REUSE64_SCREEN_PASSED','MOE_REUSE_HYBRID_PASSED')
exec(code)

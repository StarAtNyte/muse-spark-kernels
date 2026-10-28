import torch,sys,importlib.util
sys.path.insert(0,'/w/modal');from compile_split import build
sp=importlib.util.spec_from_file_location('probe','/w/run/sub.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
mod=build('muse_qreg_mapping_probe',m.CPP,m.CUDA,['qreg_probe'],['-O3','-gencode=arch=compute_90a,code=sm_90a'])
for kind in ('identity','permutation','integer','random'):
 torch.manual_seed(179)
 a=torch.eye(64,device='cuda').repeat(3,1,1) if kind=='identity' else torch.randint(-3,4,(3,64,64),device='cuda').float()
 if kind=='permutation':a=torch.eye(64,device='cuda')[:,torch.arange(64,device='cuda')*7%64].repeat(3,1,1)
 b=torch.randint(-3,4,(3,64,64),device='cuda').float()
 if kind=='random':a=torch.randn_like(a);b=torch.randn_like(b)
 a=a.to(torch.float8_e4m3fn);b=b.to(torch.float8_e4m3fn);ref=a.float()@b.float().transpose(1,2)
 for mode in (0,1,2):
  out=mod.qreg_probe(a,b,mode);error=((out-ref).norm()/ref.norm()).item();maximum=(out-ref).abs().max().item()
  print('CHECK',kind,mode,error,maximum,flush=True)
  if kind!='random':assert maximum==0.,(kind,mode,maximum)
  else:assert error<.001,(kind,mode,error)
print('QREG_MAPPING_PROBE_PASSED',flush=True)

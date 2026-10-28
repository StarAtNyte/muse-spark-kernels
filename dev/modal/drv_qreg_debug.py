exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
import triton
sys.path.insert(0,'/w/modal');from compile_split import build
mod=build('muse_qreg_integration_debug',candidate._WG_CPP,candidate._WG_CUDA,['wg5_run_t'],['-O3','-gencode=arch=compute_90a,code=sm_90a']);candidate._WG_MOD=mod
for ctx in (32,256,6000):
 data=R.generate_input(8,ctx,'tp8','full',911,q_len=8,page=64,spread='fixed',dist='coherent')
 qkv,q=base._fd_prepare(data);torch.cuda.synchronize()
 out=candidate.kernel_fn(data);ok,msg=R.check_implementation(data,R.collect(data,out));print('ATTENTION',ctx,ok,msg,flush=True)
 if ctx>=256:
  part=torch.empty((8,2,1,64,64),device='cuda');lse=torch.empty((8,2,1,64),device='cuda')
  chunk=triton.cdiv(data['page_table'].shape[1]*64,64)*64
  mod.wg5_run_t(q,data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['k_scale'],data['v_scale'],part,lse,8,16,64,0,1,chunk,1.,0,1024)
  qa=q.float().view(8,8,2,8,64).permute(0,2,1,3,4).reshape(8,2,64,64)
  slots=data['page_table'][:,0].long()[:,None]*64+torch.arange(64,device='cuda')[None,:]
  ka=data['k_cache'].view(-1,2,64)[slots].float().permute(0,2,1,3)
  expected=qa@ka.transpose(-1,-2);actual=part[:,:,0]
  err=((actual-expected).norm()/expected.norm()).item();print('SCORE',ctx,err,(actual-expected).abs().max().item(),flush=True)
print('QREG_DEBUG_FINISHED',flush=True)

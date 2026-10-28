"""cuBLASLt BF16 algorithm probe; plans contain shape metadata only."""
CPP = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <memory>
#include <vector>
#define CK(s) do { auto status_ = (s); TORCH_CHECK(status_ == CUBLAS_STATUS_SUCCESS, "cuBLASLt status ", int(status_)); } while(0)
struct Plan {
  cublasLtMatmulDesc_t op=nullptr;
  cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
  std::vector<cublasLtMatmulAlgo_t> algos;
  int64_t m,n,k;
  ~Plan() {if(op)cublasLtMatmulDescDestroy(op); if(a)cublasLtMatrixLayoutDestroy(a); if(b)cublasLtMatrixLayoutDestroy(b); if(c)cublasLtMatrixLayoutDestroy(c);}
};
static std::vector<std::unique_ptr<Plan>> plans;
static cublasLtHandle_t handle() {
  static cublasLtHandle_t h = [] {cublasLtHandle_t t; CK(cublasLtCreate(&t)); return t;}();
  return h;
}
std::vector<int64_t> lt_plan(int64_t m,int64_t n,int64_t k,int64_t bytes) {
  auto p=std::make_unique<Plan>(); p->m=m;p->n=n;p->k=k;
  CK(cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
  CK(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
  CK(cublasLtMatrixLayoutCreate(&p->a,CUDA_R_16BF,k,n,k));
  CK(cublasLtMatrixLayoutCreate(&p->b,CUDA_R_16BF,k,m,k));
  CK(cublasLtMatrixLayoutCreate(&p->c,CUDA_R_16BF,n,m,n));
  cublasLtMatmulPreference_t pref; CK(cublasLtMatmulPreferenceCreate(&pref));
  size_t cap=bytes;
  CK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&cap,sizeof(cap)));
  cublasLtMatmulHeuristicResult_t results[32];int count=0;
  auto status=cublasLtMatmulAlgoGetHeuristic(handle(),p->op,p->a,p->b,p->c,p->c,pref,32,results,&count);
  cublasLtMatmulPreferenceDestroy(pref); CK(status);
  for(int i=0;i<count;i++) if(results[i].state==CUBLAS_STATUS_SUCCESS && results[i].workspaceSize<=cap) p->algos.push_back(results[i].algo);
  int64_t id=plans.size(), size=p->algos.size(); plans.push_back(std::move(p));
  return {id,size};
}
void lt_run(int64_t id,int64_t index,torch::Tensor w,torch::Tensor x,torch::Tensor y,torch::Tensor workspace) {
  TORCH_CHECK(id>=0 && id<int64_t(plans.size()),"invalid plan");
  auto& p=*plans[id];
  TORCH_CHECK(index>=0 && index<int64_t(p.algos.size()),"invalid algorithm");
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && y.is_cuda() && x.is_contiguous() && w.is_contiguous() && y.is_contiguous(),"contiguous CUDA matrices required");
  TORCH_CHECK(x.scalar_type()==torch::kBFloat16 && w.scalar_type()==torch::kBFloat16 && y.scalar_type()==torch::kBFloat16,"BF16 matrices required");
  TORCH_CHECK(x.size(0)==p.m && x.size(1)==p.k && w.size(0)>=p.n && w.size(1)==p.k && y.numel()==p.m*p.n,"shape mismatch");
  const float alpha=1.f,beta=0.f;
  CK(cublasLtMatmul(handle(),p.op,&alpha,w.data_ptr(),p.a,x.data_ptr(),p.b,&beta,y.data_ptr(),p.c,y.data_ptr(),p.c,&p.algos[index],workspace.data_ptr(),workspace.numel(),at::cuda::getCurrentCUDAStream()));
}
'''

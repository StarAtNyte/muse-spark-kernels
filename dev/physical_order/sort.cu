
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdint.h>
struct Begin {int width;__host__ __device__ int operator()(int b)const{return b*width;}};
struct End {int width;const int* seq;__host__ __device__ int operator()(int b)const{return b*width+seq[b];}};
__global__ void pack(const int* table,const int* seq,uint32_t* keys,int n,int width,int bits){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int b=i/width,k=i%width,prefix=seq[b];uint32_t flag=1u<<bits;
 uint32_t item=k<prefix?(uint32_t)table[i]:flag|(k<prefix+8?(uint32_t)(k-prefix):flag-1);
 keys[i]=((uint32_t)b<<(bits+1))|item;
}
__global__ void unpack(const int* table,const int* seq,const uint32_t* keys,int* out,int n,int width,int bits){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 uint32_t item=keys[i]&((1u<<(bits+1))-1),flag=1u<<bits;
 int b=i/width;
 out[i]=(item&flag)?((item&~flag)<8?table[b*width+seq[b]+(item&~flag)]:-1):(int)item;
}
extern "C" int sort_workspace(const int* in,int* out,const int* seq,int B,int width,int bits,int mode,void* tmp,size_t* bytes,uint32_t* encoded,uint32_t* ordered,cudaStream_t stream){
 int n=B*width;
 if(mode==0){
  using Count=cub::CountingInputIterator<int>;
  using BI=cub::TransformInputIterator<int,Begin,Count>;
  using EI=cub::TransformInputIterator<int,End,Count>;
  return cub::DeviceSegmentedRadixSort::SortKeys(tmp,*bytes,in,out,n,B,BI(Count(0),Begin{width}),EI(Count(0),End{width,seq}),0,bits,stream);
 }
 int bbits=0;while((1u<<bbits)<(unsigned)B)bbits++;
 return cub::DeviceRadixSort::SortKeys(tmp,*bytes,encoded,ordered,n,0,bits+1+bbits,stream);
}
extern "C" int order_launch(const int* in,int* out,const int* seq,int B,int width,int bits,int mode,void* tmp,size_t bytes,uint32_t* encoded,uint32_t* ordered,cudaStream_t stream){
 int n=B*width;
 if(mode==0){
  int e=cudaMemcpyAsync(out,in,(size_t)n*4,cudaMemcpyDeviceToDevice,stream);if(e)return e;
 }else pack<<<(n+255)/256,256,0,stream>>>(in,seq,encoded,n,width,bits);
 int e=sort_workspace(in,out,seq,B,width,bits,mode,tmp,&bytes,encoded,ordered,stream);if(e)return e;
 if(mode!=0)unpack<<<(n+255)/256,256,0,stream>>>(in,seq,ordered,out,n,width,bits);
 return cudaGetLastError();
}
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int sort_workspace(const int*,int*,const int*,int,int,int,int,void*,size_t*,uint32_t*,uint32_t*,cudaStream_t);
extern "C" int order_launch(const int*,int*,const int*,int,int,int,int,void*,size_t,uint32_t*,uint32_t*,cudaStream_t);
torch::Tensor physical_sort(torch::Tensor table,torch::Tensor seq,int64_t bits,int64_t mode){
 int B=table.size(0),width=table.size(1),bbits=0;while((1u<<bbits)<(unsigned)B)bbits++;
 TORCH_CHECK(bits>0 && bits<30 && (mode==0 || bits+1+bbits<=32),"packed metadata key overflows uint32");
 auto out=torch::empty_like(table),enc=torch::empty_like(table),ordered=torch::empty_like(table);
 size_t bytes=0;auto stream=at::cuda::getCurrentCUDAStream();
 int e=sort_workspace(table.data_ptr<int>(),out.data_ptr<int>(),seq.data_ptr<int>(),B,width,bits,mode,nullptr,&bytes,(uint32_t*)enc.data_ptr<int>(),(uint32_t*)ordered.data_ptr<int>(),stream);
 TORCH_CHECK(e==0,cudaGetErrorString((cudaError_t)e));
 auto workspace=torch::empty({(long)bytes},table.options().dtype(torch::kUInt8));
 e=order_launch(table.data_ptr<int>(),out.data_ptr<int>(),seq.data_ptr<int>(),B,width,bits,mode,workspace.data_ptr(),bytes,(uint32_t*)enc.data_ptr<int>(),(uint32_t*)ordered.data_ptr<int>(),stream);
 TORCH_CHECK(e==0,cudaGetErrorString((cudaError_t)e));return out;
}

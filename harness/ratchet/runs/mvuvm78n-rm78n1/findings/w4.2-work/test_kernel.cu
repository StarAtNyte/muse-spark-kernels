#include <cuda_runtime.h>
__global__ void test_k(float* x) {
    x[threadIdx.x] += 1.0f;
}

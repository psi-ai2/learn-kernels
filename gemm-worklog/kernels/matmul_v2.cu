#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

inline int CEIL_DIV(int a, int b) { return (a + b - 1) / b; }


__global__ void sgemm_coalesced(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    
}

//Memory copy from host to device

void MatrixMultiplication (const float *A, const float *B, float *C,
                          int M, int N, int K,
                          float alpha = 1.0f, float beta = 0.0f) {

                             size_t sizeA = (size_t)M * K * sizeof(float);
                             size_t sizeB = (size_t)K * N * sizeof(float);
                             size_t sizeC = (size_t)M * N * sizeof(float);

                             float *dA, *dB, *dC;

                             cudaMalloc(&dA, sizeA);
                             cudaMalloc(&dB, sizeB);
                             cudaMalloc(&dC, sizeC);

                             cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice);
                             cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice);
                             
                             if (beta != 0.0f) {
                                cudaMemcpy(dC, C, sizeC, cudaMemcpyHostToDevice);
                             }

                             dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
                             dim3 blockDim(32, 32, 1);

                             sgemm_coalesced<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
                             cudaDeviceSynchronize();


                             cudaFree(dA);
                             cudaFree(dB);
                             cudaFree(dC);
                             
                             
}

int main() {



}

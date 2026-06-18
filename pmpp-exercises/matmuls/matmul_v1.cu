#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

inline int CEIL_DIV(int a, int b) { return (a + b - 1) / b; }


__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            const float *A, const float *B,
                            float beta, float *C) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i)
            tmp += A[row * K + i] * B[i * N + col];
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

void MatrixMultiplication(const float *A, const float *B, float *C,
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
    if (beta != 0.0f)
        cudaMemcpy(dC, C, sizeC, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32, 1);
    dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32), 1);

    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main() {
    const int M = 512, K = 384, N = 256;

    float *A = (float *)malloc((size_t)M * K * sizeof(float));
    float *B = (float *)malloc((size_t)K * N * sizeof(float));
    float *C = (float *)malloc((size_t)M * N * sizeof(float));

    srand(42);
    for (int i = 0; i < M * K; ++i) A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; ++i) B[i] = (float)(rand() % 100) / 100.0f;

    MatrixMultiplication(A, B, C, M, N, K);

    printf("sgemm_naive C(%dx%d) = A(%dx%d) * B(%dx%d)\n", M, N, M, K, K, N);
    printf("C[0]=%.4f  C[last]=%.4f\n", C[0], C[M * N - 1]);

    free(A); free(B); free(C);
    return 0;


}

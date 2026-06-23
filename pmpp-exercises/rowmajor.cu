//Chapter 3 Exercise 1


__global__ void sgemm_rowmajor(int M, int N, int K, float alpha,
                            const float *A, const float *B,
                            float beta, float *C) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        for (int col=0; col < N; col++) {
            float tmp = 0.0f;
            for (int i=0; i < K; ++i) {
                tmp += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = alpha * tmp + beta * C[row*N + col];
        }
}

__global__ void sgemm_colmajor(int M, int N, int K, float alpha, 
                                const float *A, const float* B,
                                float beta, float *C) {
                                    const unsigned int col = blockIdx.x * blockDim.x +threadIdx.x;

                                    if (col < N) {
                                        for (int row=0; row < M; row++) {
                                            float tmp = 0.0f;
                                            for (int i = 0; i < K; ++i) {
                                                tmp += A[row * K + i] * B[i * N + col];
                                            }
                                            C[row * N + col] = alpha * tmp + beta * C[row * N + col];
                                        }
                                    }

                                }
                                

                                

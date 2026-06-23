__global__ void matrixVecKernel(int* A, const int* B,  const int* V, int n) {
    //Goal here is to write a kernel that computes A[i] = sum B[i][j] @ V[i]
    const unsigned int baseIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (baseIdx < n) {
        int tmp = 0.0f;
        for (int j = 0; j < n; ++j) {
            tmp += B[baseIdx * n + j]  + V[j];
            A[baseIdx] = tmp;
        }

    }
}
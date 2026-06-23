//Naive CPU implemementation of vecAdd

__host__ void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    for (int i = 0; i < n; ++i) {
        C_h[i] = A_h[i] + B_h[i];

    }

}

//vecAddKernel using blockIdx and threadIdx to locate specific execution threads
__global__ void vecAddKernel(const float* A_h,const  float* B_h, float* C_h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C_h[i] = A_h[i] + B_h[i];
    }
    
}



//Malloc --> Memcpy --> Free Device
__host__ void vecAdd(const float* A_h, const float* B_h, float* C_h, int n) {
    size_t size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc(&A_d, size);
    cudaMalloc(&B_d, size);
    cudaMalloc(&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1)/threads;

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);



    vecAddKernel<<<blocks, threads>>>(A_d, B_d, C_d, n);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}





 
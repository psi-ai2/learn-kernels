---
tags: [cuda, pmpp, chapter-5]
source: "Programming Massively Parallel Processors — Chapter 5"
---

# Chapter 5 — Memory Architecture and Data Locality

## Exercises

**1.** Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption? Hint: Analyze the elements that are accessed by each thread and see whether there is any commonality between threads.

There are no additional loads for matricies here, so shared memory can not be used to reduce global memory bandwidth as we only need to load each element once for our computation

**2.** Draw the equivalent of Fig. 5.7 for an 8×8 matrix multiplication with 2×2 tiling and 4×4 tiling. Verify that the reduction in global memory bandwidth is indeed proportional to the dimension size of the tiles.

  Skipping for now, will TeX such a figure at some later date (hopefully)

**3.** What type of incorrect execution behavior can happen if one forgot to use one or both `__syncthreads()` in the kernel of Fig. 5.9?

Writing into shared memory like this can lead to race conditions and threads trying to access elements that do not exist when writing back into global memory in the next steps and vice versa, trying to access global memory elements that are being written by other threads. The first is RAW prevention, while the second is WAR prevention.

**4.** Assuming that capacity is not an issue for registers or shared memory, give one important reason why it would be valuable to use shared memory instead of registers to hold values fetched from global memory? Explain your answer.

GPU registers and what they own are privately owned by each thread; whereas, within a block, shared memory variables are capable of being accessed and viewed by any thread within that block (eg. they view the same shared variable)



**5.** For our tiled matrix–matrix multiplication kernel, if we use a 32×32 tile, what is the reduction of memory bandwidth usage for input matrices M and N? 

A 32x32 tile loads in 1024 elements into shared memory for the GEMM, so we no longer have to reload these elements from off-chip memory, so this leads to a 32x reduction.

**6.** Assume that a CUDA kernel is launched with 1000 thread blocks, each of which has 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel? 512,000 versions of this variable will exist, each being local to the thread declaring it.

**7.** In the previous question, if a variable is declared as a shared memory variable, how many versions of the variable will be created through the lifetime of the execution of the kernel? 1000, one shared variable is created for each block, of which there are 1000 blocks.

**8.** Consider performing a matrix multiplication of two input matrices with dimensions N×N. How many times is each element in the input matrices requested from global memory when:
- **a.** There is no tiling? N times
- **b.** Tiles of size T×T are used? T times

**9.** A kernel performs 36 floating-point operations and seven 32-bit global memory accesses per thread. For each of the following device properties, indicate whether this kernel is compute-bound or memory-bound.
- **a.** Peak FLOPS = 200 GFLOPS, peak memory bandwidth = 100 GB/second. Memory Bound. 
- **b.** Peak FLOPS = 300 GFLOPS, peak memory bandwidth = 250 GB/second. Compute Bound.

**10.** To manipulate tiles, a new CUDA programmer has written a device kernel that will transpose each tile in a matrix. The tiles are of size `BLOCK_WIDTH` by `BLOCK_WIDTH`, and each of the dimensions of matrix A is known to be a multiple of `BLOCK_WIDTH`. The kernel invocation and code are shown below. `BLOCK_WIDTH` is known at compile time and could be set anywhere from 1 to 20.

```cuda-cpp
dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
dim3 gridDim(A_width/blockDim.x, A_height/blockDim.y);
BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

__global__ void
BlockTranspose(float* A_elements, int A_width, int A_height)
{
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];

    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
```

- **a.** Out of the possible range of values for `BLOCK_SIZE`, for what values of `BLOCK_SIZE` will this kernel function correctly on the device? There are potential race conditions for threads, so we need them to be in the same warp for lockstep execution. The ideal block sizes are such that BLOCK_SIZE^2 <= 32 (warp size) so 1, 2, 3, 4, 5, this is completely true on pre-volta architectures; however, ITS directly challenges this approach as it no longer guarantees lockstep execution within warp, so this solution is no value is deterministically correct on post-Volta devices.
- **b.** If the code does not execute correctly for all `BLOCK_SIZE` values, what is the root cause of this incorrect execution behavior? Suggest a fix to the code to make it work for all `BLOCK_SIZE` values. We can add _syncthreads to solve RAW  issues after writing to shared memory.

**11.** Consider the following CUDA kernel and the corresponding host function that calls it:

```cuda-cpp
__global__ void foo_kernel(float* a, float* b) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    float x[4];
    __shared__ float y_s;
    __shared__ float b_s[128];
    for(unsigned int j = 0; j < 4; ++j) {
        x[j] = a[j*blockDim.x*gridDim.x + i];
    }
    if(threadIdx.x == 0) {
        y_s = 7.4f;
    }
    b_s[threadIdx.x] = b[i];
    __syncthreads();
    b[i] = 2.5f*x[0] + 3.7f*x[1] + 6.3f*x[2] + 8.5f*x[3]
           + y_s*b_s[threadIdx.x] + b_s[(threadIdx.x + 3)%128];
}
void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel <<< (N + 128 - 1)/128, 128 >>>(a_d, b_d);
}
```

- **a.** How many versions of the variable `i` are there? This is essentially number of threads in the grid. Which is  [(N + 128 - 1)]/128 * 128 for N = 1024, 1024 versions of i
- **b.** How many versions of the array `x[]` are there? 1024 versions of x[] as well.
- **c.** How many versions of the variable `y_s` are there? 8
- **d.** How many versions of the array `b_s[]` are there? 1024/128 = 8 versions
- **e.** What is the amount of shared memory used per block (in bytes)? sizeof(float) * 128 + sizeof(float) = 516 Bytes
- **f.** What is the floating-point to global memory access ratio of the kernel (in OP/B)? 10 FLOPs/24 Bytes = 0.417 OP/B

**12.** Consider a GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM, 64K (65,536) registers/SM, and 96 KB of shared memory/SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.
- **a.** The kernel uses 64 threads/block, 27 registers/thread, and 4 KB of shared memory/SM.  32 blocks of 64 threads precisely achieves full occupancy there. As for registers and shared memory, registers are around 55k < 65536, so we have no issues there. 32 * 4 = 128 KB of shared memory, which does not satisfy occupancy levels, does not achieve full occupancy.
- **b.** The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of shared memory/SM. 8 blocks of 256 threads uses 2048 threads/SM so no issues there. Registers are also fine based on simple computation, shared memory 8*8 = 64 < 96, so we achieve full occupancy on all axises for this kernel
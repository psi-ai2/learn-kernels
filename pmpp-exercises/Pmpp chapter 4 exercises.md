
1. Consider the following CUDA kernel and the corresponding host function that calls it:
    
    ```cuda-cpp
    01    __global__ void foo_kernel(int* a, int* b) {
    02        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    03        if(threadIdx.x < 40 || threadIdx.x >= 104) {
    04            b[i] = a[i] + 1;
    05        }
    06        if(i % 2 == 0) {
    07            a[i] = b[i] * 2;
    08        }
    09        for(unsigned int j = 0; j < 5 - (i % 3); ++j) {
    10            b[i] += j;
    11        }
    12    }
    13    void foo(int* a_d, int* b_d) {
    14        unsigned int N = 1024;
    15        foo_kernel <<< (N + 128 - 1) / 128, 128 >>>(a_d, b_d);
    16    }
    ```
    
    - **a.** What is the number of warps per block? 128 / 32 = 4 
    - **b.** What is the number of warps in the grid? num warps in block * num block in grid = num warps in grid = 4 * 8 = 32
    - **c.** For the statement on line 04: 
        - **i.** How many warps in the grid are active? For  each block, 0-31 active, 32-63 partially active, 64-95 fully inactive, 96-127 partially active, so 1/4 of the block is fully inactive, then 32 - 1/4 * 32 fully inactive in the grid = 24
        - **ii.** How many warps in the grid are divergent? 32 - 2/4 * 32 non-partially activated warps = 16 warps
        - **iii.** What is the SIMD efficiency (in %) of warp 0 of block 0? 32/32 * 100%* = 100%
        - **iv.** What is the SIMD efficiency (in %) of warp 1 of block 0? 8/32 * 100% = 25%
        - **v.** What is the SIMD efficiency (in %) of warp 3 of block 0? 24 / 32 * 100% = 75%
    - **d.** For the statement on line 07:
        - **i.** How many warps in the grid are active? 32
        - **ii.** How many warps in the grid are divergent? 32
        - **iii.** What is the SIMD efficiency (in %) of warp 0 of block 0? 50%
    - **e.** For the loop on line 09:
        - **i.** How many iterations have no divergence? 3
        - **ii.** How many iterations have divergence? 2
1. For a vector addition, assume that the vector length is 2000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid? 2048 threads
    
2. For the previous question, how many warps do you expect to have divergence due to the boundary check on vector length? 1 warp inactive, 1 warp diverges since 48 - 32 fully inactive warps, while 16 remains in a warp with some 50% activity
    
3. Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and 2.9; they spend the rest of their time waiting for the barrier. What percentage of the threads' total execution time is spent waiting for the barrier? Threads wait (still uses execution resources) for the slowest thread (3.0 microseconds), so total execution time is 8 * 3.0 microseconds = 24.0 microseconds. Then for individial threads [i], we calculate he sum of 3.0 - i to get waiting time 1.0 + .7 + .0 + .2 + .6 + 1.1 + .4 + .1 = 4.1, so 4.1/24.0 microseconds spent weighting = 17% total execution time
    
4. A CUDA programmer says that if they launch a kernel with only 32 threads in each block, they can leave out the `__syncthreads()` instruction wherever barrier synchronization is needed. Do you think this is a good idea? Explain. Firstly, a grid with 32 threads in each block leaves a lot to be desired in terms of compute efficiency because SMs are limited by scheduling the blocks resources and a limited number of blocks may be used per SM. In terms of excluding __syncthreads__, this does not work for Volta+ architectures due to independent thread scheduling, which will still cause some race conditions if not carefully synchronized. 
    
5. If a CUDA device's SM can take up to 1536 threads and up to 4 thread blocks, which of the following block configurations would result in the most number of threads in the SM?
    
    - **a.** 128 threads per block
    - **b.** 256 threads per block
    - ==**c.** 512 threads per block==
    - **d.** 1024 threads per block
6. Assume a device that allows up to 64 blocks per SM and 2048 threads per SM. Indicate which of the following assignments per SM are possible. In the cases in which it is possible, indicate the occupancy level.
    
    - **a.** 8 blocks with 128 threads each
    - **b.** 16 blocks with 64 threads each
    - **c.** 32 blocks with 32 threads each
    - ==**d.** 64 blocks with 32 threads each==
    - **e.** 32 blocks with 64 threads each
7. Consider a GPU with the following hardware limits: 2048 threads per SM, 32 blocks per SM, and 64K (65,536) registers per SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.
    
    - **a.** The kernel uses 128 threads per block and 30 registers per thread. This achieves full occupancy.
    - **b.** The kernel uses 32 threads per block and 29 registers per thread. This does not achieve full occupancy. The main issue is the number of threads per block, we may at most call 32 * 32 threads due to the block limitations of the SM, but we are able to schedule 2048 threads per SM with more efficient block sizing. So we may cap at exactly 50% occupancy
    - **c.** The kernel uses 256 threads per block and 34 registers per thread. This does not achieve full occupancy. The limiting factor here is the use of more registers per thread than the registers on the SM. 34 * 2048 > 64K registers (32 registers per thread)
8. A student mentions that they were able to multiply two 1024 × 1024 matrices using a matrix multiplication kernel with 32 × 32 thread blocks. The student is using a CUDA device that allows up to 512 threads per block and up to 8 blocks per SM. The student further mentions that each thread in a thread block calculates one element of the result matrix. What would be your reaction and why?**

My first reaction would be that we are using 32*32 which is 1024 threads thread blocks, but we allow only 512 threads per block, which is not allowed for this CUDA architecture, and would recommend using 16x16 instead.
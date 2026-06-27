# This is from Version 3 of the Book!!!! See Pmpp Chapter 4.exercises.md for the updated solutions


* Blocks can only hold a total of 512 threads, regardless of shapes
* Grids themselves span thousands to millions of light-weight threads for SPMD computation
* dim3 grid/blockDIM(x, y, z) is how we initialize these type of things when we launch new kernels, kernal_name<<gridDim, blockDim>>
* A warp is a further regionalized region of 32 threads, we can do specific warp scheduling to optimize our kernels further. (**Read papers here in future**)
	* Warps are the main units of scheduling threads in a SM.
	* Warps help us with long-latency tasks such as accessing global memory.
	* We hide latency by executing more ready threads while waiting for signals for high latency tasks.

We use __synchthreads()  to make sure each thread reaches a specific point before continuing, we call the calling point of this function a 'barrier' as it quite literally halts execution until all other threads reach it. If not grouped, if-else __syncthreads() could deadlock program if threads don't stop themselves.
	How exactly does the OS handle this?
			CUDA runtime system assign execution resources to all threads (in a block) as a unit, which gives some degree of time proximity between threads within the block.

Kernel Launch --> CUDA runtime system gens grid of threads --> threads given execution resources block-by-block, currently we phrase this as streaming multiprocessors which can provide resources for up to 8 blocks simultaneously (so a device with 30 SMs can provide execution resources to 30 * 8 = 240 blocks simultaneously), and limited to 1024 threads per SM.

__syncthreads() only works on threads within the same block,

> [!question] Question 4.1
> A student mentioned that he was able to multiply two 1024×1024 matrices
> using a tiled matrix multiplication code with 1024 thread blocks on the G80. He further mentioned that each thread in a thread block calculates one element of the result matrix. What would be your reaction and why?

A (1024 * 1024) * (1024 * 1024) matrix multiply requires 2 billion operations. the G80 comes with 768 threads per SM and 16 SMs, so 1024 thread blocks would not be schedulable on each SM as they may only hold 768 < 1024, which is not feasible as we need to launch block threads together. Moreover, as of compute capability 1.0, each block may only hold 512 threads in the first place.

> [!question] Question 4.2
> The following kernel is executed on a large matrix... BLOCK_SIZE is known at
> compile time, but could be set anywhere from 1 to 20.
>
> ```cpp
> dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
> dim3 gridDim(A_width/blockDim.x, A_height/blockDim.y);
> BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);
>
> __global__ void
> BlockTranspose(float* A_elements, int A_width, int A_height)
> {
>     __shared__ float blockA[BLOCK_SIZE][BLOCK_SIZE];
>
>     int baseIdx  = blockIdx.x * BLOCK_SIZE + threadIdx.x;
>     baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;
>
>     blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
>     A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
> }
> ```
>
> Out of the possible range of values for BLOCK_SIZE, for what values will this
> kernel function correctly when executing on the device?

This transposition kernel function involves reading and writing from shared memory, where we write to (ty, tx) and then read our own data; however, there is no barrier to prevent race conditions from occuring, so we must make sure there occurs no READ before WRITE. The only way to ensure this is to ensure the entire tiles are executed as a single warp (forcing lockstep SIMD), so BLOCK_SIZE^2 < 32, BLOCK_SIZE=1,2,3,4,5.


>[!question] Question 4.3
>If the code does not executer correctly for all BLOCK_SIZE values, suggest a fix to the code to make it work for all BLOCK_SIZE values.

```cpp
dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
dim3 gridDim(A_width/blockDim.x, A_height/blockDim.y);
BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

__global__ void
BlockTranspose(float* A_elements, int A_width, int A_height)
{
    __shared__ float blockA[BLOCK_SIZE][BLOCK_SIZE];

    int baseIdx  = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
    __syncthreads();
    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}

```

A simple fix to this problem is calling syncthreads() before writing to shared memory and then reading from shared memory to avoid race conditions by adding a barrier between writes and reads (establishing a write-then-read hierarchy.)

Note to self: Improve flattening arithmetic, make a few sketches and solidify mental model before moving on from here, would hold back quick calculations without understanding these basics well.











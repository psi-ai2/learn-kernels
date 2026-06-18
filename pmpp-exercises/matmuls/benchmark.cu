// Benchmark harness comparing the matmul implementations.
//
//   Build:  nvcc benchmark.cu -O3 -o benchmark.exe
//   Run:    .\benchmark.exe [size] [runs]      (defaults: 512, 5)
//
// The implementations are pulled straight from the exercise files so there
// is a single source of truth. Each file has its own main() for standalone
// use, so we rename those out of the way before including.
//
// To compare a new version: add its file include + a one-line adapter, then
// one entry in the impls[] registry below.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define main matmul_cpu_main
#include "matmul.cu"       
#undef main

#define main matmul_v1_main
#include "matmul_v1.cu"     // GPU: MatrixMultiplication(A, B, C, M, N, K)
#undef main

// Uniform interface: C = A * B  (A: MxK, B: KxN, C: MxN, row-major).
typedef void (*MatmulFn)(const float *A, const float *B, float *C,
                         int M, int N, int K);

// Thin adapters onto the real functions (benchmark uses square sizes).
void cpu_naive(const float *A, const float *B, float *C, int M, int N, int K) {
    MatrixMultiplication((float *)A, (float *)B, C, M);
}
void gpu_naive_v1(const float *A, const float *B, float *C, int M, int N, int K) {
    MatrixMultiplication(A, B, C, M, N, K);
}


struct Impl { const char *name; MatmulFn fn; };
Impl impls[] = {
    {"cpu_naive    (matmul.cu)",    cpu_naive},
    {"gpu_naive_v1 (matmul_v1.cu)", gpu_naive_v1},
};
const int NUM_IMPLS = sizeof(impls) / sizeof(impls[0]);

static double time_ms(MatmulFn fn, const float *A, const float *B, float *C,
                      int M, int N, int K, int runs) {
    using clock = std::chrono::high_resolution_clock;
    fn(A, B, C, M, N, K);  // warmup (also pays one-time CUDA context cost)
    auto t0 = clock::now();
    for (int r = 0; r < runs; ++r) fn(A, B, C, M, N, K);
    auto t1 = clock::now();
    std::chrono::duration<double, std::milli> dt = t1 - t0;
    return dt.count() / runs;
}

int main(int argc, char **argv) {
    int size = (argc > 1) ? atoi(argv[1]) : 512;
    int runs = (argc > 2) ? atoi(argv[2]) : 5;
    const int M = size, N = size, K = size;

    size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
    float *A   = (float *)malloc(nA * sizeof(float));
    float *B   = (float *)malloc(nB * sizeof(float));
    float *C   = (float *)malloc(nC * sizeof(float));
    float *ref = (float *)malloc(nC * sizeof(float));

    srand(42);
    for (size_t i = 0; i < nA; ++i) A[i] = (float)(rand() % 100) / 100.0f;
    for (size_t i = 0; i < nB; ++i) B[i] = (float)(rand() % 100) / 100.0f;

    cpu_naive(A, B, ref, M, N, K);  // ground truth for correctness checks
    const double flops = 2.0 * M * N * K;

    printf("Matmul benchmark  M=N=K=%d  runs=%d\n\n", size, runs);
    printf("%-28s %12s %10s %10s   %s\n",
           "impl", "time(ms)", "GFLOP/s", "speedup", "check");
    printf("----------------------------------------------------------------------------\n");

    FILE *csv = fopen("results.csv", "r");
    bool needHeader = (csv == NULL);
    if (csv) fclose(csv);
    csv = fopen("results.csv", "a");
    if (needHeader) fprintf(csv, "M,N,K,impl,time_ms,gflops\n");

    double baseMs = 0.0;
    for (int i = 0; i < NUM_IMPLS; ++i) {
        double ms = time_ms(impls[i].fn, A, B, C, M, N, K, runs);
        double maxErr = 0.0;
        for (size_t j = 0; j < nC; ++j) {
            double e = fabs((double)C[j] - (double)ref[j]);
            if (e > maxErr) maxErr = e;
        }
        double gflops = flops / (ms / 1000.0) / 1e9;
        if (i == 0) baseMs = ms;

        printf("%-28s %12.3f %10.2f %9.2fx   %s\n",
               impls[i].name, ms, gflops, baseMs / ms,
               maxErr < 1e-2 ? "OK" : "FAIL");
        fprintf(csv, "%d,%d,%d,%s,%.4f,%.4f\n",
                M, N, K, impls[i].name, ms, gflops);
    }
    fclose(csv);

    printf("\nspeedup is relative to %s; results appended to results.csv\n",
           impls[0].name);

    free(A); free(B); free(C); free(ref);
    return 0;
}

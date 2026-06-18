#include <cstdio>
#include <cstdlib>

void MatrixMultiplication(float* M, float* N, float* P, int WIDTH) {
    for (int i = 0; i < WIDTH; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            float sum = 0;
            for (int k = 0; 
                k < WIDTH; ++k) {
                float a = M[i * WIDTH + k];
                float b = N[k * WIDTH + j];
                sum += a * b;
            }
            P[i * WIDTH + j] = sum;
        }
    }
}

int main() {
    const int WIDTH = 4;
    const int n = WIDTH * WIDTH;

    float M[n], N[n], P[n];

    // M and N are identity-scaled / simple values so the result is easy to check.
    for (int i = 0; i < n; ++i) {
        M[i] = (float)(i + 1);   // 1..16
        N[i] = (i % (WIDTH + 1) == 0) ? 1.0f : 0.0f;  // identity matrix
    }

    MatrixMultiplication(M, N, P, WIDTH);

    // Since N is the identity, P should equal M.
    printf("Result P = M * I:\n");
    for (int i = 0; i < WIDTH; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            printf("%6.1f ", P[i * WIDTH + j]);
        }
        printf("\n");
    }

    return 0;
}






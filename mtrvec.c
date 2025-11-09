// matmul_row_major_core.c
// Compile: gcc -O3 -march=native -std=c11 matmul_row_major_core.c -o matmul
// Example: gcc -O3 -march=native -std=c11 matmul_row_major_core.c -o matmul

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>


#ifndef TILE
#define TILE 64
#endif

static inline size_t min_sz(size_t a, size_t b) { return a < b ? a : b; }

/*
  Row-major layout:
    element (i,j) at idx = i*N + j

  Block order: ii, kk, jj
  Inner annotated core uses pointers to avoid repeated index arithmetic.
*/
void matmul_row_major_core(const double *restrict A,
                           const double *restrict B,
                           double *restrict C,
                           size_t N)
{
    memset(C, 0, N * N * sizeof(double)); // zero output

    for (size_t ii = 0; ii < N; ii += TILE) {
        size_t iimax = min_sz(ii + TILE, N);
        for (size_t kk = 0; kk < N; kk += TILE) {
            size_t kkmax = min_sz(kk + TILE, N);
            for (size_t jj = 0; jj < N; jj += TILE) {
                size_t jjmax = min_sz(jj + TILE, N);

                /* Annotated core:
                   - iterate rows i in ii..iimax
                   - for each k in kk..kkmax:
                       load a_ik once
                       set pointer to row k of B
                       set pointer to row i of C
                       inner j loop walks j contiguous via pointers
                */
                for (size_t i = ii; i < iimax; ++i) {
                    const double *Arow = A + i * N;   // start of A[i,*]
                    double *Crow = C + i * N;         // start of C[i,*]

                    for (size_t k = kk; k < kkmax; ++k) {
                        double a_ik = Arow[k];        // A[i,k] loaded once
                        const double *Brow = B + k * N + jj; // pointer to B[k,jj]
                        double *Cptr = Crow + jj;    // pointer to C[i,jj]

                        // inner j loop walks contiguous memory in B row and C row
                        size_t j = jj;
                        for (; j + 3 < jjmax; j += 4) { // small unroll for throughput
                            // unrolled 4-way
                            Cptr[0] += a_ik * Brow[0];
                            Cptr[1] += a_ik * Brow[1];
                            Cptr[2] += a_ik * Brow[2];
                            Cptr[3] += a_ik * Brow[3];
                            Brow += 4;
                            Cptr += 4;
                        }
                        for (; j < jjmax; ++j) { // tail
                            Cptr[0] += a_ik * Brow[0];
                            Brow += 1;
                            Cptr += 1;
                        }
                    }
                }

            }
        }
    }
}

/* Simple correctness test and timing */
int main(void)
{
    const size_t N = 512;
    double *A = NULL, *B = NULL, *C = NULL;
    if (posix_memalign((void**)&A, 64, N*N*sizeof(double)) != 0) return 1;
    if (posix_memalign((void**)&B, 64, N*N*sizeof(double)) != 0) return 1;
    if (posix_memalign((void**)&C, 64, N*N*sizeof(double)) != 0) return 1;

    for (size_t i = 0; i < N*N; ++i) {
        A[i] = (double)((i * 7 + 1) & 0xFF) / 7.0;
        B[i] = (double)((i * 11 + 3) & 0xFF) / 11.0;
    }

    struct timespec t0, t1;
    matmul_row_major_core(A, B, C, N);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("N=%zu TILE=%d elapsed=%.6f s\n", N, TILE, elapsed);

    // spot-check correctness for first 3x3 block
    int ok = 1;
    for (size_t i = 0; i < 3 && ok; ++i)
    for (size_t j = 0; j < 3 && ok; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < N; ++k) sum += A[i*N + k] * B[k*N + j];
        if (fabs(sum - C[i*N + j]) > 1e-9) ok = 0;
    }
    printf("spot-check: %s\n", ok ? "PASS" : "FAIL");

    free(A); free(B); free(C);
    return ok ? 0 : 2;
}

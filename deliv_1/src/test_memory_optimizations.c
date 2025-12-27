/* File:     
 *     test_memory_optimizations.c 
 *
 * Purpose:  
 *     Phase 3: Memory-level optimizations for sparse matrix-vector multiplication
 *     Tests register blocking, software prefetching, and combined approaches
 *     to address memory bandwidth bottlenecks identified in Phase 1 & 2.
 *
 * Compile:  gcc -O3 -Wall -Wextra -march=native -fopenmp -o test_memory_opt \
 *               src/test_memory_optimizations.c src/generator.c src/m_to_csr.c -lm
 * Usage:
 *     ./test_memory_opt <thread_count> <matrix_file> [iterations]
 *     Example: ./test_memory_opt 24 10k_1p5.mtx 30
 *
 * Notes:  
 *     - Builds on best Phase 1 configuration: Static+SIMD+Align+Affin
 *     - Tests memory-level optimizations: register blocking, prefetching
 *     - Uses 90th percentile averaging (30 iterations recommended)
 *     - Focuses on larger matrices (>10k) where memory bottleneck is dominant
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "generator.h"
#include "m_to_csr.h"

#ifdef _WIN32
#include <malloc.h>
#define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ?0 :errno)
#endif

/* Global variables */
int m, n;
float* A;
float* x;
float* y;

/* Configuration structure */
typedef struct {
    char name[64];
    char optimization_type[32];
    double avg_time;
    double median_time;
    double std_dev;
    double min_time;
    double max_time;
    double speedup;
    double improvement_vs_baseline;
} Configuration;

/* Function prototypes */
void serial_mat_vect(float A[], float x[], float y[], int m, int n);

// Phase 1 baseline (best configuration)
void spmv_baseline(int thread_count, csr_matrix *csr_A);

// Register blocking variants (4x and 8x unrolling)
void spmv_regblock_4(int thread_count, csr_matrix *csr_A);
void spmv_regblock_8(int thread_count, csr_matrix *csr_A);

// Software prefetching variants (distance 8 and 16)
void spmv_prefetch_8(int thread_count, csr_matrix *csr_A);
void spmv_prefetch_16(int thread_count, csr_matrix *csr_A);

// Combined optimizations
void spmv_regblock4_prefetch8(int thread_count, csr_matrix *csr_A);
void spmv_regblock4_prefetch16(int thread_count, csr_matrix *csr_A);
void spmv_regblock8_prefetch8(int thread_count, csr_matrix *csr_A);

// Helper functions
void run_benchmark(Configuration *config, int thread_count, csr_matrix *csr_A, 
                   void (*func)(int, csr_matrix*), int iterations);
void print_comparison_table(Configuration configs[], int num_configs, double baseline_time, double phase1_time);
double calculate_std_dev(double times[], int count, double mean);
int compare_doubles(const void *a, const void *b);

/*------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s <thread_count> <matrix_file> [iterations]\n", argv[0]);
        fprintf(stderr, "Example: %s 24 10k_1p5.mtx 30\n", argv[0]);
        exit(1);
    }

    int thread_count = atoi(argv[1]);
    int num_iterations = (argc == 4) ? atoi(argv[3]) : 30;
    
    char matrix_file[256];
    // Support both with and without matrices/ prefix
    if (strchr(argv[2], '/') != NULL) {
        snprintf(matrix_file, sizeof(matrix_file), "%s", argv[2]);
    } else {
        snprintf(matrix_file, sizeof(matrix_file), "matrices/%s", argv[2]);
    }
    
    printf("=== Phase 3: Memory-Level Optimization Testing ===\n");
    printf("Threads: %d\n", thread_count);
    printf("Matrix: %s\n", argv[2]);
    printf("Iterations per config: %d\n", num_iterations);
    printf("Using 90th percentile averaging\n\n");

    // Load matrix
    A = import_matrix(matrix_file, &m, &n);
    if (!A) {
        fprintf(stderr, "Failed to import matrix from %s\n", matrix_file);
        fprintf(stderr, "Ensure matrix file exists and is readable.\n");
        return 1;
    }
    
    // Allocate aligned vectors (64-byte alignment for cache lines)
    if (posix_memalign((void**)&x, 64, (size_t)n * sizeof(float)) != 0) {
        fprintf(stderr, "Failed to allocate aligned memory for x\n");
        free(A);
        return 1;
    }
    
    if (posix_memalign((void**)&y, 64, (size_t)m * sizeof(float)) != 0) {
        fprintf(stderr, "Failed to allocate aligned memory for y\n");
        free(A);
        free(x);
        return 1;
    }
    
    // Initialize x vector with random values
    for (int i = 0; i < n; i++) {
        x[i] = (float)(rand() % 100) / 10.0f;
    }
    
    // Calculate matrix statistics
    int nnz_count = 0;
    size_t total_size = (size_t)m * (size_t)n;
    for (size_t i = 0; i < total_size; i++) {
        if (A[i] != 0.0f) nnz_count++;
    }
    double nnz_percentage = (nnz_count / (double)total_size) * 100.0;
    
    printf("Matrix: %d × %d, %.4f%% non-zero, %d NNZ\n", 
           m, n, nnz_percentage, nnz_count);
    printf("Memory footprint: CSR ~%.2f MB, vectors ~%.2f MB\n\n",
           (nnz_count * (sizeof(float) + sizeof(int)) + (m + 1) * sizeof(int)) / (1024.0 * 1024.0),
           (m + n) * sizeof(float) / (1024.0 * 1024.0));

    // Convert to CSR
    csr_matrix *csr_A;
    if (matrix_to_csr(A, m, n, &csr_A) != 0) {
        fprintf(stderr, "CSR conversion failed\n");
        free(A);
        free(x);
        free(y);
        return 1;
    }

    // Measure serial baseline
    printf("Measuring serial baseline...\n");
    double *serial_times = malloc(num_iterations * sizeof(double));
    if (!serial_times) {
        fprintf(stderr, "Failed to allocate timing arrays\n");
        return 1;
    }
    
    for (int iter = 0; iter < num_iterations; iter++) {
        double start = omp_get_wtime();
        serial_mat_vect(A, x, y, m, n);
        serial_times[iter] = omp_get_wtime() - start;
    }
    
    qsort(serial_times, num_iterations, sizeof(double), compare_doubles);
    int percentile_count = (int)(num_iterations * 0.9);
    if (percentile_count == 0) percentile_count = 1;
    
    double baseline_time = 0.0;
    for (int i = 0; i < percentile_count; i++) {
        baseline_time += serial_times[i];
    }
    baseline_time /= percentile_count;
    free(serial_times);
    
    printf("Serial baseline: %.6f ms\n\n", baseline_time * 1000);

    // Define configurations to test
    printf("Testing memory optimizations...\n\n");
    
    Configuration configs[8];
    int num_configs = 0;

    // Baseline: Phase 1 best configuration (Static+SIMD+Align+Affin)
    strcpy(configs[num_configs].name, "Baseline (Phase 1 winner)");
    strcpy(configs[num_configs].optimization_type, "Static+SIMD+Align+Affin");
    run_benchmark(&configs[num_configs], thread_count, csr_A, spmv_baseline, num_iterations);
    double phase1_time = configs[num_configs].avg_time;
    num_configs++;

    // Register blocking variants
    strcpy(configs[num_configs].name, "RegBlock-4 + Baseline");
    strcpy(configs[num_configs].optimization_type, "Register blocking (4x)");
    run_benchmark(&configs[num_configs], thread_count, csr_A, spmv_regblock_4, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "RegBlock-8 + Baseline");
    strcpy(configs[num_configs].optimization_type, "Register blocking (8x)");
    run_benchmark(&configs[num_configs], thread_count, csr_A, spmv_regblock_8, num_iterations);
    num_configs++;

    // Prefetching variants
    strcpy(configs[num_configs].name, "Prefetch-8 + Baseline");
    strcpy(configs[num_configs].optimization_type, "Software prefetch (d=8)");
    run_benchmark(&configs[num_configs], thread_count, csr_A, spmv_prefetch_8, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Prefetch-16 + Baseline");
    strcpy(configs[num_configs].optimization_type, "Software prefetch (d=16)");
    run_benchmark(&configs[num_configs], thread_count, csr_A, spmv_prefetch_16, num_iterations);
    num_configs++;

    // Combined optimizations
    strcpy(configs[num_configs].name, "RegBlock4 + Prefetch8");
    strcpy(configs[num_configs].optimization_type, "Combined (4x + d=8)");
    run_benchmark(&configs[num_configs], thread_count, csr_A, spmv_regblock4_prefetch8, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "RegBlock4 + Prefetch16");
    strcpy(configs[num_configs].optimization_type, "Combined (4x + d=16)");
    run_benchmark(&configs[num_configs], thread_count, csr_A, spmv_regblock4_prefetch16, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "RegBlock8 + Prefetch8");
    strcpy(configs[num_configs].optimization_type, "Combined (8x + d=8)");
    run_benchmark(&configs[num_configs], thread_count, csr_A, spmv_regblock8_prefetch8, num_iterations);
    num_configs++;

    // Print results
    printf("\n");
    print_comparison_table(configs, num_configs, baseline_time, phase1_time);

    // Cleanup
    free(A);
    free(x);
    free(y);
    csr_free(csr_A);
    
    return 0;
}

/*------------------------------------------------------------------*/
void run_benchmark(Configuration *config, int thread_count, csr_matrix *csr_A, 
                   void (*func)(int, csr_matrix*), int iterations) {
    double *times = malloc(iterations * sizeof(double));
    if (!times) {
        fprintf(stderr, "Failed to allocate timing array\n");
        return;
    }
    
    // Run iterations
    for (int i = 0; i < iterations; i++) {
        double start = omp_get_wtime();
        func(thread_count, csr_A);
        times[i] = omp_get_wtime() - start;
    }
    
    // Sort for percentile calculation
    qsort(times, iterations, sizeof(double), compare_doubles);
    
    // Use 90th percentile (discard worst 10%)
    int percentile_count = (int)(iterations * 0.9);
    if (percentile_count == 0) percentile_count = 1;
    
    // Calculate statistics
    double sum = 0.0;
    config->min_time = times[0];
    config->max_time = times[iterations - 1];
    config->median_time = times[iterations / 2];
    
    for (int i = 0; i < percentile_count; i++) {
        sum += times[i];
    }
    config->avg_time = sum / percentile_count;
    config->std_dev = calculate_std_dev(times, percentile_count, config->avg_time);
    
    free(times);
}

/*------------------------------------------------------------------*/
void print_comparison_table(Configuration configs[], int num_configs, double baseline_time, double phase1_time) {
    printf("╔════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                   Memory Optimization Results                              ║\n");
    printf("╠════════════════════════════════╦═══════════╦═══════════╦══════════╦════════╣\n");
    printf("║ Configuration                  ║  Time(ms) ║ vs Serial ║ vs Phase1║  Gain  ║\n");
    printf("╠════════════════════════════════╬═══════════╬═══════════╬══════════╬════════╣\n");
    
    for (int i = 0; i < num_configs; i++) {
        configs[i].speedup = baseline_time / configs[i].avg_time;
        configs[i].improvement_vs_baseline = ((phase1_time - configs[i].avg_time) / phase1_time) * 100.0;
        
        printf("║ %-30s ║ %9.4f ║ %7.1fx  ║ %7.2f%% ║ ", 
               configs[i].name,
               configs[i].avg_time * 1000,
               configs[i].speedup,
               configs[i].improvement_vs_baseline);
        
        if (i == 0) {
            printf("  ---  ║\n");
        } else if (configs[i].improvement_vs_baseline > 0) {
            printf("+%5.2f%%║\n", configs[i].improvement_vs_baseline);
        } else {
            printf("%6.2f%%║\n", configs[i].improvement_vs_baseline);
        }
    }
    
    printf("╚════════════════════════════════╩═══════════╩═══════════╩══════════╩════════╝\n");
    printf("\nLegend:\n");
    printf("  vs Serial  - Speedup vs serial baseline\n");
    printf("  vs Phase1  - Improvement vs Phase 1 best configuration\n");
    printf("  Gain       - Additional improvement from memory optimizations\n");
}

/*------------------------------------------------------------------*/
double calculate_std_dev(double times[], int count, double mean) {
    double sum_sq_diff = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return sqrt(sum_sq_diff / count);
}

/*------------------------------------------------------------------*/
int compare_doubles(const void *a, const void *b) {
    double diff = (*(double*)a - *(double*)b);
    return (diff > 0) - (diff < 0);
}

/*------------------------------------------------------------------*/
void serial_mat_vect(float A[], float x[], float y[], int m, int n) {
    for (int i = 0; i < m; i++) {
        y[i] = 0.0f;
        for (int j = 0; j < n; j++) {
            y[i] += A[i*n + j] * x[j];
        }
    }
}

/*------------------------------------------------------------------
 * BASELINE: Phase 1 Best Configuration
 * Static+SIMD+Align+Affin (chunk=32)
 */
void spmv_baseline(int thread_count, csr_matrix *csr_A) {
    const int chunk = 32;
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(close) default(none) \
        shared(csr_A, x_aligned, y_aligned, m)
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        
        #pragma omp simd reduction(+:sum) aligned(x_aligned, y_aligned: 64)
        for (int j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++) {
            sum += csr_A->values[j] * x_aligned[csr_A->col_ind[j]];
        }
        
        y_aligned[i] = sum;
    }
}

/*------------------------------------------------------------------
 * REGISTER BLOCKING: 4-way unrolling
 * Uses 4 independent accumulators to hide memory latency
 * and improve instruction-level parallelism
 */
void spmv_regblock_4(int thread_count, csr_matrix *csr_A) {
    const int chunk = 32;
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(close) default(none) \
        shared(csr_A, x_aligned, y_aligned, m)
    for (int i = 0; i < m; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int row_start = csr_A->row_ptr[i];
        int row_end = csr_A->row_ptr[i + 1];
        int j;
        
        // Process 4 elements at a time (register blocking)
        for (j = row_start; j < row_end - 3; j += 4) {
            sum0 += csr_A->values[j]   * x_aligned[csr_A->col_ind[j]];
            sum1 += csr_A->values[j+1] * x_aligned[csr_A->col_ind[j+1]];
            sum2 += csr_A->values[j+2] * x_aligned[csr_A->col_ind[j+2]];
            sum3 += csr_A->values[j+3] * x_aligned[csr_A->col_ind[j+3]];
        }
        
        // Handle remainder elements
        for (; j < row_end; j++) {
            sum0 += csr_A->values[j] * x_aligned[csr_A->col_ind[j]];
        }
        
        y_aligned[i] = sum0 + sum1 + sum2 + sum3;
    }
}

/*------------------------------------------------------------------
 * REGISTER BLOCKING: 8-way unrolling
 * More aggressive unrolling for processors with many registers
 */
void spmv_regblock_8(int thread_count, csr_matrix *csr_A) {
    const int chunk = 32;
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(close) default(none) \
        shared(csr_A, x_aligned, y_aligned, m)
    for (int i = 0; i < m; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
        int row_start = csr_A->row_ptr[i];
        int row_end = csr_A->row_ptr[i + 1];
        int j;
        
        // Process 8 elements at a time
        for (j = row_start; j < row_end - 7; j += 8) {
            sum0 += csr_A->values[j]   * x_aligned[csr_A->col_ind[j]];
            sum1 += csr_A->values[j+1] * x_aligned[csr_A->col_ind[j+1]];
            sum2 += csr_A->values[j+2] * x_aligned[csr_A->col_ind[j+2]];
            sum3 += csr_A->values[j+3] * x_aligned[csr_A->col_ind[j+3]];
            sum4 += csr_A->values[j+4] * x_aligned[csr_A->col_ind[j+4]];
            sum5 += csr_A->values[j+5] * x_aligned[csr_A->col_ind[j+5]];
            sum6 += csr_A->values[j+6] * x_aligned[csr_A->col_ind[j+6]];
            sum7 += csr_A->values[j+7] * x_aligned[csr_A->col_ind[j+7]];
        }
        
        // Handle remainder
        for (; j < row_end; j++) {
            sum0 += csr_A->values[j] * x_aligned[csr_A->col_ind[j]];
        }
        
        y_aligned[i] = (sum0 + sum1 + sum2 + sum3) + (sum4 + sum5 + sum6 + sum7);
    }
}

/*------------------------------------------------------------------
 * SOFTWARE PREFETCHING: Distance 8
 * Prefetches data 8 iterations ahead to hide memory latency
 */
void spmv_prefetch_8(int thread_count, csr_matrix *csr_A) {
    const int chunk = 32;
    const int prefetch_distance = 8;
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(close) default(none) \
        shared(csr_A, x_aligned, y_aligned, m)
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        int row_start = csr_A->row_ptr[i];
        int row_end = csr_A->row_ptr[i + 1];
        
        for (int j = row_start; j < row_end; j++) {
            // Prefetch future data
            if (j + prefetch_distance < row_end) {
                __builtin_prefetch(&x_aligned[csr_A->col_ind[j + prefetch_distance]], 0, 1);
                __builtin_prefetch(&csr_A->values[j + prefetch_distance], 0, 1);
            }
            
            sum += csr_A->values[j] * x_aligned[csr_A->col_ind[j]];
        }
        
        y_aligned[i] = sum;
    }
}

/*------------------------------------------------------------------
 * SOFTWARE PREFETCHING: Distance 16
 * Larger prefetch distance for higher memory latency systems
 */
void spmv_prefetch_16(int thread_count, csr_matrix *csr_A) {
    const int chunk = 32;
    const int prefetch_distance = 16;
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(close) default(none) \
        shared(csr_A, x_aligned, y_aligned, m)
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        int row_start = csr_A->row_ptr[i];
        int row_end = csr_A->row_ptr[i + 1];
        
        for (int j = row_start; j < row_end; j++) {
            if (j + prefetch_distance < row_end) {
                __builtin_prefetch(&x_aligned[csr_A->col_ind[j + prefetch_distance]], 0, 1);
                __builtin_prefetch(&csr_A->values[j + prefetch_distance], 0, 1);
            }
            
            sum += csr_A->values[j] * x_aligned[csr_A->col_ind[j]];
        }
        
        y_aligned[i] = sum;
    }
}

/*------------------------------------------------------------------
 * COMBINED: Register Blocking (4x) + Prefetching (d=8)
 */
void spmv_regblock4_prefetch8(int thread_count, csr_matrix *csr_A) {
    const int chunk = 32;
    const int prefetch_distance = 8;
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(close) default(none) \
        shared(csr_A, x_aligned, y_aligned, m)
    for (int i = 0; i < m; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int row_start = csr_A->row_ptr[i];
        int row_end = csr_A->row_ptr[i + 1];
        int j;
        
        for (j = row_start; j < row_end - 3; j += 4) {
            // Prefetch ahead
            if (j + prefetch_distance < row_end) {
                __builtin_prefetch(&x_aligned[csr_A->col_ind[j + prefetch_distance]], 0, 1);
                __builtin_prefetch(&csr_A->values[j + prefetch_distance], 0, 1);
            }
            
            // Register blocking
            sum0 += csr_A->values[j]   * x_aligned[csr_A->col_ind[j]];
            sum1 += csr_A->values[j+1] * x_aligned[csr_A->col_ind[j+1]];
            sum2 += csr_A->values[j+2] * x_aligned[csr_A->col_ind[j+2]];
            sum3 += csr_A->values[j+3] * x_aligned[csr_A->col_ind[j+3]];
        }
        
        for (; j < row_end; j++) {
            sum0 += csr_A->values[j] * x_aligned[csr_A->col_ind[j]];
        }
        
        y_aligned[i] = sum0 + sum1 + sum2 + sum3;
    }
}

/*------------------------------------------------------------------
 * COMBINED: Register Blocking (4x) + Prefetching (d=16)
 */
void spmv_regblock4_prefetch16(int thread_count, csr_matrix *csr_A) {
    const int chunk = 32;
    const int prefetch_distance = 16;
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(close) default(none) \
        shared(csr_A, x_aligned, y_aligned, m)
    for (int i = 0; i < m; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int row_start = csr_A->row_ptr[i];
        int row_end = csr_A->row_ptr[i + 1];
        int j;
        
        for (j = row_start; j < row_end - 3; j += 4) {
            if (j + prefetch_distance < row_end) {
                __builtin_prefetch(&x_aligned[csr_A->col_ind[j + prefetch_distance]], 0, 1);
                __builtin_prefetch(&csr_A->values[j + prefetch_distance], 0, 1);
            }
            
            sum0 += csr_A->values[j]   * x_aligned[csr_A->col_ind[j]];
            sum1 += csr_A->values[j+1] * x_aligned[csr_A->col_ind[j+1]];
            sum2 += csr_A->values[j+2] * x_aligned[csr_A->col_ind[j+2]];
            sum3 += csr_A->values[j+3] * x_aligned[csr_A->col_ind[j+3]];
        }
        
        for (; j < row_end; j++) {
            sum0 += csr_A->values[j] * x_aligned[csr_A->col_ind[j]];
        }
        
        y_aligned[i] = sum0 + sum1 + sum2 + sum3;
    }
}

/*------------------------------------------------------------------
 * COMBINED: Register Blocking (8x) + Prefetching (d=8)
 */
void spmv_regblock8_prefetch8(int thread_count, csr_matrix *csr_A) {
    const int chunk = 32;
    const int prefetch_distance = 8;
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(close) default(none) \
        shared(csr_A, x_aligned, y_aligned, m)
    for (int i = 0; i < m; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
        int row_start = csr_A->row_ptr[i];
        int row_end = csr_A->row_ptr[i + 1];
        int j;
        
        for (j = row_start; j < row_end - 7; j += 8) {
            if (j + prefetch_distance < row_end) {
                __builtin_prefetch(&x_aligned[csr_A->col_ind[j + prefetch_distance]], 0, 1);
                __builtin_prefetch(&csr_A->values[j + prefetch_distance], 0, 1);
            }
            
            sum0 += csr_A->values[j]   * x_aligned[csr_A->col_ind[j]];
            sum1 += csr_A->values[j+1] * x_aligned[csr_A->col_ind[j+1]];
            sum2 += csr_A->values[j+2] * x_aligned[csr_A->col_ind[j+2]];
            sum3 += csr_A->values[j+3] * x_aligned[csr_A->col_ind[j+3]];
            sum4 += csr_A->values[j+4] * x_aligned[csr_A->col_ind[j+4]];
            sum5 += csr_A->values[j+5] * x_aligned[csr_A->col_ind[j+5]];
            sum6 += csr_A->values[j+6] * x_aligned[csr_A->col_ind[j+6]];
            sum7 += csr_A->values[j+7] * x_aligned[csr_A->col_ind[j+7]];
        }
        
        for (; j < row_end; j++) {
            sum0 += csr_A->values[j] * x_aligned[csr_A->col_ind[j]];
        }
        
        y_aligned[i] = (sum0 + sum1 + sum2 + sum3) + (sum4 + sum5 + sum6 + sum7);
    }
}

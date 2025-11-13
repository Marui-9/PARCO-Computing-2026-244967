/* File:     
 *     test_configurations_numa.c 
 *
 * Purpose:  
 *     NUMA-aware OpenMP configurations for high thread counts (24-96 threads)
 *     Tests scheduling strategies optimized for multi-NUMA node systems.
 *
 * Compile:  gcc -g -Wall -O3 -fopenmp -o test_config_numa \
 *               test_configurations_numa.c generator.c m_to_csr.c -lm -lnuma
 * Usage:
 *     ./test_config_numa <thread_count> <matrix_file> [iterations]
 *     Example: ./test_config_numa 48 large_matrix.mtx 30
 *
 * Thread counts: 24, 28, 32, 36, 40, 44, 48, 54, 60, 66, 72, 78, 84, 90, 96
 *
 * Notes:  
 *     - Optimized for 4 NUMA nodes (24 cores each)
 *     - Tests proc_bind policies: close, spread, master
 *     - Includes first-touch initialization
 *     - Reduced default iterations (30) for large matrices
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "generator.h"
#include "m_to_csr.h"

/* Global variables */
int m, n;
float* A;
float* x;
float* y;

/* Configuration structure */
typedef struct {
    char name[80];
    char schedule_type[20];
    char bind_policy[20];
    int chunk_size;
    double avg_time;
    double median_time;
    double std_dev;
    double min_time;
    double max_time;
    double speedup;
    double efficiency_pct;
} Configuration;

/* Function prototypes */
void first_touch_init(int thread_count);
void mat_vect_static_simd_close(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_static_simd_spread(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_static_simd_master(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_dynamic_simd_spread(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_guided_simd_spread(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_static_simd_register_close(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_static_simd_register_spread(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_static_simd_affinity_close(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_static_simd_affinity_spread(int thread_count, csr_matrix *csr_A, int chunk);
void serial_mat_vect(float A[], float x[], float y[], int m, int n);
void run_benchmark(Configuration *config, int thread_count, csr_matrix *csr_A, 
                   void (*func)(int, csr_matrix*, int), int chunk, int iterations);
void print_comparison_table(Configuration configs[], int num_configs, double baseline_time, int thread_count);
double calculate_std_dev(double times[], int count, double mean);
int compare_doubles(const void *a, const void *b);

/*------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s <thread_count> <matrix_file> [iterations]\n", argv[0]);
        fprintf(stderr, "Example: %s 48 large_matrix.mtx 30\n", argv[0]);
        fprintf(stderr, "Thread counts: 24-96 (increment by 4 up to 48, then by 6)\n");
        exit(1);
    }

    int thread_count = atoi(argv[1]);
    int num_iterations = (argc == 4) ? atoi(argv[3]) : 30;  // Lower default for large matrices
    
    // Validate thread count for NUMA system
    if (thread_count < 24 || thread_count > 96) {
        fprintf(stderr, "Warning: Thread count %d outside recommended range [24-96]\n", thread_count);
    }
    
    char matrix_file[256];
    snprintf(matrix_file, sizeof(matrix_file), "matrices_large/%s", argv[2]);
    
    printf("=== NUMA-Aware OpenMP Configuration Comparison ===\n");
    printf("Threads: %d\n", thread_count);
    printf("Matrix: %s\n", argv[2]);
    printf("Iterations per config: %d\n", num_iterations);
    printf("Using 90th percentile averaging\n");
    printf("NUMA nodes: 4 (assumed 24 cores each)\n\n");

    // Load matrix
    A = import_matrix(matrix_file, &m, &n);
    if (!A) {
        fprintf(stderr, "Failed to import matrix from %s\n", matrix_file);
        fprintf(stderr, "Trying matrices/ directory...\n");
        snprintf(matrix_file, sizeof(matrix_file), "matrices/%s", argv[2]);
        A = import_matrix(matrix_file, &m, &n);
        if (!A) {
            fprintf(stderr, "Failed to import matrix\n");
            return 1;
        }
    }
    
    // Use aligned vector allocation for better cache performance
    x = generate_vector_aligned(n);
    
    // Allocate y with cache line alignment (64 bytes) for better performance
    if (posix_memalign((void**)&y, 64, (size_t)m * sizeof(float)) != 0) {
        fprintf(stderr, "Failed to allocate aligned memory for y\n");
        free(A);
        free(x);
        return 1;
    }
    
    // First-touch initialization for NUMA awareness
    printf("Performing first-touch initialization...\n");
    first_touch_init(thread_count);
    
    // Calculate matrix statistics
    int nnz_count = 0;
    size_t total_size = (size_t)m * (size_t)n;
    for (size_t i = 0; i < total_size; i++) {
        if (A[i] != 0.0f) nnz_count++;
    }
    double nnz_percentage = (nnz_count / (double)total_size) * 100.0;
    
    printf("Matrix: %d × %d, %.4f%% non-zero, %d NNZ\n\n", 
           m, n, nnz_percentage, nnz_count);

    // Convert to CSR
    csr_matrix *csr_A;
    if (matrix_to_csr(A, m, n, &csr_A) != 0) {
        fprintf(stderr, "CSR conversion failed\n");
        free(A);
        return 1;
    }

    // Measure serial baseline
    printf("Measuring serial baseline...\n");
    double *serial_times = malloc(num_iterations * sizeof(double));
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

    // Define NUMA-aware configurations to test
    Configuration configs[20];
    int num_configs = 0;

    printf("Testing NUMA-aware configurations...\n");
    
    // Static + SIMD with different binding policies
    strcpy(configs[num_configs].name, "Static+SIMD+close");
    strcpy(configs[num_configs].schedule_type, "static");
    strcpy(configs[num_configs].bind_policy, "close");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static_simd_close, 32, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Static+SIMD+spread");
    strcpy(configs[num_configs].schedule_type, "static");
    strcpy(configs[num_configs].bind_policy, "spread");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static_simd_spread, 32, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Static+SIMD+master");
    strcpy(configs[num_configs].schedule_type, "static");
    strcpy(configs[num_configs].bind_policy, "master");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static_simd_master, 32, num_iterations);
    num_configs++;

    // Dynamic + SIMD with spread (NUMA-friendly)
    strcpy(configs[num_configs].name, "Dynamic+SIMD+spread, chunk=16");
    strcpy(configs[num_configs].schedule_type, "dynamic");
    strcpy(configs[num_configs].bind_policy, "spread");
    configs[num_configs].chunk_size = 16;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_dynamic_simd_spread, 16, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Dynamic+SIMD+spread, chunk=32");
    strcpy(configs[num_configs].schedule_type, "dynamic");
    strcpy(configs[num_configs].bind_policy, "spread");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_dynamic_simd_spread, 32, num_iterations);
    num_configs++;

    // Guided + SIMD with spread
    strcpy(configs[num_configs].name, "Guided+SIMD+spread, chunk=16");
    strcpy(configs[num_configs].schedule_type, "guided");
    strcpy(configs[num_configs].bind_policy, "spread");
    configs[num_configs].chunk_size = 16;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_guided_simd_spread, 16, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Guided+SIMD+spread, chunk=32");
    strcpy(configs[num_configs].schedule_type, "guided");
    strcpy(configs[num_configs].bind_policy, "spread");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_guided_simd_spread, 32, num_iterations);
    num_configs++;

    // Register blocking with different binding policies
    strcpy(configs[num_configs].name, "Static+SIMD+Register+close");
    strcpy(configs[num_configs].schedule_type, "static");
    strcpy(configs[num_configs].bind_policy, "close");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static_simd_register_close, 32, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Static+SIMD+Register+spread");
    strcpy(configs[num_configs].schedule_type, "static");
    strcpy(configs[num_configs].bind_policy, "spread");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static_simd_register_spread, 32, num_iterations);
    num_configs++;

    // Affinity optimization with binding policies
    strcpy(configs[num_configs].name, "Static+SIMD+Affinity+close");
    strcpy(configs[num_configs].schedule_type, "static");
    strcpy(configs[num_configs].bind_policy, "close");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static_simd_affinity_close, 32, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Static+SIMD+Affinity+spread");
    strcpy(configs[num_configs].schedule_type, "static");
    strcpy(configs[num_configs].bind_policy, "spread");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static_simd_affinity_spread, 32, num_iterations);
    num_configs++;

    // Print results
    print_comparison_table(configs, num_configs, baseline_time, thread_count);

    // Cleanup
    free(A);
    free(x);
    free(y);
    csr_free(csr_A);
    
    return 0;
}

/*------------------------------------------------------------------
 * First-touch initialization for NUMA awareness
 * Initializes y vector in parallel so each thread's data is on its NUMA node
 */
void first_touch_init(int thread_count) {
    #pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int i = 0; i < m; i++) {
        y[i] = 0.0f;
    }
}

/* Serial baseline */
void serial_mat_vect(float A[], float x[], float y[], int m, int n) {
    for (int i = 0; i < m; i++) {
        y[i] = 0.0f;
        for (int j = 0; j < n; j++) {
            y[i] += A[i*n + j] * x[j];
        }
    }
}

/* NUMA-aware implementations */

// Static + SIMD + close binding (good for single NUMA node)
void mat_vect_static_simd_close(int thread_count, csr_matrix *csr_A, int chunk) {
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(close) default(none) shared(csr_A, x_aligned, y_aligned, m, chunk)
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum) aligned(x_aligned, y_aligned: 64)
        for (int j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++) {
            sum += csr_A->values[j] * x_aligned[csr_A->col_ind[j]];
        }
        y_aligned[i] = sum;
    }
}

// Static + SIMD + spread binding (distributes threads across NUMA nodes)
void mat_vect_static_simd_spread(int thread_count, csr_matrix *csr_A, int chunk) {
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(spread) default(none) shared(csr_A, x_aligned, y_aligned, m, chunk)
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum) aligned(x_aligned, y_aligned: 64)
        for (int j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++) {
            sum += csr_A->values[j] * x_aligned[csr_A->col_ind[j]];
        }
        y_aligned[i] = sum;
    }
}

// Static + SIMD + master binding (keeps threads close to master)
void mat_vect_static_simd_master(int thread_count, csr_matrix *csr_A, int chunk) {
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(master) default(none) shared(csr_A, x_aligned, y_aligned, m, chunk)
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum) aligned(x_aligned, y_aligned: 64)
        for (int j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++) {
            sum += csr_A->values[j] * x_aligned[csr_A->col_ind[j]];
        }
        y_aligned[i] = sum;
    }
}

// Dynamic + SIMD + spread (good for load balancing across NUMA nodes)
void mat_vect_dynamic_simd_spread(int thread_count, csr_matrix *csr_A, int chunk) {
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(dynamic, chunk) num_threads(thread_count) \
        proc_bind(spread) default(none) shared(csr_A, x_aligned, y_aligned, m, chunk)
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum) aligned(x_aligned, y_aligned: 64)
        for (int j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++) {
            sum += csr_A->values[j] * x_aligned[csr_A->col_ind[j]];
        }
        y_aligned[i] = sum;
    }
}

// Guided + SIMD + spread
void mat_vect_guided_simd_spread(int thread_count, csr_matrix *csr_A, int chunk) {
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(guided, chunk) num_threads(thread_count) \
        proc_bind(spread) default(none) shared(csr_A, x_aligned, y_aligned, m, chunk)
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum) aligned(x_aligned, y_aligned: 64)
        for (int j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++) {
            sum += csr_A->values[j] * x_aligned[csr_A->col_ind[j]];
        }
        y_aligned[i] = sum;
    }
}

// Static + SIMD + Register blocking + close
void mat_vect_static_simd_register_close(int thread_count, csr_matrix *csr_A, int chunk) {
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(close) default(none) shared(csr_A, x_aligned, y_aligned, m, chunk)
    for (int i = 0; i < m; i++) {
        register float sum = 0.0f;
        #pragma omp simd reduction(+:sum) aligned(x_aligned, y_aligned: 64)
        for (int j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++) {
            sum += csr_A->values[j] * x_aligned[csr_A->col_ind[j]];
        }
        y_aligned[i] = sum;
    }
}

// Static + SIMD + Register blocking + spread
void mat_vect_static_simd_register_spread(int thread_count, csr_matrix *csr_A, int chunk) {
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(spread) default(none) shared(csr_A, x_aligned, y_aligned, m, chunk)
    for (int i = 0; i < m; i++) {
        register float sum = 0.0f;
        #pragma omp simd reduction(+:sum) aligned(x_aligned, y_aligned: 64)
        for (int j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++) {
            sum += csr_A->values[j] * x_aligned[csr_A->col_ind[j]];
        }
        y_aligned[i] = sum;
    }
}

// Static + SIMD + Affinity + close
void mat_vect_static_simd_affinity_close(int thread_count, csr_matrix *csr_A, int chunk) {
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    float *vals_aligned = __builtin_assume_aligned(csr_A->values, 64);
    
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(close) default(none) shared(csr_A, x_aligned, y_aligned, vals_aligned, m, chunk)
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum) aligned(x_aligned, y_aligned, vals_aligned: 64)
        for (int j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++) {
            sum += vals_aligned[j] * x_aligned[csr_A->col_ind[j]];
        }
        y_aligned[i] = sum;
    }
}

// Static + SIMD + Affinity + spread
void mat_vect_static_simd_affinity_spread(int thread_count, csr_matrix *csr_A, int chunk) {
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    float *vals_aligned = __builtin_assume_aligned(csr_A->values, 64);
    
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(spread) default(none) shared(csr_A, x_aligned, y_aligned, vals_aligned, m, chunk)
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum) aligned(x_aligned, y_aligned, vals_aligned: 64)
        for (int j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++) {
            sum += vals_aligned[j] * x_aligned[csr_A->col_ind[j]];
        }
        y_aligned[i] = sum;
    }
}

/* Benchmark runner */
void run_benchmark(Configuration *config, int thread_count, csr_matrix *csr_A, 
                   void (*func)(int, csr_matrix*, int), int chunk, int iterations) {
    double *times = malloc(iterations * sizeof(double));
    
    for (int iter = 0; iter < iterations; iter++) {
        double start = omp_get_wtime();
        func(thread_count, csr_A, chunk);
        times[iter] = omp_get_wtime() - start;
    }
    
    qsort(times, iterations, sizeof(double), compare_doubles);
    int percentile_count = (int)(iterations * 0.9);
    if (percentile_count == 0) percentile_count = 1;
    
    double sum = 0.0;
    for (int i = 0; i < percentile_count; i++) {
        sum += times[i];
    }
    
    config->avg_time = sum / percentile_count;
    config->median_time = times[percentile_count / 2];
    config->min_time = times[0];
    config->max_time = times[iterations - 1];
    config->std_dev = calculate_std_dev(times, percentile_count, config->avg_time);
    
    free(times);
}

/* Print comparison table */
void print_comparison_table(Configuration configs[], int num_configs, double baseline_time, int thread_count) {
    printf("\n=== Configuration Comparison (sorted by speedup) ===\n");
    printf("%-40s %10s %10s %10s %10s\n", 
           "Configuration", "Time(ms)", "Speedup", "Efficiency", "Improvement");
    printf("%-40s %10s %10s %10s %10s\n", 
           "----------------------------------------", "----------", "----------", "----------", "----------");
    
    // Calculate speedups
    for (int i = 0; i < num_configs; i++) {
        configs[i].speedup = baseline_time / configs[i].avg_time;
        configs[i].efficiency_pct = (configs[i].speedup / thread_count) * 100.0;
    }
    
    // Sort by speedup (bubble sort for simplicity)
    for (int i = 0; i < num_configs - 1; i++) {
        for (int j = 0; j < num_configs - i - 1; j++) {
            if (configs[j].speedup < configs[j+1].speedup) {
                Configuration temp = configs[j];
                configs[j] = configs[j+1];
                configs[j+1] = temp;
            }
        }
    }
    
    double best_parallel_time = configs[0].avg_time;
    
    for (int i = 0; i < num_configs; i++) {
        double improvement_pct = ((best_parallel_time - configs[i].avg_time) / best_parallel_time) * 100.0;
        printf("%-40s %10.2f %10.2fx %9.2f%% %+9.2f%%\n",
               configs[i].name,
               configs[i].avg_time * 1000,
               configs[i].speedup,
               configs[i].efficiency_pct,
               improvement_pct);
    }
    
    printf("\nBest configuration: %s\n", configs[0].name);
    printf("Best speedup: %.2fx (%.2f%% efficiency)\n", configs[0].speedup, configs[0].efficiency_pct);
}

/* Utility functions */
double calculate_std_dev(double times[], int count, double mean) {
    double sum_sq_diff = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = times[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return sqrt(sum_sq_diff / count);
}

int compare_doubles(const void *a, const void *b) {
    double diff = (*(double*)a - *(double*)b);
    return (diff > 0) - (diff < 0);
}

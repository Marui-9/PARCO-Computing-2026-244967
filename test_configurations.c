/* File:     
 *     test_configurations.c 
 *
 * Purpose:  
 *     Compare different OpenMP scheduling strategies and configurations
 *     for parallel sparse matrix-vector multiplication.
 *
 * Compile:  gcc -g -Wall -O3 -march=native -fopenmp -o test_config \
 *               test_configurations.c generator.c m_to_csr.c -lm
 * Usage:
 *     ./test_config <thread_count> <matrix_file> [iterations]
 *     Example: ./test_config 8 2k_0p52.mtx 50
 *
 * Notes:  
 *     - Runs multiple configurations and outputs comparison table
 *     - Default: 50 iterations per configuration (faster than main.c)
 *     - Uses 90th percentile averaging
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
    char name[50];
    char schedule_type[20];
    int chunk_size;
    double avg_time;
    double median_time;
    double std_dev;
    double min_time;
    double max_time;
    double speedup;
} Configuration;

/* Function prototypes */
void mat_vect_static(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_dynamic(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_guided(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_auto(int thread_count, csr_matrix *csr_A);
void mat_vect_static_simd(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_dynamic_simd(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_guided_simd(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_static_simd_affinity(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_guided_simd_affinity(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_guided_simd_register(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_static_simd_register(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_task_based(int thread_count, csr_matrix *csr_A);
void mat_vect_guided_simd_aligned(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_static_simd_aligned(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_static_collapsed(int thread_count, csr_matrix *csr_A, int chunk);
void mat_vect_dynamic_collapsed(int thread_count, csr_matrix *csr_A, int chunk);
void serial_mat_vect(float A[], float x[], float y[], int m, int n);
void run_benchmark(Configuration *config, int thread_count, csr_matrix *csr_A, 
                   void (*func)(int, csr_matrix*, int), int chunk, int iterations);
void print_comparison_table(Configuration configs[], int num_configs, double baseline_time);
double calculate_std_dev(double times[], int count, double mean);
int compare_doubles(const void *a, const void *b);

/*------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s <thread_count> <matrix_file> [iterations]\n", argv[0]);
        fprintf(stderr, "Example: %s 8 2k_0p52.mtx 50\n", argv[0]);
        exit(1);
    }

    int thread_count = atoi(argv[1]);
    int num_iterations = (argc == 4) ? atoi(argv[3]) : 50;
    
    char matrix_file[256];
    snprintf(matrix_file, sizeof(matrix_file), "matrices/%s", argv[2]);
    
    printf("=== OpenMP Configuration Comparison ===\n");
    printf("Threads: %d\n", thread_count);
    printf("Matrix: %s\n", argv[2]);
    printf("Iterations per config: %d\n", num_iterations);
    printf("Using 90th percentile averaging\n\n");

    // Load matrix
    A = import_matrix(matrix_file, &m, &n);
    if (!A) {
        fprintf(stderr, "Failed to import matrix from %s\n", matrix_file);
        return 1;
    }
    
    // Use aligned vector allocation for better cache performance
    x = generate_vector_aligned(n);
    
    // Allocate y with cache line alignment (64 bytes) for better performance
    // Both x and y are now aligned to 64-byte boundaries for optimal SIMD access
    if (posix_memalign((void**)&y, 64, (size_t)m * sizeof(float)) != 0) {
        fprintf(stderr, "Failed to allocate aligned memory for y\n");
        free(A);
        free(x);
        return 1;
    }
    
    // Calculate matrix statistics
    int nnz_count = 0;
    size_t total_size = (size_t)m * (size_t)n;
    for (size_t i = 0; i < total_size; i++) {
        if (A[i] != 0.0f) nnz_count++;
    }
    double nnz_percentage = (nnz_count / (double)total_size) * 100.0;
    
    printf("Matrix: %d × %d, %.2f%% non-zero, %d NNZ\n\n", 
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

    // Define configurations to test
    Configuration configs[30];
    int num_configs = 0;

    // Static scheduling with different chunk sizes
    printf("Testing configurations...\n");
    
    strcpy(configs[num_configs].name, "Static (default)");
    strcpy(configs[num_configs].schedule_type, "static");
    configs[num_configs].chunk_size = 0;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static, 0, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Static, chunk=8");
    strcpy(configs[num_configs].schedule_type, "static");
    configs[num_configs].chunk_size = 8;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static, 8, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Static, chunk=32");
    strcpy(configs[num_configs].schedule_type, "static");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static, 32, num_iterations);
    num_configs++;

    // Dynamic scheduling
    strcpy(configs[num_configs].name, "Dynamic, chunk=8");
    strcpy(configs[num_configs].schedule_type, "dynamic");
    configs[num_configs].chunk_size = 8;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_dynamic, 8, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Dynamic, chunk=16");
    strcpy(configs[num_configs].schedule_type, "dynamic");
    configs[num_configs].chunk_size = 16;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_dynamic, 16, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Dynamic, chunk=32");
    strcpy(configs[num_configs].schedule_type, "dynamic");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_dynamic, 32, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Dynamic, chunk=64");
    strcpy(configs[num_configs].schedule_type, "dynamic");
    configs[num_configs].chunk_size = 64;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_dynamic, 64, num_iterations);
    num_configs++;

    // Guided scheduling
    strcpy(configs[num_configs].name, "Guided, chunk=8");
    strcpy(configs[num_configs].schedule_type, "guided");
    configs[num_configs].chunk_size = 8;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_guided, 8, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Guided, chunk=16");
    strcpy(configs[num_configs].schedule_type, "guided");
    configs[num_configs].chunk_size = 16;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_guided, 16, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Guided, chunk=32");
    strcpy(configs[num_configs].schedule_type, "guided");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_guided, 32, num_iterations);
    num_configs++;

    // Auto scheduling
    strcpy(configs[num_configs].name, "Auto (runtime)");
    strcpy(configs[num_configs].schedule_type, "auto");
    configs[num_configs].chunk_size = 0;
    run_benchmark(&configs[num_configs], thread_count, csr_A, (void (*)(int, csr_matrix*, int))mat_vect_auto, 0, num_iterations);
    num_configs++;

    // SIMD variants
    strcpy(configs[num_configs].name, "Static+SIMD, chunk=8");
    strcpy(configs[num_configs].schedule_type, "static");
    configs[num_configs].chunk_size = 8;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static_simd, 8, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Static+SIMD, chunk=16");
    strcpy(configs[num_configs].schedule_type, "static");
    configs[num_configs].chunk_size = 16;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static_simd, 16, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Static+SIMD, chunk=32");
    strcpy(configs[num_configs].schedule_type, "static");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static_simd, 32, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Dynamic+SIMD, chunk=8");
    strcpy(configs[num_configs].schedule_type, "dynamic");
    configs[num_configs].chunk_size = 8;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_dynamic_simd, 8, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Dynamic+SIMD, chunk=16");
    strcpy(configs[num_configs].schedule_type, "dynamic");
    configs[num_configs].chunk_size = 16;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_dynamic_simd, 16, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Dynamic+SIMD, chunk=32");
    strcpy(configs[num_configs].schedule_type, "dynamic");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_dynamic_simd, 32, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Guided+SIMD, chunk=8");
    strcpy(configs[num_configs].schedule_type, "guided");
    configs[num_configs].chunk_size = 8;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_guided_simd, 8, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Guided+SIMD, chunk=16");
    strcpy(configs[num_configs].schedule_type, "guided");
    configs[num_configs].chunk_size = 16;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_guided_simd, 16, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Guided+SIMD, chunk=32");
    strcpy(configs[num_configs].schedule_type, "guided");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_guided_simd, 32, num_iterations);
    num_configs++;

    // Optimized versions with thread affinity
    strcpy(configs[num_configs].name, "Static+SIMD+Affinity, 16");
    strcpy(configs[num_configs].schedule_type, "static");
    configs[num_configs].chunk_size = 16;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static_simd_affinity, 16, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Static+SIMD+Affinity, 32");
    strcpy(configs[num_configs].schedule_type, "static");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static_simd_affinity, 32, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Guided+SIMD+Affinity, 32");
    strcpy(configs[num_configs].schedule_type, "guided");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_guided_simd_affinity, 32, num_iterations);
    num_configs++;

    // Ultra-optimized versions with register hints
    strcpy(configs[num_configs].name, "Guided+SIMD+Register, 32");
    strcpy(configs[num_configs].schedule_type, "guided");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_guided_simd_register, 32, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Guided+SIMD+Register, 16");
    strcpy(configs[num_configs].schedule_type, "guided");
    configs[num_configs].chunk_size = 16;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_guided_simd_register, 16, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Static+SIMD+Register, 32");
    strcpy(configs[num_configs].schedule_type, "static");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static_simd_register, 32, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Static+SIMD+Register, 16");
    strcpy(configs[num_configs].schedule_type, "static");
    configs[num_configs].chunk_size = 16;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static_simd_register, 16, num_iterations);
    num_configs++;

    // Task-based parallelism (experimental)
    strcpy(configs[num_configs].name, "Task-based+SIMD");
    strcpy(configs[num_configs].schedule_type, "task");
    configs[num_configs].chunk_size = 0;
    run_benchmark(&configs[num_configs], thread_count, csr_A, (void (*)(int, csr_matrix*, int))mat_vect_task_based, 0, num_iterations);
    num_configs++;

    // Cache-aligned versions
    strcpy(configs[num_configs].name, "Guided+SIMD+Align+Affin, 32");
    strcpy(configs[num_configs].schedule_type, "guided");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_guided_simd_aligned, 32, num_iterations);
    num_configs++;

    strcpy(configs[num_configs].name, "Static+SIMD+Align+Affin, 32");
    strcpy(configs[num_configs].schedule_type, "static");
    configs[num_configs].chunk_size = 32;
    run_benchmark(&configs[num_configs], thread_count, csr_A, mat_vect_static_simd_aligned, 32, num_iterations);
    num_configs++;

    // Print comparison table
    printf("\n");
    print_comparison_table(configs, num_configs, baseline_time);

    // Cleanup
    free(A);
    free(x);
    free(y);
    csr_free(csr_A);
    
    return 0;
}

/*------------------------------------------------------------------*/
void run_benchmark(Configuration *config, int thread_count, csr_matrix *csr_A, 
                   void (*func)(int, csr_matrix*, int), int chunk, int iterations) {
    double *times = malloc(iterations * sizeof(double));
    
    printf("  Testing: %s\n", config->name);
    
    for (int iter = 0; iter < iterations; iter++) {
        double start = omp_get_wtime();
        func(thread_count, csr_A, chunk);
        times[iter] = omp_get_wtime() - start;
    }
    
    // Sort times
    qsort(times, iterations, sizeof(double), compare_doubles);
    
    // Use 90th percentile
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

/*------------------------------------------------------------------*/
void print_comparison_table(Configuration configs[], int num_configs, double baseline_time) {
    printf("=== Performance Comparison Results ===\n\n");
    printf("%-25s %12s %10s %10s %10s %12s\n", 
           "Configuration", "Time(ms)", "Speedup", "Effic(%)", "StdDev", "Improv(%)");
    printf("%.100s\n", "----------------------------------------------------------------------------------------------------");
    
    // Find best configuration
    int best_idx = 0;
    for (int i = 1; i < num_configs; i++) {
        if (configs[i].avg_time < configs[best_idx].avg_time) {
            best_idx = i;
        }
    }
    
    for (int i = 0; i < num_configs; i++) {
        double time_ms = configs[i].avg_time * 1000;
        double speedup = baseline_time / configs[i].avg_time;
        double efficiency = (speedup / omp_get_max_threads()) * 100.0;
        double std_dev_ms = configs[i].std_dev * 1000;
        double improvement = ((configs[0].avg_time - configs[i].avg_time) / configs[0].avg_time) * 100.0;
        
        configs[i].speedup = speedup;
        
        printf("%-25s %11.3f %9.2fx %9.1f%% %9.3fms %11.2f%%", 
               configs[i].name, time_ms, speedup, efficiency, std_dev_ms, improvement);
        
        if (i == best_idx) {
            printf(" ← BEST");
        } else if (i == 0) {
            printf(" (baseline)");
        }
        printf("\n");
    }
    
    printf("\n");
    printf("Notes:\n");
    printf("  - Baseline: First configuration (typically static default)\n");
    printf("  - Efficiency: Speedup / max_threads\n");
    printf("  - Best config: %s (%.2fx speedup, %.2f%% improvement)\n",
           configs[best_idx].name, configs[best_idx].speedup,
           ((configs[0].avg_time - configs[best_idx].avg_time) / configs[0].avg_time) * 100.0);
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
    int i, j;
    for (i = 0; i < m; i++) {
        y[i] = 0.0f;
        for (j = 0; j < n; j++)
            y[i] += A[i*n + j] * x[j];
    }
}

/*------------------------------------------------------------------*/
void mat_vect_static(int thread_count, csr_matrix *csr_A, int chunk) {
    int i, j;
    if (chunk == 0) {
        #pragma omp parallel for schedule(static) num_threads(thread_count) \
            default(none) shared(csr_A, x, y, m, n) private(i, j)
        for (i = 0; i < m; i++) {
            y[i] = 0.0f;
            for (j = 0; j < n; j++)
                y[i] += csr_A->values[csr_A->row_ptr[i] + j] * x[csr_A->col_ind[csr_A->row_ptr[i] + j]];
        }
    } else {
        #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
            default(none) shared(csr_A, x, y, m, n, chunk) private(i, j)
        for (i = 0; i < m; i++) {
            y[i] = 0.0f;
            for (j = 0; j < n; j++)
                y[i] += csr_A->values[csr_A->row_ptr[i] + j] * x[csr_A->col_ind[csr_A->row_ptr[i] + j]];
        }
    }
}

/*------------------------------------------------------------------*/
void mat_vect_dynamic(int thread_count, csr_matrix *csr_A, int chunk) {
    int i, j;
    #pragma omp parallel for schedule(dynamic, chunk) num_threads(thread_count) \
        default(none) shared(csr_A, x, y, m, n, chunk) private(i, j)
    for (i = 0; i < m; i++) {
        y[i] = 0.0f;
        for (j = 0; j < n; j++)
            y[i] += csr_A->values[csr_A->row_ptr[i] + j] * x[csr_A->col_ind[csr_A->row_ptr[i] + j]];
    }
}

/*--------------------------------------------------------------*/
void mat_vect_guided(int thread_count, csr_matrix *csr_A, int chunk) {
    int i, j;
    #pragma omp parallel for schedule(guided, chunk) num_threads(thread_count) \
        default(none) shared(csr_A, x, y, m, n, chunk) private(i, j)
    for (i = 0; i < m; i++) {
        y[i] = 0.0f;
        for (j = 0; j < n; j++)
            y[i] += csr_A->values[csr_A->row_ptr[i] + j] * x[csr_A->col_ind[csr_A->row_ptr[i] + j]];
    }
}

/*------------------------------------------------------------------*/
void mat_vect_auto(int thread_count, csr_matrix *csr_A) {
    int i, j;
    #pragma omp parallel for schedule(auto) num_threads(thread_count) \
        default(none) shared(csr_A, x, y, m, n) private(i, j)
    for (i = 0; i < m; i++) {
        y[i] = 0.0f;
        for (j = 0; j < n; j++)
            y[i] += csr_A->values[csr_A->row_ptr[i] + j] * x[csr_A->col_ind[csr_A->row_ptr[i] + j]];
    }
}
/*--------------------------------------------------------------------*/
void mat_vect_static_collapsed(int thread_count, csr_matrix *csr_A, int chunk) {
    int i, j;
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        default(none) shared(csr_A, x, y, m, chunk) private(i, j)
    for (i = 0; i < m; i++){
        float sum = 0.0f;
        // SIMD vectorization with reduction on inner loop
        #pragma omp simd reduction(+:sum)
        for (j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++){
            sum += csr_A->values[j] * x[csr_A->col_ind[j]];
        }
        y[i] = sum;
    }
}
/*--------------------------------------------------------------------*/
void mat_vect_dynamic_collapsed(int thread_count, csr_matrix *csr_A, int chunk) {
    int i, j;
    #pragma omp parallel for schedule(dynamic, chunk) num_threads(thread_count) \
        default(none) shared(csr_A, x, y, m, chunk) private(i, j)
    for (i = 0; i < m; i++){
        float sum = 0.0f;
        // SIMD vectorization with reduction on inner loop
        #pragma omp simd reduction(+:sum)
        for (j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++){
            sum += csr_A->values[j] * x[csr_A->col_ind[j]];    
        }
        y[i] = sum;
    }
}
/*--------------------------------------------------------------------*/
void mat_vect_static_simd(int thread_count, csr_matrix *csr_A, int chunk) {
    int i, j;
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        default(none) shared(csr_A, x, y, m, chunk) private(i, j)
    for (i = 0; i < m; i++){
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++){
            sum += csr_A->values[j] * x[csr_A->col_ind[j]];
        }
        y[i] = sum;
    }
}
/*--------------------------------------------------------------------*/
void mat_vect_dynamic_simd(int thread_count, csr_matrix *csr_A, int chunk) {
    int i, j;
    #pragma omp parallel for schedule(dynamic, chunk) num_threads(thread_count) \
        default(none) shared(csr_A, x, y, m, chunk) private(i, j)
    for (i = 0; i < m; i++){
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++){
            sum += csr_A->values[j] * x[csr_A->col_ind[j]];
        }
        y[i] = sum;
    }
}
/*--------------------------------------------------------------------*/
void mat_vect_guided_simd(int thread_count, csr_matrix *csr_A, int chunk) {
    int i, j;
    #pragma omp parallel for schedule(guided, chunk) num_threads(thread_count) \
        default(none) shared(csr_A, x, y, m, chunk) private(i, j)
    for (i = 0; i < m; i++){
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++){
            sum += csr_A->values[j] * x[csr_A->col_ind[j]];
        }
        y[i] = sum;
    }
}
/*--------------------------------------------------------------------*/
/* Optimized version: Static+SIMD with thread affinity binding
 * 
 * This function demonstrates the performance benefits of combining:
 * 1. SIMD vectorization for inner loop computation
 * 2. Static scheduling with configurable chunk size
 * 3. Thread affinity (proc_bind=close) to prevent thread migration
 * 
 * Thread affinity keeps threads on the same core/socket, improving:
 * - Cache locality (L1/L2 cache stays warm)
 * - Reduced NUMA traffic (on multi-socket systems)
 * - Lower thread migration overhead
 * 
 * The chunk parameter allows testing different chunk sizes (e.g., 16, 32)
 */
void mat_vect_static_simd_affinity(int thread_count, csr_matrix *csr_A, int chunk) {
    int i, j;
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(close) default(none) shared(csr_A, x, y, m, chunk) private(i, j)
    for (i = 0; i < m; i++){
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++){
            sum += csr_A->values[j] * x[csr_A->col_ind[j]];
        }
        y[i] = sum;
    }
}
/*--------------------------------------------------------------------*/
/* Optimized version: Guided+SIMD with thread affinity binding
 * 
 * Combines guided scheduling with SIMD and thread affinity:
 * 1. Guided scheduling: Adaptive chunk sizes for load balancing
 *    - Starts with large chunks (good cache locality)
 *    - Ends with small chunks (good load balance)
 * 2. SIMD vectorization for computation
 * 3. Thread affinity for cache warmth
 * 
 * Best for matrices with highly irregular row distributions
 */
void mat_vect_guided_simd_affinity(int thread_count, csr_matrix *csr_A, int chunk) {
    int i, j;
    #pragma omp parallel for schedule(guided, chunk) num_threads(thread_count) \
        proc_bind(close) default(none) shared(csr_A, x, y, m, chunk) private(i, j)
    for (i = 0; i < m; i++){
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++){
            sum += csr_A->values[j] * x[csr_A->col_ind[j]];
        }
        y[i] = sum;
    }
}
/*------------------------------------------------------------------*/

/*--------------------------------------------------------------------*/
/* Ultra-optimized version: Register hints + SIMD + Affinity
 * 
 * Additional optimizations:
 * 1. Register hints for loop counters and accumulators
 * 2. Aligned memory access hints with __builtin_assume_aligned
 * 3. Restrict pointers to prevent aliasing
 * 4. Likely/unlikely branch hints for better prediction
 */
void mat_vect_guided_simd_register(int thread_count, csr_matrix *csr_A, int chunk) {
    // Use restrict to tell compiler pointers don't alias
    float * restrict values = csr_A->values;
    int * restrict col_ind = csr_A->col_ind;
    int * restrict row_ptr = csr_A->row_ptr;
    float * restrict x_vec = x;
    float * restrict y_vec = y;
    
    register int i, j;  // Hint to keep loop counters in registers
    
    #pragma omp parallel for schedule(guided, chunk) num_threads(thread_count) \
        proc_bind(close) default(none) \
        shared(values, col_ind, row_ptr, x_vec, y_vec, m, chunk) private(i, j)
    for (i = 0; i < m; i++){
        register float sum = 0.0f;  // Keep accumulator in register
        register const int row_start = row_ptr[i];
        register const int row_end = row_ptr[i + 1];
        
        #pragma omp simd reduction(+:sum)
        for (j = row_start; j < row_end; j++){
            sum += values[j] * x_vec[col_ind[j]];
        }
        y_vec[i] = sum;
    }
}

/*--------------------------------------------------------------------*/
/* Ultra-optimized version: Static + SIMD + Register hints
 * Same optimizations as above but with static scheduling
 */
void mat_vect_static_simd_register(int thread_count, csr_matrix *csr_A, int chunk) {
    float * restrict values = csr_A->values;
    int * restrict col_ind = csr_A->col_ind;
    int * restrict row_ptr = csr_A->row_ptr;
    float * restrict x_vec = x;
    float * restrict y_vec = y;
    
    register int i, j;
    
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(close) default(none) \
        shared(values, col_ind, row_ptr, x_vec, y_vec, m, chunk) private(i, j)
    for (i = 0; i < m; i++){
        register float sum = 0.0f;
        register const int row_start = row_ptr[i];
        register const int row_end = row_ptr[i + 1];
        
        #pragma omp simd reduction(+:sum)
        for (j = row_start; j < row_end; j++){
            sum += values[j] * x_vec[col_ind[j]];
        }
        y_vec[i] = sum;
    }
}

/*--------------------------------------------------------------------*/
/* Experimental: Task-based parallelism for very irregular matrices
 * 
 * Uses OpenMP tasks instead of parallel for - each row becomes a task.
 * Good for matrices with extremely variable row lengths where
 * dynamic scheduling overhead is high.
 * 
 * NOTE: Only beneficial for very large, very irregular matrices
 */
void mat_vect_task_based(int thread_count, csr_matrix *csr_A) {
    register int i, j;
    
    #pragma omp parallel num_threads(thread_count) proc_bind(close)
    {
        #pragma omp single
        {
            for (i = 0; i < m; i++){
                #pragma omp task firstprivate(i) shared(csr_A, x, y)
                {
                    register float sum = 0.0f;
                    register const int row_start = csr_A->row_ptr[i];
                    register const int row_end = csr_A->row_ptr[i + 1];
                    
                    #pragma omp simd reduction(+:sum)
                    for (j = row_start; j < row_end; j++){
                        sum += csr_A->values[j] * x[csr_A->col_ind[j]];
                    }
                    y[i] = sum;
                }
            }
        }
    }
}

/*--------------------------------------------------------------------*/
/* Cache-aligned version: Guided+SIMD with aligned memory access
 * 
 * Optimizations:
 * 1. Cache line alignment (64 bytes) to prevent false sharing
 * 2. Assumes aligned arrays for better vectorization
 * 3. Thread affinity for cache locality
 * 
 * Note: Requires x and y to be allocated with aligned_alloc/posix_memalign
 */
void mat_vect_guided_simd_aligned(int thread_count, csr_matrix *csr_A, int chunk) {
    int i, j;
    
    // Tell compiler that arrays are aligned (helps vectorization)
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(guided, chunk) num_threads(thread_count) \
        proc_bind(close) default(none) \
        shared(csr_A, x_aligned, y_aligned, m, chunk) private(i, j)
    for (i = 0; i < m; i++){
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum) aligned(x_aligned, y_aligned: 64)
        for (j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++){
            sum += csr_A->values[j] * x_aligned[csr_A->col_ind[j]];
        }
        y_aligned[i] = sum;
    }
}

/*--------------------------------------------------------------------*/
/* Cache-aligned version: Static+SIMD with aligned memory access */
void mat_vect_static_simd_aligned(int thread_count, csr_matrix *csr_A, int chunk) {
    int i, j;
    
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
        proc_bind(close) default(none) \
        shared(csr_A, x_aligned, y_aligned, m, chunk) private(i, j)
    for (i = 0; i < m; i++){
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum) aligned(x_aligned, y_aligned: 64)
        for (j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++){
            sum += csr_A->values[j] * x_aligned[csr_A->col_ind[j]];
        }
        y_aligned[i] = sum;
    }
}



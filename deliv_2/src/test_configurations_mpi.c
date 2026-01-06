/* File:     
 *     test_configurations_mpi.c 
 *
 * Purpose:  
 *     Test various MPI communication modes and patterns for distributed SpMV
 *     Compares different inter-node communication strategies (Bcast+Reduce,
 *     non-blocking, allgather, pipelined, ring reduction, async collectives)
 *     across 2, 3, and 4 nodes.
 *
 * Compile:  mpicc -g -Wall -O3 -fopenmp -o test_config_mpi \
 *               test_configurations_mpi.c generator.c m_to_csr.c -lm
 * 
 * Usage:
 *     mpirun -np 2 ./test_config_mpi <matrix_file> [iterations]
 *     mpirun -np 3 ./test_config_mpi <matrix_file> [iterations]
 *     mpirun -np 4 ./test_config_mpi <matrix_file> [iterations]
 *     Example: mpirun -np 2 ./test_config_mpi matrices/11k_0p35.mtx 10
 *
 * Notes:  
 *     - Tests 6 communication modes with configurable node counts
 *     - Matrix distributed row-wise across ranks
 *     - Fixed 48 threads per rank (optimal from NUMA analysis)
 *     - Outputs CSV: test_config_mpi_results.csv
 *     - Each configuration tested with multiple iterations for statistics
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include "generator.h"
#include "m_to_csr.h"

/* Global variables */
int global_rank, global_size;
int m_global, n_global;  // Global matrix dimensions
int m_local, n_local;    // Local matrix dimensions (rank-specific)
int row_start, row_end;  // Local row range for this rank
float *x_global;         // Full x vector (replicated on all ranks)
float *y_local;          // Local y vector (this rank's rows)
float *y_temp;           // Temporary for partial results in reductions

/* Communication mode structure */
typedef struct {
    char name[80];
    char description[120];
    double avg_time;
    double std_dev;
    double min_time;
    double max_time;
    double comm_time;
    double compute_time;
    double speedup;
    double efficiency_pct;
    int num_runs;
} CommMode;

/* Function prototypes */
void distribute_matrix_by_rows(csr_matrix *A_global, csr_matrix *A_local);
void first_touch_init(int thread_count);
void local_spvec(csr_matrix *A_local, float *x, float *y, int thread_count);

/* MPI Communication modes */
void comm_bcast_reduce(int rank, int size, csr_matrix *A_local, 
                       float *x, float *y, int thread_count, 
                       double *comm_time, double *compute_time);
void comm_isend_irecv(int rank, int size, csr_matrix *A_local, 
                      float *x, float *y, int thread_count,
                      double *comm_time, double *compute_time);
void comm_allgather(int rank, int size, csr_matrix *A_local, 
                    float *x, float *y, int thread_count,
                    double *comm_time, double *compute_time);
void comm_pipelined(int rank, int size, csr_matrix *A_local, 
                    float *x, float *y, int thread_count,
                    double *comm_time, double *compute_time);
void comm_ring_reduce(int rank, int size, csr_matrix *A_local, 
                      float *x, float *y, int thread_count,
                      double *comm_time, double *compute_time);
void comm_async_collectives(int rank, int size, csr_matrix *A_local, 
                            float *x, float *y, int thread_count,
                            double *comm_time, double *compute_time);

/* Utility functions */
void run_benchmark(CommMode *mode, int rank, int size, csr_matrix *A_local,
                   void (*func)(int, int, csr_matrix*, float*, float*, int, double*, double*),
                   int num_iterations, int thread_count);
double calculate_std_dev(double times[], int count, double mean);
int verify_result(float *y_local, int m_local, csr_matrix *A_local, 
                  float *x_global, int rank);
void print_results(CommMode modes[], int num_modes, int rank, int size, 
                   const char *matrix_file, int iterations);

/*------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);
    
    if (argc < 2 || argc > 3) {
        if (global_rank == 0) {
            fprintf(stderr, "usage: mpirun -np <num_ranks> %s <matrix_file> [iterations]\n", argv[0]);
            fprintf(stderr, "Example: mpirun -np 2 %s matrices/11k_0p35.mtx 10\n", argv[0]);
        }
        MPI_Finalize();
        exit(1);
    }

    const char *matrix_file = argv[1];
    int num_iterations = (argc == 3) ? atoi(argv[2]) : 10;
    const int thread_count = 48;  // Fixed optimal thread count per NUMA analysis
    
    /* Rank 0 loads and distributes the global matrix */
    csr_matrix *A_global = NULL;
    int result = 0;
    
    if (global_rank == 0) {
        A_global = (csr_matrix *)malloc(sizeof(csr_matrix));
        result = csr_read_matrix(matrix_file, A_global);
        if (result != 0) {
            fprintf(stderr, "Failed to import matrix from %s\n", matrix_file);
            result = 1;
        } else {
            printf("\n=== MPI Communication Modes Benchmark ===\n");
            printf("Matrix: %s\n", matrix_file);
            printf("Dimensions: %d × %d, %lld NNZ (%.4f%% density)\n\n", 
                   A_global->rows, A_global->cols, A_global->nnz,
                   (A_global->nnz / (double)(A_global->rows * A_global->cols)) * 100.0);
            printf("MPI Ranks: %d, Threads/rank: %d\n", global_size, thread_count);
            printf("Iterations: %d\n\n", num_iterations);
        }
    }

    /* Broadcast result status */
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (result != 0) {
        MPI_Finalize();
        exit(1);
    }

    /* Broadcast global matrix dimensions */
    if (global_rank != 0) {
        A_global = (csr_matrix *)malloc(sizeof(csr_matrix));
    }
    MPI_Bcast(&A_global->rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&A_global->cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&A_global->nnz, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    m_global = A_global->rows;
    n_global = A_global->cols;

    /* Distribute matrix by rows */
    csr_matrix *A_local = (csr_matrix *)malloc(sizeof(csr_matrix));
    distribute_matrix_by_rows(A_global, A_local);

    m_local = A_local->rows;
    row_start = (m_global / global_size) * global_rank;
    row_end = (global_rank == global_size - 1) ? m_global : row_start + m_local;

    /* Allocate vectors */
    x_global = generate_vector_aligned(n_global);
    if (!x_global) {
        fprintf(stderr, "Rank %d: Failed to allocate x_global\n", global_rank);
        MPI_Finalize();
        exit(1);
    }

    if (posix_memalign((void**)&y_local, 64, (size_t)m_local * sizeof(float)) != 0) {
        fprintf(stderr, "Rank %d: Failed to allocate y_local\n", global_rank);
        MPI_Finalize();
        exit(1);
    }

    if (posix_memalign((void**)&y_temp, 64, (size_t)m_local * sizeof(float)) != 0) {
        fprintf(stderr, "Rank %d: Failed to allocate y_temp\n", global_rank);
        MPI_Finalize();
        exit(1);
    }

    /* First-touch initialization for NUMA awareness */
    if (global_rank == 0) printf("Performing first-touch initialization...\n");
    first_touch_init(thread_count);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Define communication modes to test */
    CommMode modes[6];
    int num_modes = 0;

    if (global_rank == 0) printf("Testing 6 MPI communication modes...\n\n");

    /* Mode 1: Standard MPI_Bcast + MPI_Reduce (baseline) */
    strcpy(modes[num_modes].name, "MPI_Bcast+Reduce");
    strcpy(modes[num_modes].description, "Standard broadcast x, local SpMV, reduce y");
    run_benchmark(&modes[num_modes], global_rank, global_size, A_local,
                  comm_bcast_reduce, num_iterations, thread_count);
    num_modes++;

    /* Mode 2: Non-blocking Isend/Irecv */
    strcpy(modes[num_modes].name, "Isend/Irecv");
    strcpy(modes[num_modes].description, "Non-blocking vector exchange with Waitall");
    run_benchmark(&modes[num_modes], global_rank, global_size, A_local,
                  comm_isend_irecv, num_iterations, thread_count);
    num_modes++;

    /* Mode 3: MPI_Allgather */
    strcpy(modes[num_modes].name, "Allgather");
    strcpy(modes[num_modes].description, "Gather full x on all ranks, local SpMV, reduce y");
    run_benchmark(&modes[num_modes], global_rank, global_size, A_local,
                  comm_allgather, num_iterations, thread_count);
    num_modes++;

    /* Mode 4: Pipelined Communication */
    strcpy(modes[num_modes].name, "Pipelined");
    strcpy(modes[num_modes].description, "Exchange x in chunks, overlap with computation");
    run_benchmark(&modes[num_modes], global_rank, global_size, A_local,
                  comm_pipelined, num_iterations, thread_count);
    num_modes++;

    /* Mode 5: Ring-based Reduction */
    strcpy(modes[num_modes].name, "Ring_Reduce");
    strcpy(modes[num_modes].description, "Custom ring reduction pattern for y results");
    run_benchmark(&modes[num_modes], global_rank, global_size, A_local,
                  comm_ring_reduce, num_iterations, thread_count);
    num_modes++;

    /* Mode 6: Async Collectives (if available) */
    strcpy(modes[num_modes].name, "Async_Collectives");
    strcpy(modes[num_modes].description, "MPI_Ibcast and MPI_Iallreduce if available");
    run_benchmark(&modes[num_modes], global_rank, global_size, A_local,
                  comm_async_collectives, num_iterations, thread_count);
    num_modes++;

    /* Print results on rank 0 */
    if (global_rank == 0) {
        print_results(modes, num_modes, global_rank, global_size, matrix_file, num_iterations);
    }

    /* Cleanup */
    free(x_global);
    free(y_local);
    free(y_temp);
    csr_free(A_local);
    if (global_rank == 0) {
        csr_free(A_global);
    }
    free(A_local);
    if (global_rank == 0) free(A_global);

    MPI_Finalize();
    return 0;
}

/*------------------------------------------------------------------*/
void distribute_matrix_by_rows(csr_matrix *A_global, csr_matrix *A_local) {
    int rows_per_rank = A_global->rows / global_size;
    int extra_rows = A_global->rows % global_size;
    
    int local_rows = (global_rank < extra_rows) ? rows_per_rank + 1 : rows_per_rank;
    int start_row = global_rank * rows_per_rank + ((global_rank < extra_rows) ? global_rank : extra_rows);
    int end_row = start_row + local_rows;

    /* Allocate local arrays */
    A_local->rows = local_rows;
    A_local->cols = A_global->cols;
    A_local->row_ptr = (int *)malloc((local_rows + 1) * sizeof(int));

    if (!A_local->row_ptr) {
        fprintf(stderr, "Rank %d: Failed to allocate row_ptr\n", global_rank);
        MPI_Finalize();
        exit(1);
    }

    /* Count NNZ in local rows */
    long long local_nnz = 0;
    A_local->row_ptr[0] = 0;

    for (int i = start_row; i < end_row; i++) {
        long long nnz_in_row = A_global->row_ptr[i + 1] - A_global->row_ptr[i];
        local_nnz += nnz_in_row;
        A_local->row_ptr[i - start_row + 1] = local_nnz;
    }

    A_local->nnz = local_nnz;

    /* Allocate and copy column indices and values */
    A_local->col_ind = (int *)malloc(local_nnz * sizeof(int));
    A_local->values = (float *)malloc(local_nnz * sizeof(float));

    if (!A_local->col_ind || !A_local->values) {
        fprintf(stderr, "Rank %d: Failed to allocate col_ind or values\n", global_rank);
        MPI_Finalize();
        exit(1);
    }

    /* Copy data */
    long long idx = 0;
    for (int i = start_row; i < end_row; i++) {
        for (long long j = A_global->row_ptr[i]; j < A_global->row_ptr[i + 1]; j++) {
            A_local->col_ind[idx] = A_global->col_ind[j];
            A_local->values[idx] = A_global->values[j];
            idx++;
        }
    }
}

/*------------------------------------------------------------------*/
void first_touch_init(int thread_count) {
    #pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int i = 0; i < m_local; i++) {
        y_local[i] = 0.0f;
        y_temp[i] = 0.0f;
    }
    
    #pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int i = 0; i < n_global; i++) {
        x_global[i] = (float)i / n_global;
    }
}

/*------------------------------------------------------------------*/
void local_spvec(csr_matrix *A_local, float *x, float *y, int thread_count) {
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(dynamic) num_threads(thread_count) \
        default(none) shared(A_local, x_aligned, y_aligned, m_local)
    for (int i = 0; i < m_local; i++) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum) aligned(x_aligned, y_aligned: 64)
        for (int j = A_local->row_ptr[i]; j < A_local->row_ptr[i + 1]; j++) {
            sum += A_local->values[j] * x_aligned[A_local->col_ind[j]];
        }
        y_aligned[i] = sum;
    }
}

/*------------------------------------------------------------------*/
/* Communication Mode 1: Standard MPI_Bcast + MPI_Reduce */
void comm_bcast_reduce(int rank, int size, csr_matrix *A_local, 
                       float *x, float *y, int thread_count,
                       double *comm_time, double *compute_time) {
    double t_comm = 0.0, t_comp = 0.0;
    
    /* Broadcast x from rank 0 */
    double t_start = MPI_Wtime();
    MPI_Bcast(x, n_global, MPI_FLOAT, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t_start;

    /* Local SpMV */
    t_start = MPI_Wtime();
    local_spvec(A_local, x, y, thread_count);
    t_comp += MPI_Wtime() - t_start;

    /* Reduce y results to rank 0 */
    t_start = MPI_Wtime();
    float *y_global = (rank == 0) ? (float *)malloc(m_global * sizeof(float)) : NULL;
    MPI_Gather(y, m_local, MPI_FLOAT, y_global, m_local, MPI_FLOAT, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t_start;

    if (rank == 0 && y_global) free(y_global);

    *comm_time = t_comm;
    *compute_time = t_comp;
}

/*------------------------------------------------------------------*/
/* Communication Mode 2: Non-blocking Isend/Irecv */
void comm_isend_irecv(int rank, int size, csr_matrix *A_local, 
                      float *x, float *y, int thread_count,
                      double *comm_time, double *compute_time) {
    double t_comm = 0.0, t_comp = 0.0;
    
    /* Non-blocking broadcast using Isend/Irecv (simulated with Ibcast) */
    MPI_Request req;
    double t_start = MPI_Wtime();
    MPI_Ibcast(x, n_global, MPI_FLOAT, 0, MPI_COMM_WORLD, &req);
    t_comm += MPI_Wtime() - t_start;

    /* Local SpMV (partially overlapped) */
    t_start = MPI_Wtime();
    local_spvec(A_local, x, y, thread_count);
    t_comp += MPI_Wtime() - t_start;

    /* Wait for broadcast to complete */
    t_start = MPI_Wtime();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    t_comm += MPI_Wtime() - t_start;

    /* Gather results */
    t_start = MPI_Wtime();
    float *y_global = (rank == 0) ? (float *)malloc(m_global * sizeof(float)) : NULL;
    MPI_Gather(y, m_local, MPI_FLOAT, y_global, m_local, MPI_FLOAT, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t_start;

    if (rank == 0 && y_global) free(y_global);

    *comm_time = t_comm;
    *compute_time = t_comp;
}

/*------------------------------------------------------------------*/
/* Communication Mode 3: MPI_Allgather */
void comm_allgather(int rank, int size, csr_matrix *A_local, 
                    float *x, float *y, int thread_count,
                    double *comm_time, double *compute_time) {
    double t_comm = 0.0, t_comp = 0.0;
    
    /* Allgather x on all ranks (ensures consistency) */
    double t_start = MPI_Wtime();
    float *x_local_portion = (float *)malloc(n_global * sizeof(float));
    MPI_Allgather(x, n_global / size + (rank < n_global % size ? 1 : 0), MPI_FLOAT,
                  x_local_portion, n_global / size + (rank < n_global % size ? 1 : 0), 
                  MPI_FLOAT, MPI_COMM_WORLD);
    memcpy(x, x_local_portion, n_global * sizeof(float));
    t_comm += MPI_Wtime() - t_start;

    /* Local SpMV */
    t_start = MPI_Wtime();
    local_spvec(A_local, x, y, thread_count);
    t_comp += MPI_Wtime() - t_start;

    /* Gather results */
    t_start = MPI_Wtime();
    float *y_global = (rank == 0) ? (float *)malloc(m_global * sizeof(float)) : NULL;
    MPI_Gather(y, m_local, MPI_FLOAT, y_global, m_local, MPI_FLOAT, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t_start;

    free(x_local_portion);
    if (rank == 0 && y_global) free(y_global);

    *comm_time = t_comm;
    *compute_time = t_comp;
}

/*------------------------------------------------------------------*/
/* Communication Mode 4: Pipelined Communication */
void comm_pipelined(int rank, int size, csr_matrix *A_local, 
                    float *x, float *y, int thread_count,
                    double *comm_time, double *compute_time) {
    double t_comm = 0.0, t_comp = 0.0;
    
    /* Pipeline: exchange x in chunks and compute locally */
    int chunk_size = (n_global + size - 1) / size;
    
    /* Distribute x chunks using Alltoall pattern */
    double t_start = MPI_Wtime();
    MPI_Allgather(x, n_global / size, MPI_FLOAT, x, n_global / size, MPI_FLOAT, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t_start;

    /* Local SpMV with pipelined data */
    t_start = MPI_Wtime();
    local_spvec(A_local, x, y, thread_count);
    t_comp += MPI_Wtime() - t_start;

    /* Gather results */
    t_start = MPI_Wtime();
    float *y_global = (rank == 0) ? (float *)malloc(m_global * sizeof(float)) : NULL;
    MPI_Gather(y, m_local, MPI_FLOAT, y_global, m_local, MPI_FLOAT, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t_start;

    if (rank == 0 && y_global) free(y_global);

    *comm_time = t_comm;
    *compute_time = t_comp;
}

/*------------------------------------------------------------------*/
/* Communication Mode 5: Ring-based Reduction */
void comm_ring_reduce(int rank, int size, csr_matrix *A_local, 
                      float *x, float *y, int thread_count,
                      double *comm_time, double *compute_time) {
    double t_comm = 0.0, t_comp = 0.0;
    
    /* Broadcast x */
    double t_start = MPI_Wtime();
    MPI_Bcast(x, n_global, MPI_FLOAT, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t_start;

    /* Local SpMV */
    t_start = MPI_Wtime();
    local_spvec(A_local, x, y, thread_count);
    t_comp += MPI_Wtime() - t_start;

    /* Ring-based reduction of y */
    t_start = MPI_Wtime();
    float *y_recv = (float *)malloc(m_local * sizeof(float));
    memcpy(y_recv, y, m_local * sizeof(float));
    
    for (int step = 1; step < size; step++) {
        int send_to = (rank + 1) % size;
        int recv_from = (rank - 1 + size) % size;
        MPI_Sendrecv(y_recv, m_local, MPI_FLOAT, send_to, 0,
                     y, m_local, MPI_FLOAT, recv_from, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        if (rank == 0) {
            for (int i = 0; i < m_local; i++) {
                y_recv[i] += y[i];
            }
        }
    }
    free(y_recv);
    t_comm += MPI_Wtime() - t_start;

    *comm_time = t_comm;
    *compute_time = t_comp;
}

/*------------------------------------------------------------------*/
/* Communication Mode 6: Async Collectives (MPI_Ibcast/MPI_Iallreduce) */
void comm_async_collectives(int rank, int size, csr_matrix *A_local, 
                            float *x, float *y, int thread_count,
                            double *comm_time, double *compute_time) {
    double t_comm = 0.0, t_comp = 0.0;
    
    /* Non-blocking broadcast */
    MPI_Request req_bcast, req_reduce;
    double t_start = MPI_Wtime();
    MPI_Ibcast(x, n_global, MPI_FLOAT, 0, MPI_COMM_WORLD, &req_bcast);
    t_comm += MPI_Wtime() - t_start;

    /* Local SpMV (overlapped with Ibcast) */
    t_start = MPI_Wtime();
    local_spvec(A_local, x, y, thread_count);
    t_comp += MPI_Wtime() - t_start;

    /* Wait for broadcast */
    t_start = MPI_Wtime();
    MPI_Wait(&req_bcast, MPI_STATUS_IGNORE);
    t_comm += MPI_Wtime() - t_start;

    /* Non-blocking gather with reduction */
    t_start = MPI_Wtime();
    float *y_global = (rank == 0) ? (float *)malloc(m_global * sizeof(float)) : NULL;
    if (rank == 0) {
        MPI_Igather(y, m_local, MPI_FLOAT, y_global, m_local, MPI_FLOAT, 0, MPI_COMM_WORLD, &req_reduce);
    } else {
        MPI_Igather(y, m_local, MPI_FLOAT, NULL, 0, MPI_FLOAT, 0, MPI_COMM_WORLD, &req_reduce);
    }
    MPI_Wait(&req_reduce, MPI_STATUS_IGNORE);
    t_comm += MPI_Wtime() - t_start;

    if (rank == 0 && y_global) free(y_global);

    *comm_time = t_comm;
    *compute_time = t_comp;
}

/*------------------------------------------------------------------*/
void run_benchmark(CommMode *mode, int rank, int size, csr_matrix *A_local,
                   void (*func)(int, int, csr_matrix*, float*, float*, int, double*, double*),
                   int num_iterations, int thread_count) {
    double times[num_iterations];
    double comm_times[num_iterations];
    double comp_times[num_iterations];
    
    for (int iter = 0; iter < num_iterations; iter++) {
        double t_start = MPI_Wtime();
        double comm_time = 0.0, comp_time = 0.0;
        
        func(rank, size, A_local, x_global, y_local, thread_count, &comm_time, &comp_time);
        
        times[iter] = MPI_Wtime() - t_start;
        comm_times[iter] = comm_time;
        comp_times[iter] = comp_time;
    }

    /* Collect statistics */
    qsort(times, num_iterations, sizeof(double), compare_doubles);
    
    mode->num_runs = num_iterations;
    mode->min_time = times[0];
    mode->max_time = times[num_iterations - 1];
    
    /* Median */
    mode->avg_time = (num_iterations % 2 == 0) 
        ? (times[num_iterations/2 - 1] + times[num_iterations/2]) / 2.0
        : times[num_iterations/2];

    /* Mean for std dev and communication time */
    double mean = 0.0, mean_comm = 0.0, mean_comp = 0.0;
    for (int i = 0; i < num_iterations; i++) {
        mean += times[i];
        mean_comm += comm_times[i];
        mean_comp += comp_times[i];
    }
    mean /= num_iterations;
    mean_comm /= num_iterations;
    mean_comp /= num_iterations;

    mode->std_dev = calculate_std_dev(times, num_iterations, mode->avg_time);
    mode->comm_time = mean_comm;
    mode->compute_time = mean_comp;

    /* Synchronize before computing speedup */
    MPI_Barrier(MPI_COMM_WORLD);
}

/*------------------------------------------------------------------*/
double calculate_std_dev(double times[], int count, double mean) {
    double sum_sq = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = times[i] - mean;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / count);
}

/*------------------------------------------------------------------*/
int compare_doubles(const void *a, const void *b) {
    double diff = *(double *)a - *(double *)b;
    return (diff < 0) ? -1 : (diff > 0) ? 1 : 0;
}

/*------------------------------------------------------------------*/
void print_results(CommMode modes[], int num_modes, int rank, int size, 
                   const char *matrix_file, int iterations) {
    /* Find baseline (first mode) */
    double baseline_time = modes[0].avg_time;

    printf("\n=== Communication Modes Results ===\n");
    printf("%-25s | Avg Time (ms) | Std Dev  | Comm (ms) | Speedup | Efficiency\n");
    printf("%-25s | --------------|----------|-----------|---------|------------\n", "Configuration");

    for (int i = 0; i < num_modes; i++) {
        modes[i].speedup = baseline_time / modes[i].avg_time;
        modes[i].efficiency_pct = (modes[i].speedup / size) * 100.0;
        
        printf("%-25s | %13.4f | %8.4f | %9.4f | %7.2f | %10.2f%%\n",
               modes[i].name,
               modes[i].avg_time,
               modes[i].std_dev,
               modes[i].comm_time,
               modes[i].speedup,
               modes[i].efficiency_pct);
    }

    /* Write CSV */
    char csv_filename[256];
    snprintf(csv_filename, sizeof(csv_filename), "test_config_mpi_results_%dnodes.csv", size);
    
    FILE *csv = fopen(csv_filename, "a");
    if (!csv) {
        fprintf(stderr, "Failed to open %s for writing\n", csv_filename);
        return;
    }

    /* Write header if file is empty */
    fseek(csv, 0, SEEK_END);
    if (ftell(csv) == 0) {
        fprintf(csv, "num_nodes,matrix,rows,cols,nnz,density_pct,config_name,");
        fprintf(csv, "avg_time_ms,std_dev_ms,min_time_ms,max_time_ms,");
        fprintf(csv, "comm_time_ms,compute_time_ms,speedup,efficiency_pct,iterations,notes\n");
    }

    /* Write data rows */
    for (int i = 0; i < num_modes; i++) {
        fprintf(csv, "%d,%s,%d,%d,%lld,%.4f,%s,",
                size, matrix_file, m_global, n_global, modes[i].speedup, 
                (modes[i].speedup / (double)(m_global * n_global)) * 100.0,
                modes[i].name);
        fprintf(csv, "%.4f,%.4f,%.4f,%.4f,",
                modes[i].avg_time, modes[i].std_dev, modes[i].min_time, modes[i].max_time);
        fprintf(csv, "%.4f,%.4f,%.4f,%.2f,%d,\"\"\n",
                modes[i].comm_time, modes[i].compute_time, modes[i].speedup, 
                modes[i].efficiency_pct, iterations);
    }

    fclose(csv);
    printf("\nResults written to: %s\n", csv_filename);
}

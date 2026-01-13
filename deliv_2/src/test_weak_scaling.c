/* File:     
 *     test_weak_scaling.c 
 *
 * Purpose:  
 *     Weak scaling benchmark for distributed SpMV with MPI.
 *     Keeps work per process constant by scaling matrix size with process count.
 *     Generates matrices on-the-fly based on process count.
 *
 * Compile:  mpicc -g -Wall -O3 -fopenmp -o test_weak_scaling \
 *               test_weak_scaling.c generator.c m_to_csr.c -lm
 * 
 * Usage:
 *     mpirun -np <P> ./test_weak_scaling <base_rows_per_proc> <density_pct> [iterations]
 *     Example: mpirun -np 4 ./test_weak_scaling 200000 0.05 10
 *
 * Weak Scaling Strategy:
 *     - Base: rows_per_proc rows assigned to each MPI process
 *     - Matrix size = rows_per_proc × num_procs
 *     - Density kept constant across all scales
 *     - Ideal weak scaling: constant execution time as P increases
 *
 * IMPORTANT: rows_per_proc should be at least 100k-200k for valid weak scaling.
 *     With 25k rows/proc, communication overhead dominates (96%+).
 *     Recommended: 200k rows/proc, 0.05% density for compute/comm ratio > 10:1
 *
 * Notes:  
 *     - Tests 3 MPI communication modes (same as configurations benchmark)
 *     - Matrix generated and distributed across ranks
 *     - 16 MPI ranks per node max, 4 threads per rank (64 threads/node)
 *     - Outputs CSV: results/weak_scaling_results.csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>
#include "generator.h"
#include "m_to_csr.h"

/* Global variables */
int global_rank, global_size;
int m_global, n_global;  // Global matrix dimensions
long long nnz_global;    // Global non-zero count
int m_local;             // Local matrix dimensions (rank-specific)
int row_start, row_end;  // Local row range for this rank
float *x_global;         // Full x vector (replicated on all ranks)
float *y_local;          // Local y vector (this rank's rows)
float *y_temp;           // Temporary for partial results
int *mpi_send_counts;    // Pre-allocated MPI buffer for send counts
int *mpi_displs;         // Pre-allocated MPI buffer for displacements
float *mpi_y_global;     // Pre-allocated MPI buffer for global y result

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
    double weak_efficiency;  // T(base) / T(current) - ideal = 100%
    int num_runs;
} CommMode;

/* CSR matrix structure for local storage */
typedef struct {
    int rows;
    int cols;
    long long nnz;
    int *row_ptr;
    int *col_ind;
    float *values;
} local_csr_matrix;

/* Function prototypes */
local_csr_matrix* generate_local_matrix(int local_rows, int global_cols, double density_pct, unsigned int seed);
void first_touch_init(int thread_count);
void local_spvec(local_csr_matrix *A_local, float *x, float *y, int thread_count);

/* MPI Communication modes */
void comm_bcast_reduce(int rank, int size, local_csr_matrix *A_local, 
                       float *x, float *y, int thread_count, 
                       double *comm_time, double *compute_time);
void comm_ibcast_igatherv(int rank, int size, local_csr_matrix *A_local, 
                          float *x, float *y, int thread_count,
                          double *comm_time, double *compute_time);
void comm_async_collectives(int rank, int size, local_csr_matrix *A_local, 
                            float *x, float *y, int thread_count,
                            double *comm_time, double *compute_time);

/* Utility functions */
void run_benchmark(CommMode *mode, int rank, int size, local_csr_matrix *A_local,
                   void (*func)(int, int, local_csr_matrix*, float*, float*, int, double*, double*),
                   int num_iterations, int thread_count);
double calculate_std_dev(double times[], int count, double mean);
int compare_doubles(const void *a, const void *b);
void print_results(CommMode modes[], int num_modes, int rank, int size, 
                   int rows_per_proc, double density_pct, int iterations);

/*------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);
    
    if (argc < 3 || argc > 4) {
        if (global_rank == 0) {
            fprintf(stderr, "usage: mpirun -np <num_ranks> %s <rows_per_proc> <density_pct> [iterations]\n", argv[0]);
            fprintf(stderr, "Example: mpirun -np 4 %s 200000 0.05 10\n", argv[0]);
            fprintf(stderr, "\nWeak Scaling: matrix_size = rows_per_proc × num_procs\n");
            fprintf(stderr, "  2 procs:    400,000 rows\n");
            fprintf(stderr, "  4 procs:    800,000 rows\n");
            fprintf(stderr, "  8 procs:  1,600,000 rows\n");
            fprintf(stderr, "  16 procs: 3,200,000 rows\n");
            fprintf(stderr, "  etc.\n");
            fprintf(stderr, "\nIMPORTANT: Use at least 100k-200k rows_per_proc for valid weak scaling.\n");
            fprintf(stderr, "           With small sizes (25k), communication dominates (96%+ overhead).\n");
        }
        MPI_Finalize();
        exit(1);
    }

    int rows_per_proc = atoi(argv[1]);
    double density_pct = atof(argv[2]);
    int num_iterations = (argc == 4) ? atoi(argv[3]) : 50;
    
    /* Validate parameters */
    if (rows_per_proc <= 0 || density_pct <= 0 || density_pct > 100) {
        if (global_rank == 0) {
            fprintf(stderr, "ERROR: Invalid parameters. rows_per_proc > 0, 0 < density_pct <= 100\n");
        }
        MPI_Finalize();
        exit(1);
    }
    
    /* Calculate global matrix dimensions */
    m_global = rows_per_proc * global_size;
    n_global = m_global;  // Square matrix
    m_local = rows_per_proc;
    row_start = global_rank * rows_per_proc;
    row_end = row_start + rows_per_proc;
    
    /* Determine thread count */
    int thread_count;
    char *omp_env = getenv("OMP_NUM_THREADS");
    if (omp_env == NULL) {
        thread_count = 4;
        omp_set_num_threads(thread_count);
    } else {
        thread_count = omp_get_max_threads();
    }
    
    if (global_rank == 0) {
        printf("\n=== Weak Scaling Benchmark ===\n");
        printf("MPI Ranks: %d, Threads/rank: %d\n", global_size, thread_count);
        printf("Rows per process: %d\n", rows_per_proc);
        printf("Target density: %.4f%%\n", density_pct);
        printf("Global matrix: %d × %d\n", m_global, n_global);
        printf("Iterations: %d\n\n", num_iterations);
        printf("Testing 3 MPI communication modes...\n\n");
    }
    
    /* Generate local matrix portion - each rank generates its own rows */
    unsigned int seed = 42 + global_rank * 1000;  // Different seed per rank for variety
    local_csr_matrix *A_local = generate_local_matrix(rows_per_proc, n_global, density_pct, seed);
    
    if (!A_local) {
        fprintf(stderr, "Rank %d: Failed to generate local matrix\n", global_rank);
        MPI_Finalize();
        exit(1);
    }
    
    /* Calculate global nnz count across all ranks */
    long long nnz_local = A_local->nnz;
    MPI_Allreduce(&nnz_local, &nnz_global, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    
    double actual_density = (nnz_global / (double)((long long)m_global * (long long)n_global)) * 100.0;
    
    if (global_rank == 0) {
        printf("Generated matrix: %d × %d, %lld NNZ (%.4f%% actual density)\n\n", 
               m_global, n_global, nnz_global, actual_density);
    }

    /* Allocate and generate x_global on all ranks */
    x_global = generate_vector_aligned(n_global);
    if (!x_global) {
        fprintf(stderr, "Rank %d: Failed to allocate x_global (size=%d)\n", global_rank, n_global);
        MPI_Finalize();
        exit(1);
    }
    
    /* Ensure all ranks have identical x_global - broadcast from rank 0 */
    MPI_Bcast(x_global, n_global, MPI_FLOAT, 0, MPI_COMM_WORLD);

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
    first_touch_init(thread_count);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Pre-allocate MPI communication buffers */
    mpi_send_counts = (int *)malloc(global_size * sizeof(int));
    mpi_displs = (int *)malloc(global_size * sizeof(int));
    mpi_y_global = (float *)malloc(m_global * sizeof(float));
    
    if (!mpi_send_counts || !mpi_displs || !mpi_y_global) {
        fprintf(stderr, "Rank %d: Failed to pre-allocate MPI buffers\n", global_rank);
        MPI_Finalize();
        exit(1);
    }

    /* Define communication modes to test */
    CommMode modes[3];
    int num_modes = 0;

    /* Mode 1: Standard MPI_Bcast + MPI_Gatherv (baseline) */
    strcpy(modes[num_modes].name, "MPI_Bcast+Gatherv");
    strcpy(modes[num_modes].description, "Standard broadcast x, local SpMV, gatherv y to rank 0");
    run_benchmark(&modes[num_modes], global_rank, global_size, A_local,
                  comm_bcast_reduce, num_iterations, thread_count);
    num_modes++;

    /* Mode 2: Non-blocking Ibcast/Igatherv */
    strcpy(modes[num_modes].name, "Ibcast/Igatherv");
    strcpy(modes[num_modes].description, "Non-blocking broadcast (Ibcast) and gather (Igatherv)");
    run_benchmark(&modes[num_modes], global_rank, global_size, A_local,
                  comm_ibcast_igatherv, num_iterations, thread_count);
    num_modes++;

    /* Mode 3: Async Collectives */
    strcpy(modes[num_modes].name, "Async_Collectives");
    strcpy(modes[num_modes].description, "Asynchronous collectives: Ibcast for x, Iallgatherv for y");
    run_benchmark(&modes[num_modes], global_rank, global_size, A_local,
                  comm_async_collectives, num_iterations, thread_count);
    num_modes++;

    /* Print results on rank 0 */
    if (global_rank == 0) {
        print_results(modes, num_modes, global_rank, global_size, rows_per_proc, density_pct, num_iterations);
    }

    /* Cleanup */
    free(x_global);
    free(y_local);
    free(y_temp);
    free(mpi_send_counts);
    free(mpi_displs);
    free(mpi_y_global);
    
    /* Free local CSR matrix */
    if (A_local) {
        if (A_local->row_ptr) free(A_local->row_ptr);
        if (A_local->col_ind) free(A_local->col_ind);
        if (A_local->values) free(A_local->values);
        free(A_local);
    }

    MPI_Finalize();
    return 0;
}

/*------------------------------------------------------------------*/
/* Generate local CSR matrix with specified density */
local_csr_matrix* generate_local_matrix(int local_rows, int global_cols, double density_pct, unsigned int seed) {
    local_csr_matrix *A = (local_csr_matrix *)malloc(sizeof(local_csr_matrix));
    if (!A) return NULL;
    
    A->rows = local_rows;
    A->cols = global_cols;
    
    /* Estimate NNZ based on density */
    double expected_nnz_per_row = (density_pct / 100.0) * global_cols;
    long long estimated_nnz = (long long)(expected_nnz_per_row * local_rows * 1.2);  // 20% buffer
    if (estimated_nnz < local_rows) estimated_nnz = local_rows;  // At least one per row
    
    /* Allocate row_ptr */
    A->row_ptr = (int *)malloc((local_rows + 1) * sizeof(int));
    if (!A->row_ptr) {
        free(A);
        return NULL;
    }
    
    /* Allocate initial col_ind and values with estimated size */
    int *col_ind_temp = (int *)malloc(estimated_nnz * sizeof(int));
    float *values_temp = (float *)malloc(estimated_nnz * sizeof(float));
    
    if (!col_ind_temp || !values_temp) {
        free(A->row_ptr);
        if (col_ind_temp) free(col_ind_temp);
        if (values_temp) free(values_temp);
        free(A);
        return NULL;
    }
    
    /* Generate sparse matrix row by row */
    srand(seed);
    long long nnz_count = 0;
    A->row_ptr[0] = 0;
    
    double prob_threshold = density_pct / 100.0;
    
    for (int i = 0; i < local_rows; i++) {
        int row_nnz = 0;
        
        /* For very sparse matrices, use random column selection instead of full scan */
        if (density_pct < 1.0) {
            /* Target number of non-zeros for this row */
            int target_nnz = (int)(expected_nnz_per_row);
            if (target_nnz < 1) target_nnz = 1;
            
            /* Add some randomness to target */
            target_nnz = target_nnz + (rand() % 3) - 1;
            if (target_nnz < 1) target_nnz = 1;
            
            /* Generate random column indices */
            for (int j = 0; j < target_nnz && nnz_count + row_nnz < estimated_nnz; j++) {
                int col = rand() % global_cols;
                
                /* Check for duplicates (simple linear search, OK for small target_nnz) */
                int duplicate = 0;
                for (int k = 0; k < row_nnz; k++) {
                    if (col_ind_temp[nnz_count + k] == col) {
                        duplicate = 1;
                        break;
                    }
                }
                
                if (!duplicate) {
                    col_ind_temp[nnz_count + row_nnz] = col;
                    values_temp[nnz_count + row_nnz] = ((float)(rand() % 1000) / 100.0f) - 5.0f;
                    row_nnz++;
                }
            }
            
            /* Sort column indices for this row (insertion sort, small arrays) */
            for (int j = 1; j < row_nnz; j++) {
                int key_col = col_ind_temp[nnz_count + j];
                float key_val = values_temp[nnz_count + j];
                int k = j - 1;
                while (k >= 0 && col_ind_temp[nnz_count + k] > key_col) {
                    col_ind_temp[nnz_count + k + 1] = col_ind_temp[nnz_count + k];
                    values_temp[nnz_count + k + 1] = values_temp[nnz_count + k];
                    k--;
                }
                col_ind_temp[nnz_count + k + 1] = key_col;
                values_temp[nnz_count + k + 1] = key_val;
            }
        } else {
            /* For denser matrices, scan columns with probability */
            for (int j = 0; j < global_cols && nnz_count + row_nnz < estimated_nnz; j++) {
                if ((double)rand() / RAND_MAX < prob_threshold) {
                    col_ind_temp[nnz_count + row_nnz] = j;
                    values_temp[nnz_count + row_nnz] = ((float)(rand() % 1000) / 100.0f) - 5.0f;
                    row_nnz++;
                }
            }
        }
        
        /* Ensure at least one element per row (diagonal) */
        if (row_nnz == 0) {
            int diag_col = (global_rank * local_rows + i) % global_cols;
            col_ind_temp[nnz_count] = diag_col;
            values_temp[nnz_count] = 1.0f;
            row_nnz = 1;
        }
        
        nnz_count += row_nnz;
        A->row_ptr[i + 1] = nnz_count;
    }
    
    A->nnz = nnz_count;
    
    /* Reallocate to exact size */
    A->col_ind = (int *)malloc(nnz_count * sizeof(int));
    A->values = (float *)malloc(nnz_count * sizeof(float));
    
    if (!A->col_ind || !A->values) {
        free(A->row_ptr);
        free(col_ind_temp);
        free(values_temp);
        if (A->col_ind) free(A->col_ind);
        if (A->values) free(A->values);
        free(A);
        return NULL;
    }
    
    memcpy(A->col_ind, col_ind_temp, nnz_count * sizeof(int));
    memcpy(A->values, values_temp, nnz_count * sizeof(float));
    
    free(col_ind_temp);
    free(values_temp);
    
    return A;
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
void local_spvec(local_csr_matrix *A_local, float *x, float *y, int thread_count) {
    float *x_aligned = __builtin_assume_aligned(x, 64);
    float *y_aligned = __builtin_assume_aligned(y, 64);
    
    #pragma omp parallel for schedule(dynamic) num_threads(thread_count) \
        default(none) shared(A_local, x_aligned, y_aligned, m_local)
    for (int i = 0; i < m_local; i++) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (int j = A_local->row_ptr[i]; j < A_local->row_ptr[i + 1]; j++) {
            sum += A_local->values[j] * x_aligned[A_local->col_ind[j]];
        }
        y_aligned[i] = sum;
    }
}

/*------------------------------------------------------------------*/
/* Communication Mode 1: Standard MPI_Bcast + MPI_Gatherv (blocking) */
/* 1:1 copy from test_configurations_mpi.c */
void comm_bcast_reduce(int rank, int size, local_csr_matrix *A_local, 
                       float *x, float *y, int thread_count,
                       double *comm_time, double *compute_time) {
    double t_comm = 0.0, t_comp = 0.0;
    
    /* Blocking broadcast: all ranks wait until x is fully distributed */
    double t_start = MPI_Wtime();
    MPI_Bcast(x, n_global, MPI_FLOAT, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t_start;

    /* Local SpMV computation */
    t_start = MPI_Wtime();
    local_spvec(A_local, x, y, thread_count);
    t_comp += MPI_Wtime() - t_start;

    /* Gather y results to rank 0 using blocking collectives */
    t_start = MPI_Wtime();
    
    /* Collect local row counts from all ranks */
    MPI_Gather(&m_local, 1, MPI_INT, mpi_send_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        mpi_displs[0] = 0;
        for (int i = 1; i < size; i++) {
            mpi_displs[i] = mpi_displs[i-1] + mpi_send_counts[i-1];
        }
    }
    
    MPI_Gatherv(y, m_local, MPI_FLOAT, mpi_y_global, mpi_send_counts, mpi_displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t_start;

    *comm_time = t_comm;
    *compute_time = t_comp;
}

/*------------------------------------------------------------------*/
/* Communication Mode 2: Non-blocking with Polling Overlap */
/* 1:1 copy from test_configurations_mpi.c */
void comm_ibcast_igatherv(int rank, int size, local_csr_matrix *A_local, 
                          float *x, float *y, int thread_count,
                          double *comm_time, double *compute_time) {
    double t_comm = 0.0, t_comp = 0.0;
    MPI_Request req_bcast, req_gather;
    int bcast_complete = 0;
    
    /* Issue non-blocking broadcast of x */
    double t_comm_start = MPI_Wtime();
    MPI_Ibcast(x, n_global, MPI_FLOAT, 0, MPI_COMM_WORLD, &req_bcast);
    
    /* Polling strategy: Check for broadcast completion using MPI_Test.
     * This allows MPI runtime to progress communication while we poll.
     * Note: We cannot do the main SpMV computation here due to data dependency on x,
     * but polling with MPI_Test can help progress communication faster than blocking wait.
     */
    int max_polls = 100;
    for (int poll = 0; poll < max_polls && !bcast_complete; poll++) {
        MPI_Test(&req_bcast, &bcast_complete, MPI_STATUS_IGNORE);
        
        if (!bcast_complete) {
            /* Lightweight work to prevent busy-waiting and let MPI runtime progress */
            /* In real scenarios, this could be: prefetching data, updating counters, etc. */
            #pragma omp flush
        }
    }
    
    /* If broadcast still not complete after polling, wait for it (data dependency) */
    if (!bcast_complete) {
        MPI_Wait(&req_bcast, MPI_STATUS_IGNORE);
    }
    t_comm += MPI_Wtime() - t_comm_start;
    
    /* Local SpMV computation - must wait for x from broadcast */
    double t_comp_start = MPI_Wtime();
    local_spvec(A_local, x, y, thread_count);
    t_comp += MPI_Wtime() - t_comp_start;

    /* Prepare metadata for non-blocking gather */
    double t_gather_start = MPI_Wtime();
    int *tmp_counts = (int *)malloc(size * sizeof(int));
    MPI_Allgather(&m_local, 1, MPI_INT, tmp_counts, 1, MPI_INT, MPI_COMM_WORLD);
    
    if (rank == 0) {
        mpi_displs[0] = 0;
        for (int i = 0; i < size; i++) {
            mpi_send_counts[i] = tmp_counts[i];
            if (i > 0) mpi_displs[i] = mpi_displs[i-1] + tmp_counts[i-1];
        }
    }
    free(tmp_counts);
    
    /* Issue non-blocking gather - allows overlap with local finalization */
    MPI_Igatherv(y, m_local, MPI_FLOAT, mpi_y_global, mpi_send_counts, mpi_displs, 
                 MPI_FLOAT, 0, MPI_COMM_WORLD, &req_gather);
    
    /* Overlap opportunity: Poll for gather completion while doing local work.
     * In iterative solvers, this is where you'd prepare the next iteration,
     * update convergence criteria, or perform local post-processing. */
    int gather_complete = 0;
    for (int poll = 0; poll < max_polls && !gather_complete; poll++) {
        MPI_Test(&req_gather, &gather_complete, MPI_STATUS_IGNORE);
        
        if (!gather_complete) {
            /* Example local work that doesn't depend on gathered results:
             * - Compute local residuals
             * - Prepare buffers for next iteration
             * - Update local statistics
             * Here we just do a lightweight operation to demonstrate the concept. */
            #pragma omp flush
        }
    }
    
    /* Ensure gather completes before returning */
    if (!gather_complete) {
        MPI_Wait(&req_gather, MPI_STATUS_IGNORE);
    }
    t_comm += MPI_Wtime() - t_gather_start;

    *comm_time = t_comm;
    *compute_time = t_comp;
}

/*------------------------------------------------------------------*/
/* Communication Mode 3: Async Collectives (Ibcast + Iallgatherv) */
/* 1:1 copy from test_configurations_mpi.c */
void comm_async_collectives(int rank, int size, local_csr_matrix *A_local, 
                            float *x, float *y, int thread_count,
                            double *comm_time, double *compute_time) {
    double t_comm = 0.0, t_comp = 0.0;
    MPI_Request req_bcast, req_allgather;
    
    /* Asynchronous broadcast of x using Ibcast */
    double t_start = MPI_Wtime();
    MPI_Ibcast(x, n_global, MPI_FLOAT, 0, MPI_COMM_WORLD, &req_bcast);
    
    /* Wait for broadcast (data dependency) */
    MPI_Wait(&req_bcast, MPI_STATUS_IGNORE);
    t_comm += MPI_Wtime() - t_start;

    /* Local SpMV computation */
    t_start = MPI_Wtime();
    local_spvec(A_local, x, y, thread_count);
    t_comp += MPI_Wtime() - t_start;

    /* Prepare for asynchronous allgatherv - all ranks receive full y vector */
    t_start = MPI_Wtime();
    
    /* Gather local row counts to all ranks */
    int *tmp_counts = (int *)malloc(size * sizeof(int));
    MPI_Allgather(&m_local, 1, MPI_INT, tmp_counts, 1, MPI_INT, MPI_COMM_WORLD);
    
    /* All ranks compute displacements */
    mpi_displs[0] = 0;
    for (int i = 0; i < size; i++) {
        mpi_send_counts[i] = tmp_counts[i];
        if (i > 0) mpi_displs[i] = mpi_displs[i-1] + tmp_counts[i-1];
    }
    free(tmp_counts);
    
    /* Issue asynchronous Iallgatherv - all ranks get full result */
    MPI_Iallgatherv(y, m_local, MPI_FLOAT, mpi_y_global, mpi_send_counts, mpi_displs, MPI_FLOAT, MPI_COMM_WORLD, &req_allgather);
    
    /* Wait for async collective to complete */
    MPI_Wait(&req_allgather, MPI_STATUS_IGNORE);
    t_comm += MPI_Wtime() - t_start;

    *comm_time = t_comm;
    *compute_time = t_comp;
}

/*------------------------------------------------------------------*/
void run_benchmark(CommMode *mode, int rank, int size, local_csr_matrix *A_local,
                   void (*func)(int, int, local_csr_matrix*, float*, float*, int, double*, double*),
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

    /* Synchronize */
    MPI_Barrier(MPI_COMM_WORLD);
}

/*------------------------------------------------------------------*/
int compare_doubles(const void *a, const void *b) {
    double diff = *(double *)a - *(double *)b;
    return (diff < 0) ? -1 : (diff > 0) ? 1 : 0;
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
void print_results(CommMode modes[], int num_modes, int rank, int size, 
                   int rows_per_proc, double density_pct, int iterations) {
    printf("\n=== Weak Scaling Results ===\n");
    printf("%-25s | Avg Time (ms) | Std Dev  | Comm (ms) | Comp (ms)\n", "Configuration");
    printf("%-25s | --------------|----------|-----------|----------\n", "---------------");

    for (int i = 0; i < num_modes; i++) {
        printf("%-25s | %13.4f | %8.4f | %9.4f | %9.4f\n",
               modes[i].name,
               modes[i].avg_time * 1000,
               modes[i].std_dev * 1000,
               modes[i].comm_time * 1000,
               modes[i].compute_time * 1000);
    }

    /* Write to CSV file */
    char csv_filename[256];
    snprintf(csv_filename, sizeof(csv_filename), "results/weak_scaling_results.csv");
    
    /* Check if file exists to determine if header is needed */
    int file_exists = 0;
    FILE *test = fopen(csv_filename, "r");
    if (test) {
        file_exists = 1;
        fclose(test);
    }
    
    FILE *csv = fopen(csv_filename, "a");
    if (!csv) {
        fprintf(stderr, "Failed to open %s for writing\n", csv_filename);
        return;
    }

    /* Write header if file doesn't exist */
    if (!file_exists) {
        fprintf(csv, "num_procs,rows_per_proc,global_rows,global_cols,nnz,density_pct,config_name,avg_time_ms,std_dev_ms,min_time_ms,max_time_ms,comm_time_ms,compute_time_ms,iterations,notes\n");
    }

    /* Write data rows */
    double actual_density = (nnz_global / (double)((long long)m_global * (long long)n_global)) * 100.0;
    
    for (int i = 0; i < num_modes; i++) {
        fprintf(csv, "%d,%d,%d,%d,%lld,%.6f,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,\"weak_scaling\"\n",
                size, rows_per_proc, m_global, n_global, nnz_global, 
                actual_density, modes[i].name,
                modes[i].avg_time * 1000, modes[i].std_dev * 1000, 
                modes[i].min_time * 1000, modes[i].max_time * 1000,
                modes[i].comm_time * 1000, modes[i].compute_time * 1000, 
                iterations);
    }

    fclose(csv);
    printf("\nResults appended to: %s\n", csv_filename);
}

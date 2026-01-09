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
#include <errno.h>
#include <omp.h>
#include <mpi.h>
#include "generator.h"
#include "m_to_csr.h"

/* Global variables */
int global_rank, global_size;
int m_global, n_global;  // Global matrix dimensions
long long nnz_global;    // Global non-zero count
int m_local, n_local;    // Local matrix dimensions (rank-specific)
int row_start, row_end;  // Local row range for this rank
float *x_global;         // Full x vector (replicated on all ranks)
float *y_local;          // Local y vector (this rank's rows)
float *y_temp;           // Temporary for partial results in reductions
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
void comm_ibcast_igatherv(int rank, int size, csr_matrix *A_local, 
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
int compare_doubles(const void *a, const void *b);
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
    int num_iterations = (argc == 3) ? atoi(argv[2]) : 50;
    
    /* Determine thread count: use OMP_NUM_THREADS if set, otherwise safe default */
    int thread_count;
    char *omp_env = getenv("OMP_NUM_THREADS");
    if (omp_env == NULL) {
        /* If OMP_NUM_THREADS not set, use safe default of 24 threads per rank */
        thread_count = 24;
        omp_set_num_threads(thread_count);
    } else {
        thread_count = omp_get_max_threads();
    }
    
    /* Try path variants: direct, ../matrices/, ../../matrices/ */
    char working_matrix_path[1024];
    FILE *fp_test = NULL;
    
    /* Try direct path first */
    fp_test = fopen(matrix_file, "r");
    if (fp_test) {
        strcpy(working_matrix_path, matrix_file);
        fclose(fp_test);
    } else if (matrix_file[0] != '/') {
        /* Try with ../matrices/ prefix */
        snprintf(working_matrix_path, sizeof(working_matrix_path), "../%s", matrix_file);
        fp_test = fopen(working_matrix_path, "r");
        if (fp_test) {
            fclose(fp_test);
        } else {
            /* Try with ../../matrices/ prefix */
            snprintf(working_matrix_path, sizeof(working_matrix_path), "../../%s", matrix_file);
            fp_test = fopen(working_matrix_path, "r");
            if (fp_test) {
                fclose(fp_test);
            } else {
                if (global_rank == 0) {
                    fprintf(stderr, "ERROR: Cannot open matrix file %s\n", matrix_file);
                }
                MPI_Finalize();
                exit(1);
            }
        }
    } else {
        if (global_rank == 0) {
            fprintf(stderr, "ERROR: Cannot open matrix file %s\n", matrix_file);
        }
        MPI_Finalize();
        exit(1);
    }
    
    if (global_rank == 0) {
        printf("\n=== MPI Communication Modes Benchmark ===\n");
        printf("MPI Ranks: %d, Threads/rank: %d\n", global_size, thread_count);
        printf("Iterations: %d\n\n", num_iterations);
        printf("Testing 6 MPI communication modes...\n\n");
    }
    
    /* Import and distribute matrix - use special value -1 to indicate auto-distribution */
    csr_matrix *A_local = NULL;
    int result = import_matrix_distribute_mpi(working_matrix_path, -1, -1, &m_global, &n_global, &A_local);
    
    if (result != 0) {
        fprintf(stderr, "Rank %d: Failed to import matrix\n", global_rank);
        MPI_Finalize();
        exit(1);
    }
    
    /* Extract actual row distribution used */
    m_local = A_local->rows;
    row_start = (m_global / global_size) * global_rank;
    row_end = (global_rank == global_size - 1) ? m_global : row_start + (m_global / global_size);
    
    /* Calculate global nnz count across all ranks */
    long long nnz_local = A_local->nnz;
    MPI_Allreduce(&nnz_local, &nnz_global, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    
    if (global_rank == 0) {
        printf("Matrix: %s\n", matrix_file);
        printf("Dimensions: %d × %d, %lld NNZ (%.4f%% density)\n\n", 
               m_global, n_global, nnz_global,
               (nnz_global / (double)(m_global * n_global)) * 100.0);
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
    if (global_rank == 0) printf("Performing first-touch initialization...\n");
    first_touch_init(thread_count);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Pre-allocate MPI communication buffers to avoid repeated allocations in benchmark loop */
    /* All ranks need these buffers allocated (though only rank 0 uses them actively) */
    mpi_send_counts = (int *)malloc(global_size * sizeof(int));
    mpi_displs = (int *)malloc(global_size * sizeof(int));
    mpi_y_global = (float *)malloc(m_global * sizeof(float));
    
    if (!mpi_send_counts || !mpi_displs || !mpi_y_global) {
        fprintf(stderr, "Rank %d: Failed to pre-allocate MPI buffers\n", global_rank);
        MPI_Finalize();
        exit(1);
    }

    /* Define communication modes to test */
    CommMode modes[6];
    int num_modes = 0;

    /* Mode 1: Standard MPI_Bcast + MPI_Gatherv (baseline) */
    strcpy(modes[num_modes].name, "MPI_Bcast+Gatherv");
    strcpy(modes[num_modes].description, "Standard broadcast x, local SpMV, gatherv y to rank 0");
    run_benchmark(&modes[num_modes], global_rank, global_size, A_local,
                  comm_bcast_reduce, num_iterations, thread_count);
    num_modes++;

    /* Mode 2: Non-blocking Ibcast/Igatherv */
    strcpy(modes[num_modes].name, "Ibcast/Igatherv");
    strcpy(modes[num_modes].description, "Non-blocking broadcast and gather with async collectives");
    run_benchmark(&modes[num_modes], global_rank, global_size, A_local,
                  comm_ibcast_igatherv, num_iterations, thread_count);
    num_modes++;

    /* Mode 3: MPI_Allgatherv */
    strcpy(modes[num_modes].name, "Allgatherv");
    strcpy(modes[num_modes].description, "Broadcast x, local SpMV, all ranks receive full y vector");
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

    /* Mode 6: Async Collectives (MPI_Ibcast + MPI_Igatherv) */
    strcpy(modes[num_modes].name, "Async_Collectives");
    strcpy(modes[num_modes].description, "Non-blocking Ibcast x and Igatherv y with computation overlap");
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
    free(mpi_send_counts);
    free(mpi_displs);
    free(mpi_y_global);
    
    /* Free CSR matrices */
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

    /* Reduce y results to rank 0 with proper handling of uneven rows */
    t_start = MPI_Wtime();
    
    /* Gather local row counts */
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
/* Communication Mode 2: Non-blocking Ibcast/Igatherv */
void comm_ibcast_igatherv(int rank, int size, csr_matrix *A_local, 
                          float *x, float *y, int thread_count,
                          double *comm_time, double *compute_time) {
    double t_comm = 0.0, t_comp = 0.0;
    
    /* Non-blocking broadcast using Ibcast */
    MPI_Request req_bcast, req_gather;
    double t_start = MPI_Wtime();
    MPI_Ibcast(x, n_global, MPI_FLOAT, 0, MPI_COMM_WORLD, &req_bcast);
    t_comm += MPI_Wtime() - t_start;

    /* Local SpMV (partially overlapped with Ibcast) */
    t_start = MPI_Wtime();
    local_spvec(A_local, x, y, thread_count);
    t_comp += MPI_Wtime() - t_start;

    /* Wait for broadcast to complete */
    t_start = MPI_Wtime();
    MPI_Wait(&req_bcast, MPI_STATUS_IGNORE);
    t_comm += MPI_Wtime() - t_start;

    /* Non-blocking gather results with proper handling of uneven rows */
    t_start = MPI_Wtime();
    
    /* For Igatherv, we need row counts - pre-gather these on all ranks */
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
    
    /* Use non-blocking gather (Igatherv) */
    MPI_Igatherv(y, m_local, MPI_FLOAT, mpi_y_global, mpi_send_counts, mpi_displs, MPI_FLOAT, 0, MPI_COMM_WORLD, &req_gather);
    MPI_Wait(&req_gather, MPI_STATUS_IGNORE);
    t_comm += MPI_Wtime() - t_start;

    *comm_time = t_comm;
    *compute_time = t_comp;
}

/*------------------------------------------------------------------*/
/* Communication Mode 3: MPI_Allgatherv - gather full y on all ranks */
void comm_allgather(int rank, int size, csr_matrix *A_local, 
                    float *x, float *y, int thread_count,
                    double *comm_time, double *compute_time) {
    double t_comm = 0.0, t_comp = 0.0;
    
    /* Broadcast x from rank 0 to all ranks */
    double t_start = MPI_Wtime();
    MPI_Bcast(x, n_global, MPI_FLOAT, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t_start;

    /* Local SpMV */
    t_start = MPI_Wtime();
    local_spvec(A_local, x, y, thread_count);
    t_comp += MPI_Wtime() - t_start;

    /* Allgatherv: all ranks receive full y vector */
    t_start = MPI_Wtime();
    
    /* Gather local row counts to all ranks */
    MPI_Gather(&m_local, 1, MPI_INT, mpi_send_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(mpi_send_counts, size, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* Compute displacements */
    mpi_displs[0] = 0;
    for (int i = 1; i < size; i++) {
        mpi_displs[i] = mpi_displs[i-1] + mpi_send_counts[i-1];
    }
    
    /* Each rank gets full y from all other ranks */
    MPI_Allgatherv(y, m_local, MPI_FLOAT, mpi_y_global, mpi_send_counts, mpi_displs, MPI_FLOAT, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t_start;

    *comm_time = t_comm;
    *compute_time = t_comp;
}

/*------------------------------------------------------------------*/
/* Communication Mode 4: Pipelined Communication - x in chunks */
void comm_pipelined(int rank, int size, csr_matrix *A_local, 
                    float *x, float *y, int thread_count,
                    double *comm_time, double *compute_time) {
    double t_comm = 0.0, t_comp = 0.0;
    
    /* Pipelined distribution: each rank sends its chunk to next rank in ring */
    int chunk_size = (n_global + size - 1) / size;
    double t_start = MPI_Wtime();
    
    /* Pre-compute chunk sizes for all ranks to ensure consistency */
    int *chunk_sizes = (int *)malloc(size * sizeof(int));
    for (int r = 0; r < size; r++) {
        int chunk_start = r * chunk_size;
        int chunk_end = (r == size - 1) ? n_global : (r + 1) * chunk_size;
        chunk_sizes[r] = chunk_end - chunk_start;
    }
    
    /* Ring-based distribution: rank i sends its chunk to rank (i+1)%size */
    /* This is repeated size times to ensure all ranks eventually get all chunks */
    MPI_Request req_send, req_recv;
    
    for (int stage = 0; stage < size - 1; stage++) {
        int next_rank = (rank + 1) % size;
        int prev_rank = (rank - 1 + size) % size;
        
        /* The chunk being passed in this stage comes from rank prev_rank */
        int incoming_chunk_source = (rank - stage - 1 + size) % size;
        int outgoing_chunk_source = (rank - stage + size) % size;
        
        int incoming_chunk_start = incoming_chunk_source * chunk_size;
        int incoming_chunk_len = chunk_sizes[incoming_chunk_source];
        
        int outgoing_chunk_start = outgoing_chunk_source * chunk_size;
        int outgoing_chunk_len = chunk_sizes[outgoing_chunk_source];
        
        /* Post non-blocking receive and send for this stage */
        int send_issued = 0, recv_issued = 0;
        
        /* Receive from previous rank (if not our own chunk on first stage) */
        if (stage > 0 || rank != 0) {
            MPI_Irecv(x + incoming_chunk_start, incoming_chunk_len, MPI_FLOAT, 
                     prev_rank, stage, MPI_COMM_WORLD, &req_recv);
            recv_issued = 1;
        }
        
        /* Send to next rank */
        MPI_Isend(x + outgoing_chunk_start, outgoing_chunk_len, MPI_FLOAT, 
                 next_rank, stage, MPI_COMM_WORLD, &req_send);
        send_issued = 1;
        
        /* Wait for both operations */
        if (recv_issued) MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
        if (send_issued) MPI_Wait(&req_send, MPI_STATUS_IGNORE);
    }
    
    free(chunk_sizes);
    t_comm += MPI_Wtime() - t_start;

    /* Local SpMV after pipelined receive completes */
    t_start = MPI_Wtime();
    local_spvec(A_local, x, y, thread_count);
    t_comp += MPI_Wtime() - t_start;

    /* Gather y results to rank 0 */
    t_start = MPI_Wtime();
    
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
/* Communication Mode 5: Ring-based Reduction - y through ring to rank 0 */
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

    /* Ring-based reduction: collect all y values at rank 0 via ring */
    t_start = MPI_Wtime();
    
    int next_rank = (rank + 1) % size;
    int prev_rank = (rank - 1 + size) % size;
    
    /* Gather local row counts to all ranks for proper indexing */
    int *all_row_counts = (int *)malloc(size * sizeof(int));
    MPI_Allgather(&m_local, 1, MPI_INT, all_row_counts, 1, MPI_INT, MPI_COMM_WORLD);
    
    /* Compute displacements for rank 0 to place received data */
    int *displs = NULL;
    if (rank == 0) {
        displs = (int *)malloc(size * sizeof(int));
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i-1] + all_row_counts[i-1];
        }
    }
    
    /* Ring communication: pass y values around, rank 0 collects in mpi_y_global */
    MPI_Request req_send, req_recv;
    
    for (int stage = 0; stage < size; stage++) {
        int sender_rank = stage;
        int sender_rows = all_row_counts[sender_rank];
        
        if (rank == sender_rank) {
            /* This rank sends its y to next rank in ring */
            MPI_Isend(y, m_local, MPI_FLOAT, next_rank, stage, MPI_COMM_WORLD, &req_send);
            MPI_Wait(&req_send, MPI_STATUS_IGNORE);
        }
        
        if (rank == 0 && sender_rank != 0) {
            /* Rank 0 receives data from each non-zero rank (passed through ring) */
            MPI_Irecv(mpi_y_global + displs[sender_rank], sender_rows, MPI_FLOAT, 
                     prev_rank, stage, MPI_COMM_WORLD, &req_recv);
            MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
        } else if (rank == 0 && sender_rank == 0) {
            /* Rank 0 copies its own y to result buffer */
            memcpy(mpi_y_global, y, m_local * sizeof(float));
        } else {
            /* Non-zero ranks pass data through the ring */
            MPI_Irecv(y_temp, all_row_counts[(rank - 1 + size) % size], MPI_FLOAT,
                     prev_rank, stage, MPI_COMM_WORLD, &req_recv);
            MPI_Isend(y_temp, sender_rows, MPI_FLOAT, next_rank, stage, MPI_COMM_WORLD, &req_send);
            MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
            MPI_Wait(&req_send, MPI_STATUS_IGNORE);
        }
    }
    
    free(all_row_counts);
    if (rank == 0) free(displs);
    t_comm += MPI_Wtime() - t_start;

    *comm_time = t_comm;
    *compute_time = t_comp;
}

/*------------------------------------------------------------------*/
/* Communication Mode 6: Async Collectives (MPI_Ibcast/MPI_Igatherv) */
void comm_async_collectives(int rank, int size, csr_matrix *A_local, 
                            float *x, float *y, int thread_count,
                            double *comm_time, double *compute_time) {
    double t_comm = 0.0, t_comp = 0.0;
    
    /* Non-blocking broadcast of x */
    MPI_Request req_bcast, req_gather;
    double t_start = MPI_Wtime();
    MPI_Ibcast(x, n_global, MPI_FLOAT, 0, MPI_COMM_WORLD, &req_bcast);
    t_comm += MPI_Wtime() - t_start;

    /* Local SpMV (overlapped with Ibcast) */
    t_start = MPI_Wtime();
    local_spvec(A_local, x, y, thread_count);
    t_comp += MPI_Wtime() - t_start;

    /* Wait for broadcast to complete */
    t_start = MPI_Wtime();
    MPI_Wait(&req_bcast, MPI_STATUS_IGNORE);
    t_comm += MPI_Wtime() - t_start;

    /* Non-blocking gather of y results to rank 0 */
    t_start = MPI_Wtime();
    
    /* For Igatherv, pre-gather local row counts on all ranks */
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
    
    /* Use Igatherv for non-blocking gather of y to rank 0 */
    MPI_Igatherv(y, m_local, MPI_FLOAT, mpi_y_global, mpi_send_counts, mpi_displs, MPI_FLOAT, 0, MPI_COMM_WORLD, &req_gather);
    MPI_Wait(&req_gather, MPI_STATUS_IGNORE);
    t_comm += MPI_Wtime() - t_start;

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
                   const char *matrix_file, int iterations) {
    /* Find baseline (first mode) */
    double baseline_time = modes[0].avg_time;

    printf("\n=== Communication Modes Results ===\n");
    printf("%-25s | Avg Time (ms) | Std Dev  | Comm (ms) | Speedup | Efficiency\n", "Configuration");
    printf("%-25s | --------------|----------|-----------|---------|------------\n", "Configuration");

    for (int i = 0; i < num_modes; i++) {
        modes[i].speedup = baseline_time / modes[i].avg_time;
        modes[i].efficiency_pct = (modes[i].speedup / size) * 100.0;
        
        printf("%-25s | %13.4f | %8.4f | %9.4f | %7.2f | %10.2f%%\n",
               modes[i].name,
               modes[i].avg_time * 1000,
               modes[i].std_dev * 1000,
               modes[i].comm_time * 1000,
               modes[i].speedup,
               modes[i].efficiency_pct);
    }

    /* Write CSV */
    char csv_dir[256];
    char csv_filename[256];
    snprintf(csv_dir, sizeof(csv_dir), "results/test_results_Xnodes");
    snprintf(csv_filename, sizeof(csv_filename), "%s/test_config_mpi_results_%dnodes.csv", csv_dir, size);
    
    FILE *csv = fopen(csv_filename, "a");
    if (!csv) {
        fprintf(stderr, "Failed to open %s for writing\n", csv_filename);
        return;
    }

    /* Extract just the matrix filename (basename) from path */
    const char *basename = strrchr(matrix_file, '/');
    const char *csv_matrix_name = basename ? basename + 1 : matrix_file;

    /* Write header if file is empty */
    fseek(csv, 0, SEEK_END);
    if (ftell(csv) == 0) {
        fprintf(csv, "num_nodes,matrix,rows,cols,nnz,density_pct,config_name,avg_time_ms,std_dev_ms,min_time_ms,max_time_ms,comm_time_ms,compute_time_ms,speedup,efficiency_pct,iterations,notes\n");
    }

    /* Write data rows */
    for (int i = 0; i < num_modes; i++) {
        double density = (nnz_global / (double)(m_global * n_global)) * 100.0;
        fprintf(csv, "%d,%s,%d,%d,%lld,%.4f,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%d,\"\"\n",
                size, csv_matrix_name, m_global, n_global, nnz_global, 
                density, modes[i].name,
                modes[i].avg_time * 1000, modes[i].std_dev * 1000, modes[i].min_time * 1000, modes[i].max_time * 1000,
                modes[i].comm_time * 1000, modes[i].compute_time * 1000, modes[i].speedup, 
                modes[i].efficiency_pct, iterations);
    }

    fclose(csv);
    printf("\nResults written to: %s\n", csv_filename);
}

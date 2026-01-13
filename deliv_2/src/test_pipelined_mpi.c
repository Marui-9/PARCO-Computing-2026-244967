/* File:     
 *     test_pipelined_mpi.c
 *
 * Purpose:  
 *     Test pipelined communication-computation overlap for distributed SpMV
 *     Compares 4 communication strategies:
 *     1. Standard blocking (MPI_Bcast + MPI_Gatherv)
 *     2. Non-blocking (Ibcast + Igatherv) 
 *     3. Async collectives (Ibcast + Igatherv with explicit compute overlap)
 *     4. Pipelined x distribution (chunked broadcast with immediate computation)
 *
 * Compile:  mpicc -g -Wall -O3 -fopenmp -o test_pipelined_mpi \
 *               test_pipelined_mpi.c generator.c m_to_csr.c -lm
 * 
 * Usage:
 *     mpirun -np 2 ./test_pipelined_mpi <matrix_file> [iterations]
 *     mpirun -np 3 ./test_pipelined_mpi <matrix_file> [iterations]
 *     mpirun -np 4 ./test_pipelined_mpi <matrix_file> [iterations]
 *     Example: mpirun -np 2 ./test_pipelined_mpi matrices/11k_0p35.mtx 10
 *
 * Notes:  
 *     - Tests 4 communication modes (3 baseline + 1 pipelined)
 *     - Matrix distributed row-wise across ranks
 *     - Fixed 48 threads per rank (optimal from NUMA analysis)
 *     - Pipelined mode divides x into K chunks and broadcasts incrementally
 *     - Outputs CSV: test_config_mpi_results.csv (compatible with existing format)
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
void comm_async_collectives(int rank, int size, csr_matrix *A_local, 
                            float *x, float *y, int thread_count,
                            double *comm_time, double *compute_time);
void comm_pipelined_chunked(int rank, int size, csr_matrix *A_local, 
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
        thread_count = 48;
        omp_set_num_threads(thread_count);
    } else {
        thread_count = omp_get_max_threads();
    }
    
    /* Try path variants: direct, ../matrices/, ../../matrices/ */
    char working_matrix_path[1024];
    FILE *fp_test = NULL;
    
    fp_test = fopen(matrix_file, "r");
    if (fp_test) {
        strcpy(working_matrix_path, matrix_file);
        fclose(fp_test);
    } else if (matrix_file[0] != '/') {
        snprintf(working_matrix_path, sizeof(working_matrix_path), "../%s", matrix_file);
        fp_test = fopen(working_matrix_path, "r");
        if (fp_test) {
            fclose(fp_test);
        } else {
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
        printf("\n=== MPI Communication Modes Benchmark (with Pipelined) ===\n");
        printf("MPI Ranks: %d, Threads/rank: %d\n", global_size, thread_count);
        printf("Iterations: %d\n\n", num_iterations);
        printf("Testing 4 MPI communication modes (3 baseline + 1 pipelined)...\n\n");
    }
    
    /* Import and distribute matrix */
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
               (nnz_global / (double)((long long)m_global * (long long)n_global)) * 100.0);
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
    CommMode modes[4];
    int num_modes = 0;

    // /* Mode 1: Standard MPI_Bcast + MPI_Gatherv (baseline) */
    // strcpy(modes[num_modes].name, "MPI_Bcast+Gatherv");
    // strcpy(modes[num_modes].description, "Standard blocking broadcast and gather");
    // run_benchmark(&modes[num_modes], global_rank, global_size, A_local,
    //               comm_bcast_reduce, num_iterations, thread_count);
    // num_modes++;

    // /* Mode 2: Non-blocking Ibcast/Igatherv */
    // strcpy(modes[num_modes].name, "Ibcast/Igatherv");
    // strcpy(modes[num_modes].description, "Non-blocking broadcast and gather");
    // run_benchmark(&modes[num_modes], global_rank, global_size, A_local,
    //               comm_ibcast_igatherv, num_iterations, thread_count);
    // num_modes++;

    // /* Mode 3: Async Collectives */
    // strcpy(modes[num_modes].name, "Async_Collectives");
    // strcpy(modes[num_modes].description, "Asynchronous collectives: Ibcast for x, Iallgatherv for y to all ranks");
    // run_benchmark(&modes[num_modes], global_rank, global_size, A_local,
    //               comm_async_collectives, num_iterations, thread_count);
    // num_modes++;

    /* Mode 4: Pipelined chunked communication */
    strcpy(modes[num_modes].name, "Pipelined_Chunked");
    strcpy(modes[num_modes].description, "x vector divided into chunks, broadcast + compute overlap");
    run_benchmark(&modes[num_modes], global_rank, global_size, A_local,
                  comm_pipelined_chunked, num_iterations, thread_count);
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

    A_local->rows = local_rows;
    A_local->cols = A_global->cols;
    A_local->row_ptr = (int *)malloc((local_rows + 1) * sizeof(int));

    if (!A_local->row_ptr) {
        fprintf(stderr, "Rank %d: Failed to allocate row_ptr\n", global_rank);
        MPI_Finalize();
        exit(1);
    }

    long long local_nnz = 0;
    A_local->row_ptr[0] = 0;

    for (int i = start_row; i < end_row; i++) {
        long long nnz_in_row = A_global->row_ptr[i + 1] - A_global->row_ptr[i];
        local_nnz += nnz_in_row;
        A_local->row_ptr[i - start_row + 1] = local_nnz;
    }

    A_local->nnz = local_nnz;

    A_local->col_ind = (int *)malloc(local_nnz * sizeof(int));
    A_local->values = (float *)malloc(local_nnz * sizeof(float));

    if (!A_local->col_ind || !A_local->values) {
        fprintf(stderr, "Rank %d: Failed to allocate col_ind or values\n", global_rank);
        MPI_Finalize();
        exit(1);
    }

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
/* Local SpMV that accumulates into y (for partial computation) */
void local_spvec_accumulate(csr_matrix *A_local, float *x, float *y, int thread_count) {
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
        y_aligned[i] += sum;  /* Accumulate instead of assign */
    }
}

/*------------------------------------------------------------------*/
/* Communication Mode 1: Standard MPI_Bcast + MPI_Gatherv */
void comm_bcast_reduce(int rank, int size, csr_matrix *A_local, 
                       float *x, float *y, int thread_count,
                       double *comm_time, double *compute_time) {
    double t_comm = 0.0, t_comp = 0.0;
    
    double t_start = MPI_Wtime();
    MPI_Bcast(x, n_global, MPI_FLOAT, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t_start;

    t_start = MPI_Wtime();
    local_spvec(A_local, x, y, thread_count);
    t_comp += MPI_Wtime() - t_start;

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
/* Communication Mode 2: Non-blocking Ibcast/Igatherv */
void comm_ibcast_igatherv(int rank, int size, csr_matrix *A_local, 
                          float *x, float *y, int thread_count,
                          double *comm_time, double *compute_time) {
    double t_comm = 0.0, t_comp = 0.0;
    
    MPI_Request req_bcast, req_gather;
    double t_start = MPI_Wtime();
    MPI_Ibcast(x, n_global, MPI_FLOAT, 0, MPI_COMM_WORLD, &req_bcast);
    
    /* Wait for broadcast to complete before using x */
    MPI_Wait(&req_bcast, MPI_STATUS_IGNORE);
    t_comm += MPI_Wtime() - t_start;

    /* Computation: Local SpMV */
    t_start = MPI_Wtime();
    local_spvec(A_local, x, y, thread_count);
    t_comp += MPI_Wtime() - t_start;

    /* Communication: Non-blocking gather */
    t_start = MPI_Wtime();
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
    
    MPI_Igatherv(y, m_local, MPI_FLOAT, mpi_y_global, mpi_send_counts, mpi_displs, MPI_FLOAT, 0, MPI_COMM_WORLD, &req_gather);
    MPI_Wait(&req_gather, MPI_STATUS_IGNORE);
    t_comm += MPI_Wtime() - t_start;

    *comm_time = t_comm;
    *compute_time = t_comp;
}

/*------------------------------------------------------------------*/
/* Communication Mode 3: Async Collectives (Ibcast + Iallgatherv) */
void comm_async_collectives(int rank, int size, csr_matrix *A_local, 
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
/* Communication Mode 4: Selective Pipelined x Distribution
 * 
 * Strategy: Divide x into chunks. Each rank determines which chunks it needs
 * (based on which columns appear in its local matrix). Only send required chunks
 * to ranks that need them. This reduces communication volume compared to 
 * broadcasting the full x vector.
 * 
 * Benefits: 
 * - Reduced communication: Ranks only receive x chunks they actually use
 * - Selective distribution: Avoids sending data that won't be used
 * - Background sends: Rank 0 sends happen in background during computation
 */
/* Optimized Communication Mode 4: Pipelined with Preprocessing */
void comm_pipelined_chunked_optimized(int rank, int size, csr_matrix *A_local, 
                                       float *x, float *y, int thread_count,
                                       double *comm_time, double *compute_time) {
    double t_comm = 0.0, t_comp = 0.0;
    
    int num_chunks = (global_size < 4) ? 4 : global_size * 2;
    if (num_chunks > n_global) num_chunks = n_global;
    int chunk_size = (n_global + num_chunks - 1) / num_chunks;
    
    /* Phase 1: Determine which chunks this rank needs */
    int *need_chunk = (int *)calloc(num_chunks, sizeof(int));
    if (!need_chunk) {
        fprintf(stderr, "Rank %d: Failed to allocate need_chunk\n", rank);
        MPI_Finalize();
        exit(1);
    }
    
    for (long long j = 0; j < A_local->nnz; j++) {
        int col = A_local->col_ind[j];
        int chunk_id = col / chunk_size;
        if (chunk_id < num_chunks) {
            need_chunk[chunk_id] = 1;
        }
    }
    
    /* Phase 2: Preprocess CSR - build per-chunk index structure */
    typedef struct {
        int row;
        long long nz_start;
        long long nz_end;
    } ChunkRowData;
    
    ChunkRowData **chunk_data = (ChunkRowData **)calloc(num_chunks, sizeof(ChunkRowData *));
    int *chunk_row_count = (int *)calloc(num_chunks, sizeof(int));
    
    if (!chunk_data || !chunk_row_count) {
        fprintf(stderr, "Rank %d: Failed to allocate chunk structures\n", rank);
        MPI_Finalize();
        exit(1);
    }
    
    /* Count rows per chunk */
    for (int i = 0; i < m_local; i++) {
        int chunks_in_row[num_chunks];
        memset(chunks_in_row, 0, num_chunks * sizeof(int));
        
        for (long long j = A_local->row_ptr[i]; j < A_local->row_ptr[i + 1]; j++) {
            int col = A_local->col_ind[j];
            int chunk_id = col / chunk_size;
            if (chunk_id < num_chunks && !chunks_in_row[chunk_id]) {
                chunks_in_row[chunk_id] = 1;
                chunk_row_count[chunk_id]++;
            }
        }
    }
    
    /* Allocate chunk data arrays */
    for (int c = 0; c < num_chunks; c++) {
        if (chunk_row_count[c] > 0) {
            chunk_data[c] = (ChunkRowData *)malloc(chunk_row_count[c] * sizeof(ChunkRowData));
            if (!chunk_data[c]) {
                fprintf(stderr, "Rank %d: Failed to allocate chunk_data[%d]\n", rank, c);
                MPI_Finalize();
                exit(1);
            }
        }
    }
    
    /* Build chunk indices - store row and NZ range for each chunk */
    int *chunk_idx = (int *)calloc(num_chunks, sizeof(int));
    for (int i = 0; i < m_local; i++) {
        for (long long j = A_local->row_ptr[i]; j < A_local->row_ptr[i + 1]; j++) {
            int col = A_local->col_ind[j];
            int chunk_id = col / chunk_size;
            if (chunk_id < num_chunks) {
                int idx = chunk_idx[chunk_id];
                if (idx == 0 || chunk_data[chunk_id][idx-1].row != i) {
                    chunk_data[chunk_id][idx].row = i;
                    chunk_data[chunk_id][idx].nz_start = j;
                    chunk_data[chunk_id][idx].nz_end = j + 1;
                    chunk_idx[chunk_id]++;
                } else {
                    chunk_data[chunk_id][idx-1].nz_end = j + 1;
                }
            }
        }
    }
    free(chunk_idx);
    
    /* Phase 3: Exchange communication plan */
    int *all_needs = NULL;
    if (rank == 0) {
        all_needs = (int *)malloc(size * num_chunks * sizeof(int));
        if (!all_needs) {
            fprintf(stderr, "Rank 0: Failed to allocate all_needs\n");
            MPI_Finalize();
            exit(1);
        }
    }
    
    double t_comm_start = MPI_Wtime();
    MPI_Gather(need_chunk, num_chunks, MPI_INT, all_needs, num_chunks, MPI_INT, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t_comm_start;
    
    /* Initialize y to zero */
    #pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int i = 0; i < m_local; i++) {
        y[i] = 0.0f;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    /* Allocate double buffers for overlap */
    float *x_chunk[2];
    for (int b = 0; b < 2; b++) {
        if (posix_memalign((void**)&x_chunk[b], 64, chunk_size * sizeof(float)) != 0) {
            fprintf(stderr, "Rank %d: Failed to allocate x_chunk[%d]\n", rank, b);
            MPI_Finalize();
            exit(1);
        }
    }
    
    MPI_Request *send_reqs = NULL;
    MPI_Request recv_req = MPI_REQUEST_NULL;
    
    if (rank == 0) {
        send_reqs = (MPI_Request *)malloc(size * num_chunks * sizeof(MPI_Request));
        if (!send_reqs) {
            fprintf(stderr, "Rank 0: Failed to allocate send_reqs\n");
            MPI_Finalize();
            exit(1);
        }
    }
    
    /* Phase 4: Rank 0 posts all sends */
    int send_count = 0;
    t_comm_start = MPI_Wtime();
    if (rank == 0) {
        for (int c = 0; c < num_chunks; c++) {
            int chunk_start = c * chunk_size;
            int chunk_end = (c == num_chunks - 1) ? n_global : (c + 1) * chunk_size;
            int actual_chunk_size = chunk_end - chunk_start;
            
            for (int r = 1; r < size; r++) {
                if (all_needs[r * num_chunks + c]) {
                    MPI_Isend(x + chunk_start, actual_chunk_size, MPI_FLOAT, r, c, 
                             MPI_COMM_WORLD, &send_reqs[send_count++]);
                }
            }
        }
    }
    t_comm += MPI_Wtime() - t_comm_start;
    
    /* Phase 5: Pipelined processing with double buffering */
    int current_buf = 0;
    int next_chunk_to_prefetch = -1;
    
    /* Find first chunk to prefetch */
    for (int c = 0; c < num_chunks; c++) {
        if (need_chunk[c]) {
            next_chunk_to_prefetch = c;
            break;
        }
    }
    
    for (int c = 0; c < num_chunks; c++) {
        if (!need_chunk[c]) continue;
        
        int chunk_start = c * chunk_size;
        int chunk_end = (c == num_chunks - 1) ? n_global : (c + 1) * chunk_size;
        int actual_chunk_size = chunk_end - chunk_start;
        
        /* Receive current chunk */
        t_comm_start = MPI_Wtime();
        if (rank == 0) {
            memcpy(x_chunk[current_buf], x + chunk_start, actual_chunk_size * sizeof(float));
        } else {
            MPI_Recv(x_chunk[current_buf], actual_chunk_size, MPI_FLOAT, 0, c, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        t_comm += MPI_Wtime() - t_comm_start;
        
        /* Prefetch next chunk in background (non-blocking) if available */
        if (rank != 0 && next_chunk_to_prefetch >= 0 && next_chunk_to_prefetch > c) {
            int next_start = next_chunk_to_prefetch * chunk_size;
            int next_end = (next_chunk_to_prefetch == num_chunks - 1) ? n_global : 
                           (next_chunk_to_prefetch + 1) * chunk_size;
            int next_size = next_end - next_start;
            
            t_comm_start = MPI_Wtime();
            MPI_Irecv(x_chunk[1 - current_buf], next_size, MPI_FLOAT, 0, 
                     next_chunk_to_prefetch, MPI_COMM_WORLD, &recv_req);
            t_comm += MPI_Wtime() - t_comm_start;
        }
        
        /* Compute using preprocessed chunk structure - NO atomic needed! */
        double t_comp_start = MPI_Wtime();
        float *x_buf = x_chunk[current_buf];
        
        #pragma omp parallel for schedule(dynamic, 8) num_threads(thread_count) \
            default(none) shared(chunk_data, chunk_row_count, A_local, x_buf, y, \
                                 c, chunk_start, chunk_end)
        for (int r = 0; r < chunk_row_count[c]; r++) {
            int i = chunk_data[c][r].row;
            float sum = 0.0f;
            
            #pragma omp simd reduction(+:sum)
            for (long long j = chunk_data[c][r].nz_start; j < chunk_data[c][r].nz_end; j++) {
                int col = A_local->col_ind[j];
                if (col >= chunk_start && col < chunk_end) {
                    sum += A_local->values[j] * x_buf[col - chunk_start];
                }
            }
            y[i] += sum;  // NO ATOMIC - each thread has unique i
        }
        t_comp += MPI_Wtime() - t_comp_start;
        
        /* Wait for prefetch to complete before next iteration */
        if (recv_req != MPI_REQUEST_NULL) {
            t_comm_start = MPI_Wtime();
            MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
            t_comm += MPI_Wtime() - t_comm_start;
            recv_req = MPI_REQUEST_NULL;
        }
        
        /* Find next chunk to prefetch */
        next_chunk_to_prefetch = -1;
        for (int nc = c + 1; nc < num_chunks; nc++) {
            if (need_chunk[nc]) {
                next_chunk_to_prefetch = nc;
                break;
            }
        }
        
        current_buf = 1 - current_buf;
    }
    
    /* Wait for all sends */
    if (rank == 0 && send_count > 0) {
        t_comm_start = MPI_Wtime();
        MPI_Waitall(send_count, send_reqs, MPI_STATUSES_IGNORE);
        t_comm += MPI_Wtime() - t_comm_start;
    }
    
    /* Phase 6: Gather results */
    t_comm_start = MPI_Wtime();
    MPI_Gather(&m_local, 1, MPI_INT, mpi_send_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        mpi_displs[0] = 0;
        for (int i = 1; i < size; i++) {
            mpi_displs[i] = mpi_displs[i-1] + mpi_send_counts[i-1];
        }
    }
    
    MPI_Gatherv(y, m_local, MPI_FLOAT, mpi_y_global, mpi_send_counts, mpi_displs, 
                MPI_FLOAT, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t_comm_start;
    
    /* Cleanup */
    free(need_chunk);
    free(chunk_row_count);
    for (int c = 0; c < num_chunks; c++) {
        if (chunk_data[c]) free(chunk_data[c]);
    }
    free(chunk_data);
    for (int b = 0; b < 2; b++) free(x_chunk[b]);
    if (rank == 0) {
        free(all_needs);
        free(send_reqs);
    }
    
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

    /* Write to unified CSV file (matching test_configurations_mpi.c format) */
    char csv_filename[256];
    snprintf(csv_filename, sizeof(csv_filename), "results/pipelined_results.csv");
    
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

    /* Extract just the matrix filename (basename) from path */
    const char *basename = strrchr(matrix_file, '/');
    const char *csv_matrix_name = basename ? basename + 1 : matrix_file;

    /* Write header if file doesn't exist */
    if (!file_exists) {
        fprintf(csv, "num_procs,matrix,rows,cols,nnz,density_pct,config_name,avg_time_ms,std_dev_ms,min_time_ms,max_time_ms,comm_time_ms,compute_time_ms,speedup,efficiency_pct,iterations,notes\n");
    }

    /* Write data rows */
    for (int i = 0; i < num_modes; i++) {
        double density = (nnz_global / (double)((long long)m_global * (long long)n_global)) * 100.0;
        fprintf(csv, "%d,%s,%d,%d,%lld,%.4f,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%d,\"\"\n",
                size, csv_matrix_name, m_global, n_global, nnz_global, 
                density, modes[i].name,
                modes[i].avg_time * 1000, modes[i].std_dev * 1000, modes[i].min_time * 1000, modes[i].max_time * 1000,
                modes[i].comm_time * 1000, modes[i].compute_time * 1000, modes[i].speedup, 
                modes[i].efficiency_pct, iterations);
    }

    fclose(csv);
    printf("\nResults appended to: %s\n", csv_filename);
}

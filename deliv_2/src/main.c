/* File:     
 *     main.c 
 *
 * Purpose:  
 *     Computes a parallel matrix-vector product using OpenMP and MPI.
 *     Matrix is distributed by block rows across MPI ranks.
 *     Each rank uses OpenMP for intra-node parallelism.
 *     Vectors are distributed using MPI_Allgatherv.
 *
 * Input:
 *     m, n: order of matrix
 *     A, x: the matrix and the vector to be multiplied
 *
 * Output:
 *     y: the product vector
 *
 * Compile:  mpicc -O3 -Wall -Wextra -march=native -fopenmp -o mtrvec main.c generator.c m_to_csr.c -lm
 * Usage:
 *     mpirun -np <num_mpi_ranks> ./mtrvec <thread_count> <matrix_file>
 *
 * Notes:  
 *     1. Local storage for A, x, y is dynamically allocated.
 *     2. Number of threads (thread_count) should ideally divide the number of matrix rows.
 *     3. We use CSR format for efficient sparse matrix storage.
 *     4. Each MPI rank owns a contiguous block of rows.
 *     5. Vector x is broadcast to all ranks using MPI_Bcast.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include "generator.h"
#include "m_to_csr.h"

/* Global variables */
int     thread_count;
int     m, n;  // global matrix dimensions
int     rank, num_ranks;
float* x;      // The full input vector
float* y;      // The output vector
float* local_y; // Local output vector for this rank

double start_time, end_time;

/* Forward declarations */
void Usage(char* prog_name);
void Print_matrix(char* title, float A[], int m, int n);
void Print_vector(char* title, float y[], int m);

/* Parallel functions */
void Omp_mat_vect(int thread_count, csr_matrix *csr_A, float *local_x, float *local_y);
void Mpi_Omp_mat_vect(int thread_count, int rank, int num_ranks,
                      int global_m, int global_n, csr_matrix *local_csr_A,
                      float *x, float *local_y, int local_m);

int compare_doubles(const void *a, const void *b) {
   double diff = (*(double*)a - *(double*)b);
   return (diff > 0) - (diff < 0);
}

/*------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

   if (argc != 3) Usage(argv[0]);
   thread_count = atoi(argv[1]);
   
   char matrix_file[256];
   if (strchr(argv[2], '/') != NULL || argv[2][0] == '.') {
      snprintf(matrix_file, sizeof(matrix_file), "%s", argv[2]);
   } else {
      snprintf(matrix_file, sizeof(matrix_file), "matrices/%s", argv[2]);
   }
   
   if (rank == 0) {
      printf("Using %d MPI ranks, %d threads per rank, matrix file: %s\n", 
             num_ranks, thread_count, matrix_file);
   }

   // ===== LOAD AND DISTRIBUTE MATRIX =====
   // Only rank 0 imports full matrix for baseline; MPI version each rank loads its rows
   // Use sparse CSR format directly (no dense allocation)
   csr_matrix *csr_A = NULL;
   
   if (rank == 0) {
      int result = import_matrix_to_csr(matrix_file, &csr_A);
      if (result != 0) {
         fprintf(stderr, "Failed to import matrix from %s\n", matrix_file);
         MPI_Finalize();
         return 1;
      }
      m = csr_A->rows;
      n = csr_A->cols;
      int nnz = csr_A->nnz;
      double nnz_percentage = (nnz / (double)(m * n)) * 100.0;
      printf("Matrix loaded from file: %s\n", matrix_file);
      printf("Matrix dimensions: %d rows × %d cols, %.2f%% non-zero entries\n", 
             m, n, nnz_percentage);
   }
   
   // Broadcast matrix dimensions to all ranks
   MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

   // Generate x vector (all ranks have full x)
   x = generate_vector(n);
   if (!x) {
      fprintf(stderr, "Failed to generate vector\n");
      if (csr_A) csr_free(csr_A);
      MPI_Finalize();
      return 1;
   }

   // Allocate full y vector on all ranks for comparison
   if (posix_memalign((void**)&y, 64, (size_t)m * sizeof(float)) != 0) {
       fprintf(stderr, "Failed to allocate aligned memory for y\n");
       if (csr_A) csr_free(csr_A);
       free(x);
       MPI_Finalize();
       return 1;
   }

   // ===== OPENMP BASELINE (Rank 0 only) =====
   int num_iterations = 10;
   double *omp_times = malloc(num_iterations * sizeof(double));
   double *mpi_times = malloc(num_iterations * sizeof(double));
   
   if (!omp_times || !mpi_times) {
      fprintf(stderr, "Failed to allocate timing arrays\n");
      if (csr_A) csr_free(csr_A);
      free(x);
      free(y);
      MPI_Finalize();
      return 1;
   }

   if (rank == 0) {
      printf("\nRunning %d iterations (OpenMP baseline on rank 0 only)...\n", num_iterations);
      fflush(stdout);
   }

   // ===== PRE-COMPUTE LOCAL CSR MATRICES FOR MPI VERSION =====
   // Strategy: Rank 0 reads full CSR once, then each rank extracts its own rows
   // This avoids reading the .mtx file multiple times for large matrices
   
   csr_matrix *full_csr_A = NULL;
   
   // Rank 0 reads the full matrix (already done above in csr_A, reuse it)
   if (rank != 0) {
      // Other ranks need the full matrix to extract their rows
      // For now, have them read it too (could optimize with MPI broadcast)
      // This is still faster than reading by row range
      int dummy_result = import_matrix_to_csr(matrix_file, &full_csr_A);
      if (dummy_result != 0) {
         fprintf(stderr, "Rank %d: Failed to import full matrix\n", rank);
         if (csr_A) csr_free(csr_A);
         free(x);
         free(y);
         free(omp_times);
         free(mpi_times);
         MPI_Finalize();
         return 1;
      }
   } else {
      full_csr_A = csr_A;  // Rank 0 uses the CSR it already read
   }
   
   // Prepare for MPI version: partition matrix by rows
   int row_start = rank * (m / num_ranks);
   int row_end = (rank == num_ranks - 1) ? m : (rank + 1) * (m / num_ranks);
   int local_m = row_end - row_start;

   // Allocate local output
   if (posix_memalign((void**)&local_y, 64, (size_t)local_m * sizeof(float)) != 0) {
      fprintf(stderr, "Rank %d: Failed to allocate local_y\n", rank);
      if (csr_A) csr_free(csr_A);
      free(x);
      free(y);
      free(omp_times);
      free(mpi_times);
      MPI_Finalize();
      return 1;
   }

   // Create local CSR matrix by extracting rows from full_csr_A
   // This is done ONCE before the iteration loop to avoid repeated file I/O
   csr_matrix *local_csr_A = (csr_matrix *)malloc(sizeof(csr_matrix));
   if (!local_csr_A) {
      fprintf(stderr, "Rank %d: Failed to allocate local CSR structure\n", rank);
      if (full_csr_A && rank != 0) csr_free(full_csr_A);
      if (csr_A) csr_free(csr_A);
      free(x);
      free(y);
      free(local_y);
      free(omp_times);
      free(mpi_times);
      MPI_Finalize();
      return 1;
   }
   
   // Extract rows [row_start, row_end) from full_csr_A
   // Count non-zeros in this range
   int local_nnz = 0;
   for (int i = row_start; i < row_end; i++) {
      local_nnz += full_csr_A->row_ptr[i+1] - full_csr_A->row_ptr[i];
   }
   
   // Allocate CSR arrays for local matrix
   int *local_row_ptr = (int *)malloc((local_m + 1) * sizeof(int));
   int *local_col_ind = (int *)malloc(local_nnz * sizeof(int));
   float *local_values = (float *)malloc(local_nnz * sizeof(float));
   
   if (!local_row_ptr || !local_col_ind || !local_values) {
      fprintf(stderr, "Rank %d: Failed to allocate local CSR arrays\n", rank);
      free(local_row_ptr);
      free(local_col_ind);
      free(local_values);
      free(local_csr_A);
      if (full_csr_A && rank != 0) csr_free(full_csr_A);
      if (csr_A) csr_free(csr_A);
      free(x);
      free(y);
      free(local_y);
      free(omp_times);
      free(mpi_times);
      MPI_Finalize();
      return 1;
   }
   
   // Copy row_ptr and values for rows [row_start, row_end)
   local_row_ptr[0] = 0;
   int nnz_offset = 0;
   for (int i = row_start; i < row_end; i++) {
      int row_nnz = full_csr_A->row_ptr[i+1] - full_csr_A->row_ptr[i];
      // Copy column indices and values
      memcpy(&local_col_ind[nnz_offset], 
             &full_csr_A->col_ind[full_csr_A->row_ptr[i]], 
             row_nnz * sizeof(int));
      memcpy(&local_values[nnz_offset], 
             &full_csr_A->values[full_csr_A->row_ptr[i]], 
             row_nnz * sizeof(float));
      nnz_offset += row_nnz;
      local_row_ptr[i - row_start + 1] = nnz_offset;
   }
   
   local_csr_A->rows = local_m;
   local_csr_A->cols = n;
   local_csr_A->nnz = local_nnz;
   local_csr_A->row_ptr = local_row_ptr;
   local_csr_A->col_ind = local_col_ind;
   local_csr_A->values = local_values;
   
   if (rank == 0) {
      printf("Rank %d: Extracted rows [%d,%d) - %d non-zeros\n", rank, row_start, row_end, local_nnz);
   }

   for (int iter = 0; iter < num_iterations; iter++) {
      //---------OPENMP VERSION (baseline, rank 0 only)---------
      if (rank == 0) {
         start_time = omp_get_wtime();
         Omp_mat_vect(thread_count, csr_A, x, y);
         end_time = omp_get_wtime();
         omp_times[iter] = end_time - start_time;
      }

      // Synchronize all ranks
      MPI_Barrier(MPI_COMM_WORLD);

      //---------MPI + OPENMP VERSION---------
      // Synchronize before timing
      MPI_Barrier(MPI_COMM_WORLD);

      start_time = MPI_Wtime();
      Mpi_Omp_mat_vect(thread_count, rank, num_ranks, m, n, local_csr_A, x, local_y, local_m);
      
      // Gather results from all ranks to rank 0
      // Prepare receive counts and displacements for rank 0
      int *recvcounts = NULL;
      int *displs = NULL;
      
      if (rank == 0) {
         recvcounts = (int *)malloc(num_ranks * sizeof(int));
         displs = (int *)malloc(num_ranks * sizeof(int));
         
         // Calculate how many rows each rank has
         int disp = 0;
         for (int r = 0; r < num_ranks; r++) {
            int r_row_start = r * (m / num_ranks);
            int r_row_end = (r == num_ranks - 1) ? m : (r + 1) * (m / num_ranks);
            int r_local_m = r_row_end - r_row_start;
            recvcounts[r] = r_local_m;
            displs[r] = disp;
            disp += r_local_m;
         }
      }
      
      // Gather local_y from all ranks into y on rank 0
      MPI_Gatherv(local_y, local_m, MPI_FLOAT,
                  y, recvcounts, displs, MPI_FLOAT,
                  0, MPI_COMM_WORLD);
      
      MPI_Barrier(MPI_COMM_WORLD);
      end_time = MPI_Wtime();
      mpi_times[iter] = end_time - start_time;
      
      // Cleanup per-iteration temporary allocations
      if (rank == 0) {
         free(recvcounts);
         free(displs);
      }
   }

   // Cleanup CSR matrices and output buffer
   // Note: local_csr_A points to arrays we allocated, need to free them
   if (local_csr_A) {
      free(local_csr_A->row_ptr);
      free(local_csr_A->col_ind);
      free(local_csr_A->values);
      free(local_csr_A);
   }
   
   // Free full matrix if rank != 0 (rank 0 uses csr_A which is freed below)
   if (full_csr_A && rank != 0) {
      csr_free(full_csr_A);
   }
   
   free(local_y);

   // Sort times
   if (rank == 0) {
      qsort(omp_times, num_iterations, sizeof(double), compare_doubles);
   }
   qsort(mpi_times, num_iterations, sizeof(double), compare_doubles);

   // Use best 90% of runs 
   int percentile_count = (int)(num_iterations * 0.9);
   if (percentile_count == 0) percentile_count = 1;

   // Calculate averages
   if (rank == 0) {
      double sum_omp = 0.0;
      for (int i = 0; i < percentile_count; i++) {
         sum_omp += omp_times[i];
      }
      double avg_omp_time = sum_omp / percentile_count;

      double sum_mpi = 0.0;
      for (int i = 0; i < percentile_count; i++) {
         sum_mpi += mpi_times[i];
      }
      double avg_mpi_time = sum_mpi / percentile_count;
      double avg_speedup = avg_omp_time / avg_mpi_time;

      printf("\n=== Results (90th percentile of %d runs, using best %d) ===\n", 
             num_iterations, percentile_count);
      printf("Average OpenMP execution time:      %.6f milliseconds\n", avg_omp_time * 1000);
      printf("Average MPI+OpenMP execution time:  %.6f milliseconds\n", avg_mpi_time * 1000);
      printf("Speedup (OpenMP vs MPI+OpenMP):     %.2fx\n", avg_speedup);
   }

   // Cleanup
   free(omp_times);
   free(mpi_times);
   if (csr_A) csr_free(csr_A);
   free(x);
   free(y);

   MPI_Finalize();
   return 0; 
}  /* main */


/*------------------------------------------------------------------
 * Function:  Usage
 * Purpose:   print a message showing what the command line should
 *            be, and terminate
 * In arg :   prog_name
 */
void Usage (char* prog_name) {
   fprintf(stderr, "usage: mpirun -np <num_mpi_ranks> %s <thread_count> <matrix_file>\n", 
           prog_name);
   exit(0);
}  /* Usage */


/* Function: Omp_mat_vect 
   Purpose: multiply a mxn matrix (CSR format) by a nx1 column vector using OpenMP
   This is the baseline optimized kernel from deliv_1.
   */
void Omp_mat_vect(int thread_count, csr_matrix *csr_A, float *local_x, float *local_y) {
   int i, j;
   const int chunk = 32;
   
   float *x_aligned = __builtin_assume_aligned(local_x, 64);
   float *y_aligned = __builtin_assume_aligned(local_y, 64);

   #pragma omp parallel for schedule(static, chunk) num_threads(thread_count) \
      proc_bind(close) default(none) \
      shared(csr_A, x_aligned, y_aligned, chunk) private(i, j)
   for (i = 0; i < csr_A->rows; i++) {
      float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
      int row_start = csr_A->row_ptr[i];
      int row_end = csr_A->row_ptr[i + 1];
      
      #pragma omp simd reduction(+:sum0, sum1, sum2, sum3) aligned(x_aligned, y_aligned: 64)
      for (j = row_start; j < row_end - 3; j += 4) {
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
}  /* Omp_mat_vect */


/*------------------------------------------------------------------
 * Function:       Mpi_Omp_mat_vect
 * Purpose:        Multiply a distributed matrix by a vector using MPI + OpenMP
 *                 Each MPI rank:
 *                   1. Owns a contiguous block of matrix rows
 *                   2. Replicates the full input vector x
 *                   3. Computes its local portion of output y using OpenMP
 *                   4. Results are gathered back to rank 0 via MPI_Gatherv
 *
 * In args:
 *     thread_count: number of OpenMP threads per rank
 *     rank: MPI rank of this process
 *     num_ranks: total number of MPI ranks
 *     global_m: global number of rows
 *     global_n: global number of columns
 *     local_csr_A: local CSR matrix (this rank's rows, created independently)
 *     x: full input vector (replicated on all ranks)
 *     local_m: number of rows owned by this rank
 *
 * Out args:
 *     local_y: local output vector (this rank's portion)
 *
 * Communication:
 *     - No inter-rank communication during computation (fully parallel)
 *     - MPI_Gatherv called externally to collect results
 */
void Mpi_Omp_mat_vect(int thread_count, int rank __attribute__((unused)), 
                      int num_ranks __attribute__((unused)),
                      int global_m __attribute__((unused)), 
                      int global_n __attribute__((unused)), 
                      csr_matrix *local_csr_A,
                      float *x, float *local_y, int local_m) {
   // Each rank computes its local portion independently using the optimized OpenMP kernel
   Omp_mat_vect(thread_count, local_csr_A, x, local_y);
}  /* Mpi_Omp_mat_vect */
void Print_matrix( char* title, float A[], int m, int n) {
   int   i, j;

   printf("%s\n", title);
   for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++)
         printf("%4.1f ", A[i*n + j]);
      printf("\n");
   }
}  /* Print_matrix */


/*------------------------------------------------------------------
 * Function:    Print_vector
 * Purpose:     Print a vector
 * In args:     title, y, m
 */
void Print_vector(char* title, float y[], int m) {
   int   i;

   printf("%s\n", title);
   for (i = 0; i < m; i++)
      printf("%4.1f ", y[i]);
   printf("\n");
}  /* Print_vector */

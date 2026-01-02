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
float* A;      // The global matrix (only on rank 0)
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
   A = import_matrix(matrix_file, &m, &n);
   if (!A) {
      if (rank == 0) {
         fprintf(stderr, "Failed to import matrix from %s\n", matrix_file);
         fprintf(stderr, "Matrix may be too large to load as dense format.\n");
      }
      MPI_Finalize();
      return 1;
   }

   // Calculate nonzero percentage and print info (rank 0 only)
   if (rank == 0) {
      int nnz_count = 0;
      size_t total_size = (size_t)m * (size_t)n;
      for (size_t i = 0; i < total_size; i++) {
         if (A[i] != 0.0f) nnz_count++;
      }
      double nnz_percentage = (nnz_count / (double)total_size) * 100.0;
      printf("Matrix loaded from file: %s\n", matrix_file);
      printf("Matrix dimensions: %d rows × %d cols, %.2f%% non-zero entries\n", 
             m, n, nnz_percentage);
   }

   // Generate x vector (all ranks have full x)
   x = generate_vector(n);
   if (!x) {
      fprintf(stderr, "Failed to generate vector\n");
      free(A);
      MPI_Finalize();
      return 1;
   }

   // Allocate full y vector on all ranks for comparison
   if (posix_memalign((void**)&y, 64, (size_t)m * sizeof(float)) != 0) {
       fprintf(stderr, "Failed to allocate aligned memory for y\n");
       free(A);
       free(x);
       MPI_Finalize();
       return 1;
   }

   // Convert full matrix to CSR (rank 0 only)
   csr_matrix *csr_A = NULL;
   if (rank == 0) {
      int result = matrix_to_csr(A, m, n, &csr_A);
      if (result != 0) {
         fprintf(stderr, "matrix_to_csr failed\n");
         free(A);
         free(x);
         free(y);
         MPI_Finalize();
         return 1;
      }
   }

   // ===== OPENMP BASELINE (Rank 0 only) =====
   int num_iterations = 30;
   double *omp_times = malloc(num_iterations * sizeof(double));
   double *mpi_times = malloc(num_iterations * sizeof(double));
   
   if (!omp_times || !mpi_times) {
      fprintf(stderr, "Failed to allocate timing arrays\n");
      if (csr_A) csr_free(csr_A);
      free(A);
      free(x);
      free(y);
      MPI_Finalize();
      return 1;
   }

   if (rank == 0) {
      printf("\nRunning %d iterations (OpenMP baseline on rank 0 only)...\n", num_iterations);
      fflush(stdout);
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
      // Prepare for MPI version: partition matrix by rows
      int row_start = rank * (m / num_ranks);
      int row_end = (rank == num_ranks - 1) ? m : (rank + 1) * (m / num_ranks);
      int local_m = row_end - row_start;

      // Allocate local output
      if (posix_memalign((void**)&local_y, 64, (size_t)local_m * sizeof(float)) != 0) {
         fprintf(stderr, "Rank %d: Failed to allocate local_y\n", rank);
         if (csr_A) csr_free(csr_A);
         free(A);
         free(x);
         free(y);
         free(omp_times);
         free(mpi_times);
         MPI_Finalize();
         return 1;
      }

      // Create local CSR matrix (dense -> CSR for this rank's rows)
      csr_matrix *local_csr_A = NULL;
      
      // ALL ranks extract and convert their OWN rows (not just rank 0)
      float *local_A = (float *)malloc((size_t)local_m * n * sizeof(float));
      if (!local_A) {
         fprintf(stderr, "Rank %d: Failed to allocate local_A\n", rank);
         if (csr_A) csr_free(csr_A);
         free(A);
         free(x);
         free(y);
         free(omp_times);
         free(mpi_times);
         MPI_Finalize();
         return 1;
      }
      
      // Extract this rank's rows from the full matrix
      for (int i = 0; i < local_m; i++) {
         memcpy(&local_A[i * n], &A[(row_start + i) * n], n * sizeof(float));
      }
      
      // Convert to CSR (each rank does this independently)
      int result = matrix_to_csr(local_A, local_m, n, &local_csr_A);
      free(local_A);
      if (result != 0) {
         fprintf(stderr, "Rank %d: matrix_to_csr failed for local matrix\n", rank);
         free(local_y);
         free(omp_times);
         free(mpi_times);
         if (csr_A) csr_free(csr_A);
         free(A);
         free(x);
         free(y);
         MPI_Finalize();
         return 1;
      }

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
      
      // Cleanup
      if (rank == 0) {
         free(recvcounts);
         free(displs);
      }
      if (local_csr_A->row_ptr) free(local_csr_A->row_ptr);
      if (local_csr_A->col_ind) free(local_csr_A->col_ind);
      if (local_csr_A->values) free(local_csr_A->values);
      free(local_csr_A);
      free(local_y);
   }

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
   free(A);
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

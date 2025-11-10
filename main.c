/* File:     
 *     main.c 
 *
 * Purpose:  
 *     Computes a parallel matrix-vector product.  Matrix
 *     is distributed by block rows.  Vectors are distributed by 
 *     blocks.
 *
 * Input:
 *     m, n: order of matrix
 *     A, x: the matrix and the vector to be multiplied
 *
 * Output:
 *     y: the product vector
 *
 * Compile:  gcc -g -Wall -o pth_mat_vect pth_mat_vect.c -lpthread
 * Usage:
 *     pth_mat_vect <thread_count>
 *
 * Notes:  
 *     1.  Local storage for A, x, y is dynamically allocated.
 *     2.  Number of threads (thread_count) should evenly divide both 
 *         m and n.  The program doesn't check for this.
 *     3.  We use a 1-dimensional array for A and compute subscripts
 *         using the formula A[i][j] = A[i*n + j]
 *     4.  Distribution of A, x, and y is logical:  all three are 
 *         globally shared.
 *
 * IPP:    Section 4.3 (pp. 159 and ff.).  Also Section 4.10 (pp. 191 and 
 *         ff.)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "generator.h"
#include "m_to_csr.h"

/* Global variables */
int     thread_count;
int     m, n;
int     percent_nonzero;
float* A; // The matrix
float* x; // The input vector
float* y; // The output vector
double start_time_serial, end_time_serial;
double start_time, end_time;

/* Serial functions */
void Usage(char* prog_name);
void Read_matrix(char* prompt, float A[], int m, int n);
void Read_vector(char* prompt, float x[], int n);
void Print_matrix(char* title, float A[], int m, int n);
void Print_vector(char* title, float y[], int m);
void serial_mat_vect(float A[], float x[], float y[], int m, int n);

/* Parallel functions */
void *Pth_mat_vect(void* rank);
void Omp_mat_vect(int thread_count, csr_matrix *csr_A);

int compare_doubles(const void *a, const void *b) {
   double diff = (*(double*)a - *(double*)b);
   return (diff > 0) - (diff < 0);
}

/*------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
   //long       thread;
   //pthread_t* thread_handles;

 //  srand((unsigned)time(NULL));
   if (argc != 3) Usage(argv[0]);
   thread_count = atoi(argv[1]);
   //matrix file path
   char matrix_file[256];
   snprintf(matrix_file, sizeof(matrix_file), "matrices/%s", argv[2]);
   
   printf("Using %d threads and matrix file: %s\n", thread_count, matrix_file);
  // thread_handles = malloc(thread_count*sizeof(pthread_t));

   // printf("Enter number of rows and number of columns\n");
   // scanf("%d%d", &m, &n);

   // printf("Enter desired integer percentage of nonzero entries (0-100)\n");
   // scanf("%d", &percent_nonzero);
   A = import_matrix(matrix_file, &m, &n);
   if (!A) {
      fprintf(stderr, "Failed to import matrix from %s\n", matrix_file);
      fprintf(stderr, "Matrix may be too large to load as dense format.\n");
      return 1;
   }
   x = generate_vector(n);
   y = malloc((size_t)m * sizeof(float));
   
   // Calculate nonzero percentage
   int nnz_count = 0;
   size_t total_size = (size_t)m * (size_t)n;
   for (size_t i = 0; i < total_size; i++) {
      if (A[i] != 0.0f) nnz_count++;
   }
   double nnz_percentage = (nnz_count / (double)total_size) * 100.0;
   
   printf("Matrix loaded from file: %s\n", matrix_file);
   printf("Matrix dimensions: %d rows × %d cols, %.2f%% non-zero entries\n", 
          m, n, nnz_percentage);
   //Print_matrix("Matrix:", A, m, n);
      
   csr_matrix *csr_A;
   int result = matrix_to_csr(
      A,
      m,
      n,
      &csr_A
   );
   if (result != 0) {
        fprintf(stderr, "matrix_to_csr failed\n");
        free(A);
        return 1;
    }

   //Read_vector("Enter the vector", x, n);
   //Print_vector("Vector: ", x, n);

   // Run multiple iterations to get average performance
   int num_iterations = 16;
   double *serial_times = malloc(num_iterations * sizeof(double));
   double *parallel_times = malloc(num_iterations * sizeof(double));
   
   if (!serial_times || !parallel_times) {
      fprintf(stderr, "Failed to allocate timing arrays\n");
      return 1;
   }
   
   printf("\nRunning %d iterations...\n", num_iterations);
   
   for (int iter = 0; iter < num_iterations; iter++) {
      //printf("Iteration %d/%d\n", iter + 1, num_iterations);
      
      //---------SERIAL VERSION---------
      start_time_serial = omp_get_wtime();
      serial_mat_vect(A, x, y, m, n);
      end_time_serial = omp_get_wtime();
      serial_times[iter] = end_time_serial - start_time_serial;
      
      //---------OPENMP VERSION----------
      start_time = omp_get_wtime();
      Omp_mat_vect(thread_count, csr_A);
      end_time = omp_get_wtime();
      parallel_times[iter] = end_time - start_time;
   }
   
   // sort the times
   qsort(serial_times, num_iterations, sizeof(double), compare_doubles);
   qsort(parallel_times, num_iterations, sizeof(double), compare_doubles);
   
   // use best 90% of runs 
   int percentile_count = (int)(num_iterations * 0.9);
   if (percentile_count == 0) percentile_count = 1;
   
   double sum_serial = 0.0;
   double sum_parallel = 0.0;
   for (int i = 0; i < percentile_count; i++) {
      sum_serial += serial_times[i];
      sum_parallel += parallel_times[i];
   }
   
   double avg_serial_time = sum_serial / percentile_count;
   double avg_parallel_time = sum_parallel / percentile_count;
   double avg_speedup = avg_serial_time / avg_parallel_time;
   
   printf("\n=== Results (90th percentile of %d runs, using best %d) ===\n", 
          num_iterations, percentile_count);
   printf("Average serial execution time:   %.6f milliseconds\n", avg_serial_time * 1000);
   printf("Average parallel execution time: %.6f milliseconds\n", avg_parallel_time * 1000);
   printf("Average speedup: %.2fx\n", avg_speedup);

   free(serial_times);
   free(parallel_times);
   free(A);
   free(x);
   //free(y); causes core dump
   //print_csr(csr_A);
   csr_free(csr_A);
   return 0; 
}  /* main */


/*------------------------------------------------------------------
 * Function:  Usage
 * Purpose:   print a message showing what the command line should
 *            be, and terminate
 * In arg :   prog_name
 */
void Usage (char* prog_name) {
   fprintf(stderr, "usage: %s <thread_count> <matrix_file>\n", prog_name);
   exit(0);
}  /* Usage */

/*------------------------------------------------------------------
 * Function:    Read_matrix
 * Purpose:     Read in the matrix
 * In args:     prompt, m, n
 * Out arg:     A
 */
void Read_matrix(char* prompt, float A[], int m, int n) {
   int             i, j;

   printf("%s\n", prompt);
   for (i = 0; i < m; i++) 
      for (j = 0; j < n; j++)
         scanf("%f", &A[i*n+j]);
}  /* Read_matrix */


   /*------------------------------------------------------------------
 * Function:        Read_vector
 * Purpose:         Read in the vector x
 * In arg:          prompt, n
 * Out arg:         x
 */
void Read_vector(char* prompt, float x[], int n) {
   int   i;

   printf("%s\n", prompt);
   for (i = 0; i < n; i++) 
      scanf("%f", &x[i]);
}  /* Read_vector */

/* Function: serial_mat_vect 
   Purpose: multiply a mxn matrix by a nx1 column vector
   */
void serial_mat_vect(float A[], float x[], float y[], int m, int n) {
   int i, j;
   for (i = 0; i < m; i++){
       y[i] = 0.0f;  // row result
       for (j = 0; j < n; j++){
            y[i] += A[i*n + j] * x[j]; 
       }  
   }
}

/*------------------------------------------------------------------
 * Function:       Pth_mat_vect
 * Purpose:        Multiply an mxn matrix by an nx1 column vector
 * In arg:         rank
 * Global in vars: A, x, m, n, thread_count
 * Global out var: y
 */
void *Pth_mat_vect(void* rank) {
   long my_rank = (long) rank;
   int i, j;
   int local_m = m/thread_count; 
   int my_first_row = my_rank*local_m; 
   int my_last_row = (my_rank+1)*local_m - 1;

   for (i = my_first_row; i <= my_last_row; i++) {
      y[i] = 0.0f;
      for (j = 0; j < n; j++)
          y[i] += A[i*n+j]*x[j];
   }

   return NULL;
}  /* Pth_mat_vect */


/*------------------------------------------------------------------
 * Function:       Omp_mat_vect
 * Purpose:        Multiply an mxn matrix by an nx1 column vector using OpenMP
 * Global in vars: A, x, m, n, thread_count
 * Global out var: y
 */
void Omp_mat_vect(int thread_count, csr_matrix *csr_A) {
   int i, j;
   #pragma omp parallel for num_threads(thread_count) \
      default(none) shared(csr_A, x, y, m, n) private(i, j)
   for (i = 0; i < m; i++) {
     // printf("Thread %d processing row %d\n", omp_get_thread_num(), i);
      y[i] = 0.0f;
      for (j = 0; j < n; j++)
         y[i] += csr_A->values[csr_A->row_ptr[i] + j] * x[csr_A->col_ind[csr_A->row_ptr[i] + j]];
   }
}  /* Omp_mat_vect */


/*------------------------------------------------------------------
 * Function:    Print_matrix
 * Purpose:     Print the matrix
 * In args:     title, A, m, n
 */
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

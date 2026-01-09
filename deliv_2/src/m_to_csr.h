#ifndef M_TO_CSR_H
#define M_TO_CSR_H

#include <stddef.h>   /* size_t */
#include <errno.h> 

typedef struct {
    int rows;
    int cols;
    long long nnz;   // Use long long to support matrices with >2.1B non-zeros
    int *row_ptr;    // size rows + 1
    int *col_ind;    // size nnz
    float *values;     // size nnz
} csr_matrix;

int matrix_to_csr(
    float *m,          /* input dense matrix, row-major, size rows*cols */
    int rows,
    int cols,
    csr_matrix **out_csr
);

/* NEW: Direct CSR import from Matrix Market file */
int import_matrix_to_csr(
    const char *filename,
    csr_matrix **out_csr
);

/* Import only specific rows from Matrix Market file to CSR format */
int import_matrix_rows_to_csr(
    const char *filename,
    int row_start,
    int row_end,
    int global_cols,
    csr_matrix **out_csr
);

/* MPI-based matrix distribution: rank 0 reads, broadcasts to all ranks
   Returns dimensions via output parameters to avoid redundant file reads */
#ifdef MPI_ENABLED
int import_matrix_distribute_mpi(
    const char *filename,
    int row_start,
    int row_end,
    int *global_rows,
    int *global_cols,
    csr_matrix **out_csr
);
#endif

void print_csr(csr_matrix *csr);
void csr_free(csr_matrix *A);
#endif

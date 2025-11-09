#ifndef M_TO_CSR_H
#define M_TO_CSR_H

#include <stddef.h>   /* size_t */
#include <errno.h> 

typedef struct {
    int rows;
    int cols;
    int nnz;
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
void print_csr(csr_matrix *csr);
void csr_free(csr_matrix *A);
#endif

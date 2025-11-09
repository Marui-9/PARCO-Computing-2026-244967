#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include "m_to_csr.h"

#define ROWS 8
#define COLS 8

void print_csr(csr_matrix *csr) {
    if (!csr) {
        printf("CSR matrix is NULL\n");
        return;
    }
    printf("CSR Matrix: %d rows, %d cols, %d non-zero entries\n", csr->rows, csr->cols, csr->nnz);
    printf("Row Pointer: ");
    for (int i = 0; i <= csr->rows; ++i) {
        printf("%d ", csr->row_ptr[i]);
    }
    printf("\nColumn Indices: ");
    for (int i = 0; i < csr->nnz; ++i) {
        printf("%d ", csr->col_ind[i]);
    }
    printf("\nValues: ");
    for (int i = 0; i < csr->nnz; ++i) {
        printf("%f ", csr->values[i]);
    }
    printf("\n");
}
void csr_free(csr_matrix *csr) {
    if (csr) {
        free(csr->row_ptr);
        free(csr->col_ind);
        free(csr->values);
        free(csr);
    }
}
int matrix_to_csr(
    float *m,          /* input dense matrix, row-major, size rows*cols */
    int rows,
    int cols,
    csr_matrix **out_csr
) {
    clock_t start, end;
    double elapsed_seconds;
    start = clock();
    printf("Converting to CSR...\n");
    if (!m || rows <= 0 || cols <= 0 )//|| !out_row_ptr || !out_col_ind || !out_values) || !out_nnz)
        return EINVAL; // EINVAL = invalid argument

    // worst-case space for col_ind and values
    int *row_ptr = malloc((size_t)(rows + 1) * sizeof(int));
    int *col_ind = malloc((size_t)(rows * cols) * sizeof(int));
    float *values  = malloc((size_t)(rows * cols) * sizeof(float));
    if (!row_ptr || !col_ind || !values) {
        free(row_ptr); free(col_ind); free(values);
        return ENOMEM; // ENOMEM = out of memory
    }   

    int nnz = 0;
    row_ptr[0] =  0;
    for (int i = 0; i < rows; ++i) {
        for (int j  = 0; j < cols; ++j) {
            if (m[i * cols + j] != 0) {
                values[nnz] = m[i * cols + j];
                col_ind[nnz] = j;
                nnz++;
            }
        }
        row_ptr[i + 1] = nnz;
    }
    csr_matrix *csr = malloc(sizeof(csr_matrix));
    if (!csr) {
        free(row_ptr); free(col_ind); free(values);
        return ENOMEM;
    }
    csr->rows = rows;
    csr->cols = cols;
    csr->nnz  = nnz;
    csr->row_ptr = row_ptr;
    csr->col_ind = col_ind;
    csr->values  = values;
    *out_csr = csr;
    /*
    Shrink the column and value arrays to the actual number of non-zero entries
    int *col_shr = realloc(col_ind, (size_t)nnz * sizeof(int));
    int *val_shr = realloc(values,  (size_t)nnz * sizeof(int));
    if (col_shr) col_ind = col_shr;
    if (val_shr) values  = val_shr;
*/
    // *out_row_ptr = row_ptr;
    // *out_col_ind = col_ind;
    // *out_values  = values;
    //*out_nnz     = nnz;
    
    end = clock();
    elapsed_seconds = (double)(end - start) / CLOCKS_PER_SEC;

    printf("  Conversion Time taken: %f seconds, %d non-zero entries\n", elapsed_seconds, nnz);
	return 0;
}
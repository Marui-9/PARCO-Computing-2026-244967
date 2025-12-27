#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <string.h>
#include "m_to_csr.h"

#define ROWS 8
#define COLS 8

void print_csr(csr_matrix *csr) {
    if (!csr) {
        printf("CSR matrix is NULL\n");
        return;
    }
    printf("CSR Matrix: %d rows, %d cols, %d non-zero entries (%f per cent)\n",\
         csr->rows, csr->cols, csr->nnz, (csr->nnz / (float)(csr->rows * csr->cols)) * 100);
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

    printf("  Conversion Time taken: %f seconds, %d non-zero entries (%f per cent)\n",\
         elapsed_seconds, nnz, (nnz / (float)(rows * cols)) * 100);
	return 0;
}

/* 
 * import_matrix_to_csr: Read Matrix Market file directly into CSR format
 * This avoids allocating the huge dense matrix for large sparse matrices
 */
int import_matrix_to_csr(const char *filename, csr_matrix **out_csr) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        return ENOENT;
    }

    char line[1024];
    int is_symmetric = 0;
    int is_pattern = 0;

    // Skip comment lines and detect matrix properties
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '%') {
            if (strstr(line, "symmetric")) {
                is_symmetric = 1;
            }
            if (strstr(line, "pattern")) {
                is_pattern = 1;
            }
            continue;
        }
        break; // First non-comment line
    }

    // Read matrix dimensions
    int rows, cols, nnz_file;
    if (sscanf(line, "%d %d %d", &rows, &cols, &nnz_file) != 3) {
        fprintf(stderr, "Failed to read matrix dimensions\n");
        fclose(file);
        return EINVAL;
    }

    printf("Reading matrix: %d x %d with %d non-zeros from file\n", rows, cols, nnz_file);
    if (is_symmetric) {
        printf("  Matrix is symmetric (will expand to full storage)\n");
    }

    // For symmetric matrices, we need to count diagonal and off-diagonal separately
    int estimated_nnz = is_symmetric ? nnz_file * 2 : nnz_file;

    // Temporary storage for COO format (row, col, value)
    typedef struct {
        int row;
        int col;
        float value;
    } coo_entry;

    coo_entry *entries = malloc(estimated_nnz * sizeof(coo_entry));
    if (!entries) {
        fprintf(stderr, "Failed to allocate temporary COO storage\n");
        fclose(file);
        return ENOMEM;
    }

    // Read all entries
    int entry_count = 0;
    for (int i = 0; i < nnz_file; i++) {
        int row, col;
        float value = 1.0f;

        if (is_pattern) {
            if (fscanf(file, "%d %d", &row, &col) != 2) {
                fprintf(stderr, "Failed to read entry %d\n", i);
                free(entries);
                fclose(file);
                return EIO;
            }
        } else {
            if (fscanf(file, "%d %d %f", &row, &col, &value) != 3) {
                fprintf(stderr, "Failed to read entry %d\n", i);
                free(entries);
                fclose(file);
                return EIO;
            }
        }

        // Convert from 1-indexed to 0-indexed
        row--;
        col--;

        // Validate indices
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            fprintf(stderr, "Invalid entry: row=%d, col=%d\n", row, col);
            continue;
        }

        // Store entry
        entries[entry_count].row = row;
        entries[entry_count].col = col;
        entries[entry_count].value = value;
        entry_count++;

        // If symmetric and not diagonal, also store transpose
        if (is_symmetric && row != col) {
            if (entry_count >= estimated_nnz) {
                // Reallocate if needed
                estimated_nnz *= 2;
                coo_entry *new_entries = realloc(entries, estimated_nnz * sizeof(coo_entry));
                if (!new_entries) {
                    free(entries);
                    fclose(file);
                    return ENOMEM;
                }
                entries = new_entries;
            }
            entries[entry_count].row = col;
            entries[entry_count].col = row;
            entries[entry_count].value = value;
            entry_count++;
        }
    }
    fclose(file);

    printf("  Total entries after expansion: %d\n", entry_count);

    // Sort entries by row, then by column (simple bubble sort for now)
    // For production use a faster sort like qsort
    for (int i = 0; i < entry_count - 1; i++) {
        for (int j = 0; j < entry_count - i - 1; j++) {
            if (entries[j].row > entries[j+1].row ||
                (entries[j].row == entries[j+1].row && entries[j].col > entries[j+1].col)) {
                coo_entry temp = entries[j];
                entries[j] = entries[j+1];
                entries[j+1] = temp;
            }
        }
    }

    // Convert COO to CSR
    int *row_ptr = malloc((rows + 1) * sizeof(int));
    int *col_ind = malloc(entry_count * sizeof(int));
    float *values = malloc(entry_count * sizeof(float));

    if (!row_ptr || !col_ind || !values) {
        free(entries);
        free(row_ptr);
        free(col_ind);
        free(values);
        return ENOMEM;
    }

    // Build CSR structure
    row_ptr[0] = 0;
    int current_row = 0;
    for (int i = 0; i < entry_count; i++) {
        // Fill in row_ptr for any empty rows
        while (current_row < entries[i].row) {
            current_row++;
            row_ptr[current_row] = i;
        }
        
        col_ind[i] = entries[i].col;
        values[i] = entries[i].value;
    }
    
    // Fill remaining row_ptr entries
    while (current_row < rows) {
        current_row++;
        row_ptr[current_row] = entry_count;
    }

    free(entries);

    // Create CSR matrix structure
    csr_matrix *csr = malloc(sizeof(csr_matrix));
    if (!csr) {
        free(row_ptr);
        free(col_ind);
        free(values);
        return ENOMEM;
    }

    csr->rows = rows;
    csr->cols = cols;
    csr->nnz = entry_count;
    csr->row_ptr = row_ptr;
    csr->col_ind = col_ind;
    csr->values = values;

    *out_csr = csr;

    double density = (entry_count / (double)(rows * cols)) * 100.0;
    printf("  CSR conversion complete: %d non-zeros (%.4f%% density)\n", entry_count, density);
    
    return 0;
}
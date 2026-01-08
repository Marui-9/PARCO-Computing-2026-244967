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

// Comparison function for qsort
int compare_coo_entries(const void *a, const void *b) {
    typedef struct {
        int row;
        int col;
        float value;
    } coo_entry;
    
    const coo_entry *ea = (const coo_entry *)a;
    const coo_entry *eb = (const coo_entry *)b;
    
    if (ea->row != eb->row) return ea->row - eb->row;
    return ea->col - eb->col;
}

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
    int rows, cols;
    long long nnz_file;  // Use long long to handle large nnz values
    if (sscanf(line, "%d %d %lld", &rows, &cols, &nnz_file) != 3) {
        fprintf(stderr, "Failed to read matrix dimensions\n");
        fclose(file);
        return EINVAL;
    }

    printf("Reading matrix: %d x %d with %lld non-zeros from file\n", rows, cols, nnz_file);
    if (is_symmetric) {
        printf("  Matrix is symmetric (will expand to full storage)\n");
    }

    // For symmetric matrices, we need to count diagonal and off-diagonal separately
    long long estimated_nnz = is_symmetric ? nnz_file * 2 : nnz_file;

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
    long long entry_count = 0;  // Use long long to count entries
    for (long long i = 0; i < nnz_file; i++) {
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

    printf("  Total entries after expansion: %lld\n", entry_count);

    // Sort entries by row, then by column using qsort (O(n log n) instead of O(n²))
    qsort(entries, entry_count, sizeof(coo_entry), compare_coo_entries);

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
    for (long long i = 0; i < entry_count; i++) {
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
    printf("  CSR conversion complete: %lld non-zeros (%.4f%% density)\n", entry_count, density);
    
    return 0;
}

/* Import specific rows [row_start, row_end) from Matrix Market file to CSR */
int import_matrix_rows_to_csr(
    const char *filename,
    int row_start,
    int row_end,
    int global_cols,
    csr_matrix **out_csr
) {
    if (!filename || !out_csr || row_start < 0 || row_end < row_start) return EINVAL;
    
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return ENOENT;
    }
    
    /* Skip comments and read matrix header */
    char line[256];
    int global_rows = 0, cols = 0, total_nnz = 0;
    
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '%') continue;
        if (sscanf(line, "%d %d %d", &global_rows, &cols, &total_nnz) == 3) break;
    }
    
    if (global_rows <= 0 || cols <= 0 || total_nnz <= 0) {
        fprintf(stderr, "Error: Invalid matrix market header\n");
        fclose(fp);
        return EINVAL;
    }
    
    if (global_cols != cols) {
        fprintf(stderr, "Error: Column mismatch (%d vs %d)\n", global_cols, cols);
        fclose(fp);
        return EINVAL;
    }
    
    int local_rows = row_end - row_start;
    
    /* First pass: count NNZ in our row range */
    int nnz_local = 0;
    while (fgets(line, sizeof(line), fp)) {
        int r, c;
        float val;
        if (sscanf(line, "%d %d %f", &r, &c, &val) == 3) {
            r--;  /* Convert from 1-indexed to 0-indexed */
            if (r >= row_start && r < row_end) {
                nnz_local++;
            }
        }
    }
    
    /* Pre-allocate exact amount needed for local rows only */
    int *row_counts = (int *)calloc(local_rows, sizeof(int));
    int *row_indices = (int *)malloc(nnz_local * sizeof(int));
    int *col_indices = (int *)malloc(nnz_local * sizeof(int));
    float *values = (float *)malloc(nnz_local * sizeof(float));
    
    if (!row_counts || !row_indices || !col_indices || !values) {
        free(row_counts);
        free(row_indices);
        free(col_indices);
        free(values);
        fclose(fp);
        return ENOMEM;
    }
    
    /* Second pass: read entries, keep only those in [row_start, row_end) */
    rewind(fp);
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '%') continue;
        int test_rows, test_cols, test_nnz;
        if (sscanf(line, "%d %d %d", &test_rows, &test_cols, &test_nnz) == 3) break;
    }
    
    nnz_local = 0;
    while (fgets(line, sizeof(line), fp)) {
        int r, c;
        float val;
        if (sscanf(line, "%d %d %f", &r, &c, &val) == 3) {
            r--;  /* Convert from 1-indexed to 0-indexed */
            c--;
            if (r >= row_start && r < row_end && c < cols) {
                row_indices[nnz_local] = r - row_start;  /* Local row index */
                col_indices[nnz_local] = c;
                values[nnz_local] = val;
                row_counts[r - row_start]++;
                nnz_local++;
            }
        }
    }
    fclose(fp);
    
    /* Build CSR row_ptr */
    int *row_ptr = (int *)malloc((local_rows + 1) * sizeof(int));
    if (!row_ptr) {
        free(row_counts);
        free(row_indices);
        free(col_indices);
        free(values);
        return ENOMEM;
    }
    
    row_ptr[0] = 0;
    for (int i = 0; i < local_rows; i++) {
        row_ptr[i + 1] = row_ptr[i] + row_counts[i];
    }
    free(row_counts);
    
    /* Sort entries by row and fill CSR arrays */
    int *csr_col_ind = (int *)malloc(nnz_local * sizeof(int));
    float *csr_values = (float *)malloc(nnz_local * sizeof(float));
    int *entry_pos = (int *)calloc(local_rows, sizeof(int));
    
    if (!csr_col_ind || !csr_values || !entry_pos) {
        free(row_ptr);
        free(row_indices);
        free(col_indices);
        free(values);
        free(csr_col_ind);
        free(csr_values);
        free(entry_pos);
        return ENOMEM;
    }
    
    for (int i = 0; i < nnz_local; i++) {
        int r = row_indices[i];
        int pos = row_ptr[r] + entry_pos[r];
        csr_col_ind[pos] = col_indices[i];
        csr_values[pos] = values[i];
        entry_pos[r]++;
    }
    
    free(row_indices);
    free(col_indices);
    free(values);
    free(entry_pos);
    
    /* Allocate and fill CSR structure */
    csr_matrix *csr = (csr_matrix *)malloc(sizeof(csr_matrix));
    if (!csr) {
        free(row_ptr);
        free(csr_col_ind);
        free(csr_values);
        return ENOMEM;
    }
    
    csr->rows = local_rows;
    csr->cols = global_cols;
    csr->nnz = nnz_local;
    csr->row_ptr = row_ptr;
    csr->col_ind = csr_col_ind;
    csr->values = csr_values;
    
    *out_csr = csr;
    
    return 0;
}
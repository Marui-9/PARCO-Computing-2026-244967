#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>


float * import_matrix(const char *filename,  
            int *out_rows, 
            int *out_cols);
float * generate_matrix(int rows, int cols, int percent_nonzero);
float * generate_vector(int size);


float * import_matrix(const char *filename,  
            int *out_rows, 
            int *out_cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    char line[1024];
    int is_symmetric = 0;
    int is_pattern = 0;

    // Skip comment lines and detect matrix properties
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '%') {
            // Check for symmetric and pattern properties
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
    int rows, cols, nnz;
    if (sscanf(line, "%d %d %d", &rows, &cols, &nnz) != 3) {
        fprintf(stderr, "Failed to read matrix dimensions\n");
        fclose(file);
        return NULL;
    }

    *out_rows = rows;
    *out_cols = cols;

    // Allocate matrix in row-major format
    size_t total_size = (size_t)rows * (size_t)cols;
    float *matrix = calloc(total_size, sizeof(float));
    if (!matrix) {
        fprintf(stderr, "Allocation failed for matrix\n");
        fclose(file);
        return NULL;
    }

    // Read matrix entries
    for (int i = 0; i < nnz; i++) {
        int row, col;
        float value = 1.0f;

        if (is_pattern) {
            if (fscanf(file, "%d %d", &row, &col) != 2) {
                fprintf(stderr, "Failed to read matrix entry %d\n", i);
                free(matrix);
                fclose(file);
                return NULL;
            }
        } else {
            if (fscanf(file, "%d %d %f", &row, &col, &value) != 3) {
                fprintf(stderr, "Failed to read matrix entry %d\n", i);
                free(matrix);
                fclose(file);
                return NULL;
            }
        }

        // Convert from 1-indexed to 0-indexed
        row--;
        col--;

        // Store value
        if (row >= 0 && row < rows && col >= 0 && col < cols) {
            matrix[row * cols + col] = value;
            
            // If symmetric and not on diagonal, also store transpose
            if (is_symmetric && row != col) {
                matrix[col * rows + row] = value;
            }
        }
    }

    fclose(file);
    return matrix;
}
float * generate_matrix(int rows, int cols, int percent_nonzero) {
    if (rows <= 0 || cols <= 0 || percent_nonzero < 0 || percent_nonzero > 100)
        return NULL;

    size_t total_size = (size_t)rows * (size_t)cols;
    float *matrix = malloc(total_size * sizeof(float));
    if (!matrix) {
        fprintf(stderr, "Allocation failed. Needed ~%zu MB for matrix alone.\n",
                total_size * sizeof(float) / (1024 * 1024));
        return NULL;
    }

    // Fill matrix
    for (size_t i = 0; i < total_size; ++i) {
        float val = rand() % 100;
        matrix[i] = (val < percent_nonzero) ? (float)(val + 1) : 0.0f;
    }

    return matrix;
}
float * generate_vector(int size) {
    float *vector = (float *)malloc((size_t)size * sizeof(float));
    if (!vector) {
        fprintf(stderr, "Allocation failed. Needed ~%zu MB for vector alone.\n",
                ((size_t)size * sizeof(float)) / (1024 * 1024));
        return NULL;
    }

    srand(time(NULL));
    #pragma omp simd
    for (int i = 0; i < size; i++) {
        vector[i] = (float)(rand() % 100);
    }
    return vector;
}
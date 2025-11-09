#ifndef GENERATOR_H
#define GENERATOR_H

float * import_matrix(const char *filename, int *out_rows, int *out_cols);
float * generate_matrix(int rows, int cols, int percent_nonzero);
float * generate_vector(int size);

#endif
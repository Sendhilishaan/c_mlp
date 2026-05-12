#ifndef MATRIX_H
#define MATRIX_H
#define AT(M, i, j) ((M)->data[(i) * (M)->cols + (j)])

typedef struct {
    size_t rows;
    size_t cols;
    double* data; // row-major order (row * num_cols + col)
} Matrix;

Matrix* matrix_create(size_t rows, size_t cols); // 0 init matrix on heap

void matrix_free(Matrix* M);

void matrix_print(Matrix* M);

Matrix* matrix_matmul(Matrix* A, Matrix* B);
#endif
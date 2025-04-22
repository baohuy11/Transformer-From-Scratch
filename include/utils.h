#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// FUNCTION TO COMPUTE THE DOT PRODUCT OF TWO VECTORS (FLOAT)
float dot_product_float(float *a, float *b, int dim);

// FUNCTION TO COMPUTE THE DOT PRODUCT OF TWO MATRICES (DOUBLE)
void dot_product_double(double A[][2], double B[][2], double result[][2]);

// FUNCTION TO APPLY SOFTMAX TO A VECTOR OF SCORES (FLOAT)
void softmax_float(float *input, float *output, int length);

// FUNCTION TO APPLY SOFTMAX TO A MATRIX (DOUBLE)
void softmax_double(double matrix[][2]);

// FUNCTION TO MULTIPLY TWO MATRICES (FLOAT)
void matrix_multiply_float(float *A, float *B, float *C, int rows_A, int cols_A, int cols_B);

// FUNCTION TO MULTIPLY TWO MATRICES (DOUBLE)
void matrix_multiply_double(double A[][2], double B[][2], double result[][2], int rowsA, int colsA, int colsB);

#endif // UTILS_H 
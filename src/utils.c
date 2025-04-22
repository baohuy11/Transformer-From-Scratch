#include "../include/utils.h"
#include <assert.h>

// FUNCTION TO COMPUTE THE DOT PRODUCT OF TWO VECTORS (FLOAT)
float dot_product_float(float *a, float *b, int dim) {
    assert(a != NULL && b != NULL && dim > 0);
    float result = 0.0f;
    for(int i = 0; i < dim; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// FUNCTION TO COMPUTE THE DOT PRODUCT OF TWO MATRICES (DOUBLE)
void dot_product_double(double A[][2], double B[][2], double result[][2]) {
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            result[i][j] = 0;
            for(int k = 0; k < 2; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// FUNCTION TO APPLY SOFTMAX TO A VECTOR OF SCORES (FLOAT)
void softmax_float(float *input, float *output, int length) {
    assert(input != NULL && output != NULL && length > 0);
    
    // Find maximum value for numerical stability
    float max_val = input[0];
    for(int i = 1; i < length; i++) {
        if(input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Compute exponentials and sum
    float sum = 0.0f;
    for(int i = 0; i < length; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }

    // Normalize to get probabilities
    for(int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

// FUNCTION TO APPLY SOFTMAX TO A MATRIX (DOUBLE)
void softmax_double(double matrix[][2]) {
    for(int i = 0; i < 2; i++) {
        double sum_exp = 0.0;
        for(int j = 0; j < 2; j++) {
            sum_exp += exp(matrix[i][j]);
        }
        for(int j = 0; j < 2; j++) {
            matrix[i][j] = exp(matrix[i][j]) / sum_exp;
        }
    }
}

// FUNCTION TO MULTIPLY TWO MATRICES (FLOAT)
void matrix_multiply_float(float *A, float *B, float *C, int rows_A, int cols_A, int cols_B) {
    for(int i = 0; i < rows_A; i++) {
        for(int j = 0; j < cols_B; j++) {
            C[i * cols_B + j] = 0.0f;
            for(int k = 0; k < cols_A; k++) {
                C[i * cols_B + j] += A[i * cols_A + k] * B[k * cols_B + j];
            }
        }
    }
}

// FUNCTION TO MULTIPLY TWO MATRICES (DOUBLE)
void matrix_multiply_double(double A[][2], double B[][2], double result[][2], int rowsA, int colsA, int colsB) {
    for(int i = 0; i < rowsA; i++) {
        for(int j = 0; j < colsB; j++) {
            result[i][j] = 0;
            for(int k = 0; k < colsA; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
} 
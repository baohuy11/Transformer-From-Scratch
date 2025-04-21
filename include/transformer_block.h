#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include <math.h>
#include <stdlib.h>

// FUNCTION TO COMPUTE POSITIONAL ENCODING
double* positional_encoding(int index, int vector_size);

#define MATRIX_SIZE 2
#define EMBEDDING_DIM 2

// DEFINE MATRICES
extern double k_matrix[MATRIX_SIZE][MATRIX_SIZE];
extern double q_matrix[MATRIX_SIZE][MATRIX_SIZE];
extern double v_matrix[MATRIX_SIZE][MATRIX_SIZE];

// FUNCTION TO READ A SINGLE VALUE FROM A FILE
double read_single_value_from_file(const char* filename);

// FUNCTION TO INITIALIZE MATRICES FROM FILES
void initialize_matrices_from_files(void);

// FUNCTION TO PRINT A MATRIX
void print_matrix(const char* name, double matrix[MATRIX_SIZE][MATRIX_SIZE]);

// FUNCTION TO COMPUTE THE DOT PRODUCT BETWEEN TWO MATRICES
void dot_product(double A[MATRIX_SIZE][MATRIX_SIZE], double B[MATRIX_SIZE][MATRIX_SIZE], double result[MATRIX_SIZE][MATRIX_SIZE]);

// FUNCTION TO TRANSPOSE A MATRIX
void transpose(double matrix[MATRIX_SIZE][MATRIX_SIZE], double transposed[MATRIX_SIZE][MATRIX_SIZE]);

// FUNCTION TO MULTIPLY TWO MATRICES
void matrix_multiply(double A[][MATRIX_SIZE], double B[][MATRIX_SIZE], double result[][MATRIX_SIZE], int rowsA, int colsA, int colsB);

// FUNCTION TO APPLY SOFTMAX TO A MATRIX
void apply_softmax(double matrix[][MATRIX_SIZE], int rows, int cols);

// FUNCTION TO COMPUTE SELF ATTENTION
void compute_self_attention(float embedding_matrix[][MATRIX_SIZE], double k_matrix[MATRIX_SIZE][MATRIX_SIZE], double q_matrix[MATRIX_SIZE][MATRIX_SIZE], double v_matrix[MATRIX_SIZE][MATRIX_SIZE], int length, double self_attention_matrix[][MATRIX_SIZE]);

// FUNCTION TO ADD TWO MATRICES
void add_matrices(float matrix1[][MATRIX_SIZE], double matrix2[][MATRIX_SIZE], double result_matrix[][MATRIX_SIZE], int rows, int cols);

// FUNCTION TO UPDATE ATTENTION MATRICES
void update_attention_matrices(double loss, double learning_rate);

// FUNCTION TO CLIP GRADIENT
double clip_gradient_transformer(double gradient);

#endif // TRANSFORMER_BLOCK_H
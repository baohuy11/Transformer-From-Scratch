#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include <math.h>
#include <stdlib.h>

/**
 * Compute the positional encoding vector for a given index.
 *
 * @param index The position in the sequence.
 * @param vector_size The size of the encoding vector.
 * @return A pointer to a dynamically allocated array of doubles representing the positional encoding.
 *         (Remember to free the allocated memory when no longer needed.)
 */
double* positional_encoding(int index, int vector_size);

#define MATRIX_SIZE 2
#define EMBEDDING_DIM 2

/* The weight matrices used for key, query, and value transformations. */
extern double k_matrix[MATRIX_SIZE][MATRIX_SIZE];
extern double q_matrix[MATRIX_SIZE][MATRIX_SIZE];
extern double v_matrix[MATRIX_SIZE][MATRIX_SIZE];

/**
 * Read a single double value from a file.
 *
 * @param filename The path to the file.
 * @return The double value read from the file.
 */
double read_single_value_from_file(const char* filename);

/**
 * Initialize the self-attention weight matrices with weights stored in files.
 */
void initialize_matrices_from_files(void);

/**
 * Print the specified matrix with a name label.
 *
 * @param name The label for the matrix.
 * @param matrix The matrix to be printed.
 */
void print_matrix(const char* name, double matrix[MATRIX_SIZE][MATRIX_SIZE]);

/**
 * Compute the dot product between two matrices.
 *
 * @param A The first matrix.
 * @param B The second matrix.
 * @param result The resulting matrix where the dot product is stored.
 */
void dot_product(double A[MATRIX_SIZE][MATRIX_SIZE], double B[MATRIX_SIZE][MATRIX_SIZE], double result[MATRIX_SIZE][MATRIX_SIZE]);

/**
 * Transpose a given matrix.
 *
 * @param matrix The original matrix.
 * @param transposed The resulting transposed matrix.
 */
void transpose(double matrix[MATRIX_SIZE][MATRIX_SIZE], double transposed[MATRIX_SIZE][MATRIX_SIZE]);

/**
 * Multiply two matrices.
 *
 * @param A The first matrix with dimensions (rowsA x colsA).
 * @param B The second matrix with dimensions (colsA x colsB).
 * @param result The resulting matrix with dimensions (rowsA x colsB).
 * @param rowsA Number of rows in matrix A.
 * @param colsA Number of columns in matrix A (and rows in matrix B).
 * @param colsB Number of columns in matrix B.
 */
void matrix_multiply(double A[][MATRIX_SIZE], double B[][MATRIX_SIZE], double result[][MATRIX_SIZE], int rowsA, int colsA, int colsB);

/**
 * Apply the softmax function to each row of the given matrix.
 *
 * @param matrix The input matrix.
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 */
void apply_softmax(double matrix[MATRIX_SIZE][MATRIX_SIZE]);

/**
 * Compute the self-attention output given an input embedding matrix and weight matrices.
 *
 * @param embedding_matrix The input embedding matrix (as float values).
 * @param k_matrix The key weight matrix.
 * @param q_matrix The query weight matrix.
 * @param v_matrix The value weight matrix.
 * @param length The length/number of rows in the embedding matrix.
 * @param self_attention_matrix The resulting self-attention output matrix.
 */
void compute_self_attention(float embedding_matrix[][MATRIX_SIZE], double k_matrix[MATRIX_SIZE][MATRIX_SIZE], double q_matrix[MATRIX_SIZE][MATRIX_SIZE], double v_matrix[MATRIX_SIZE][MATRIX_SIZE], int length, double self_attention_matrix[][MATRIX_SIZE]);

/**
 * Add two matrices (one float and one double) element-wise, storing the result in a double matrix.
 *
 * @param matrix1 The first matrix (float).
 * @param matrix2 The second matrix (double).
 * @param result_matrix The resulting matrix (double).
 * @param rows The number of rows.
 * @param cols The number of columns.
 */
void add_matrices(float matrix1[][MATRIX_SIZE], double matrix2[][MATRIX_SIZE], double result_matrix[][MATRIX_SIZE], int rows, int cols);

/**
 * Update the attention weight matrices based on the loss and learning rate.
 *
 * @param loss The computed loss.
 * @param learning_rate The learning rate to be applied.
 */
void update_attention_matrices(double loss, double learning_rate);

/**
 * Clip the gradient to prevent exploding gradients.
 *
 * @param gradient The gradient value.
 * @return The clipped gradient.
 */
double clip_gradient_transformer(double gradient);

#endif // TRANSFORMER_BLOCK_H
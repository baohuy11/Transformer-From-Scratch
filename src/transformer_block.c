#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "../include/transformer_block.h"
#include "../include/utils.h"

// FUNCTION IMPLEMENTATION FOR POSITIONAL ENCODING
double* positional_encoding(int index, int vector_size) {
    // ALLOCATE MEMORY FOR THE RETURN ARRAY
    double* encoding = (double*) malloc(vector_size*sizeof(double));
    // CALCULATE EACH VALUE IN THE ENCODING VECTOR
    for(int i = 0; i < vector_size; i++) {
        if(i % 2 == 0) {
            // EVEN INDEX
            encoding[i] = sin(index / pow(10000.0, (double)i / vector_size));
        } else {
            // ODD INDEX
            encoding[i] = cos(index / pow(10000.0, (double)(i - 1) / vector_size));
        }
    }
    return encoding;
}

/////////////////////// SELF ATTENTION ////////////////////////////

#define MATRIX_SIZE 2
#define EMBEDDING_DIM 2
#define MAX_SENTENCE_LENGTH 512
#define CLIP_THRESHOLD 100

// DEFINE MATRICES
double k_matrix[MATRIX_SIZE][MATRIX_SIZE];
double q_matrix[MATRIX_SIZE][MATRIX_SIZE];
double v_matrix[MATRIX_SIZE][MATRIX_SIZE];

// FUNCTION TO READ A SINGLE VALUE FROM A FILE
double read_single_value_from_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    printf("filename: %s \n", filename);
    if(file == NULL) {
        perror("Error opening file");
        return 0.0;
    }
    double value;
    if(fscanf(file, "%lf", &value) != 1) {
        perror("Error reading file");
        fclose(file);
        return 0.0;
    }
    fclose(file);
    return value;
}

// FUNCTION TO INITIALIZE MATRICES FROM FILES
void initialize_matrices_from_files() {
    int index = 0;
    // READ KEY MATRICES
    for(int i = 0; i < MATRIX_SIZE; i++) {
        for(int j = 0; j < MATRIX_SIZE; j++) {
            char filename[256];
            printf("key_weight_%d.txt \n", index + 1);
            snprintf(filename, sizeof(filename), "Model Trained Weights/self-attention-block-weights/key_weight_%d.txt", index + 1);
            k_matrix[i][j] = read_single_value_from_file(filename);
            index++;
        }
    }
    printf("Initialized KEY MATRIX\n");
    index = 0;
    // READ QUERY MATRICES
    for(int i = 0; i < MATRIX_SIZE; i++) {
        for(int j = 0; j < MATRIX_SIZE; j++) {
            char filename[256];
            printf("query_weight_%d.txt \n", index + 1);
            snprintf(filename, sizeof(filename), "Model Trained Weights/self-attention-block-weights/query_weight_%d.txt", index + 1);
            q_matrix[i][j] = read_single_value_from_file(filename);
            index++;
        }
    }
    printf("Initialized QUERY MATRIX\n");
    index = 0;
    // READ VALUE MATRICES
    for(int i = 0; i < MATRIX_SIZE; i++) {
        for(int j = 0; j < MATRIX_SIZE; j++) {
            char filename[256];
            printf("value_weight_%d.txt \n", index + 1);
            snprintf(filename, sizeof(filename), "Model Trained Weights/self-attention-block-weights/value_weight_%d.txt", index + 1);
            v_matrix[i][j] = read_single_value_from_file(filename);
            index++;
        }
    }
    printf("Initialized VALUE MATRIX\n");
}

// FUNCTION TO PRINT THE MATRICES
void print_matrix(const char* name, double matrix[MATRIX_SIZE][MATRIX_SIZE]) {
    printf("%s:\n", name);
    for(int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// FUNCTION TO TRANSPOSE A MATRIX
void transpose(double matrix[MATRIX_SIZE][MATRIX_SIZE], double transposed[MATRIX_SIZE][MATRIX_SIZE]) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            transposed[j][i] = matrix[i][j];
        }
    }
}

// FUNCTION TO CALCULATE ATTENTION SCORES
void calculate_attention(double Q[MATRIX_SIZE][MATRIX_SIZE], double K[MATRIX_SIZE][MATRIX_SIZE], double V[MATRIX_SIZE][MATRIX_SIZE], double result[MATRIX_SIZE][MATRIX_SIZE]) {
    double K_transposed[MATRIX_SIZE][MATRIX_SIZE];
    double QK_product[MATRIX_SIZE][MATRIX_SIZE];

    // TRANSPOSE THE K MATRIX
    transpose(K, K_transposed);

    // COMPUTE QK^T
    dot_product_double(Q, K_transposed, QK_product);

    // SCALE BY 1 / SQRT(d_k)
    double scale_factor = 1.0 / sqrt((double)MATRIX_SIZE);
    for(int i = 0; i < MATRIX_SIZE; i++) {
        for(int j = 0; j < MATRIX_SIZE; j++) {
            QK_product[i][j] *= scale_factor;
        }
    }

    // APPLY SOFTMAX TO THE RESULTING MATRIX
    softmax_double(QK_product);

    // COMPUTE FINAL ATTENTION OUTPUT: SOFTMAX(QK^T) * V
    dot_product_double(QK_product, V, result);
}

// FUNCTION TO COMPUTE SELF ATTENTION
void compute_self_attention(float embedding_matrix[][MATRIX_SIZE], double k_matrix[MATRIX_SIZE][MATRIX_SIZE], double q_matrix[MATRIX_SIZE][MATRIX_SIZE], double v_matrix[MATRIX_SIZE][MATRIX_SIZE], int length, double self_attention_matrix[][MATRIX_SIZE]) {
    double attention_scores[MATRIX_SIZE][MATRIX_SIZE];
    calculate_attention(q_matrix, k_matrix, v_matrix, attention_scores);
    
    for(int i = 0; i < length; i++) {
        for(int j = 0; j < MATRIX_SIZE; j++) {
            self_attention_matrix[i][j] = attention_scores[i][j];
        }
    }
}

// FUNCTION TO ADD TWO MATRICES
void add_matrices(float matrix1[][MATRIX_SIZE], double matrix2[][MATRIX_SIZE], double result_matrix[][MATRIX_SIZE], int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            result_matrix[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}

// FUNCTION TO UPDATE ATTENTION MATRICES
void update_attention_matrices(double loss, double learning_rate) {
    for(int i = 0; i < MATRIX_SIZE; i++) {
        for(int j = 0; j < MATRIX_SIZE; j++) {
            double gradient = clip_gradient_transformer(loss * learning_rate);
            k_matrix[i][j] -= gradient;
            q_matrix[i][j] -= gradient;
            v_matrix[i][j] -= gradient;
        }
    }
}

// FUNCTION TO CLIP GRADIENT
double clip_gradient_transformer(double gradient) {
    if(gradient > CLIP_THRESHOLD) return CLIP_THRESHOLD;
    if(gradient < -CLIP_THRESHOLD) return -CLIP_THRESHOLD;
    return gradient;
}
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/transformer_block.h"

// Helper function to compare two matrices
int compare_matrices(double A[MATRIX_SIZE][MATRIX_SIZE], double B[MATRIX_SIZE][MATRIX_SIZE]) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            if (fabs(A[i][j] - B[i][j]) > 1e-6) {
                return 0;
            }
        }
    }
    return 1;
}

// Test positional encoding
void test_positional_encoding() {
    printf("Testing positional encoding...\n");
    int index = 0;
    int vector_size = 4;
    double* encoding = positional_encoding(index, vector_size);
    
    // Check if encoding is not NULL
    if (encoding == NULL) {
        printf("FAIL: Positional encoding returned NULL\n");
        return;
    }
    
    // Check if values are within expected range
    for (int i = 0; i < vector_size; i++) {
        if (fabs(encoding[i]) > 1.0) {
            printf("FAIL: Positional encoding value out of range at index %d\n", i);
            free(encoding);
            return;
        }
    }
    
    printf("PASS: Positional encoding test\n");
    free(encoding);
}

// Test matrix operations
void test_matrix_operations() {
    printf("Testing matrix operations...\n");
    
    // Test matrices
    double A[MATRIX_SIZE][MATRIX_SIZE] = {{1.0, 2.0}, {3.0, 4.0}};
    double B[MATRIX_SIZE][MATRIX_SIZE] = {{5.0, 6.0}, {7.0, 8.0}};
    double result[MATRIX_SIZE][MATRIX_SIZE];
    double expected[MATRIX_SIZE][MATRIX_SIZE] = {{19.0, 22.0}, {43.0, 50.0}};
    
    // Test dot product
    dot_product(A, B, result);
    if (!compare_matrices(result, expected)) {
        printf("FAIL: Matrix dot product test\n");
        return;
    }
    
    // Test transpose
    double transposed[MATRIX_SIZE][MATRIX_SIZE];
    double expected_transposed[MATRIX_SIZE][MATRIX_SIZE] = {{1.0, 3.0}, {2.0, 4.0}};
    transpose(A, transposed);
    if (!compare_matrices(transposed, expected_transposed)) {
        printf("FAIL: Matrix transpose test\n");
        return;
    }
    
    printf("PASS: Matrix operations test\n");
}

// Test attention mechanism
void test_attention() {
    printf("Testing attention mechanism...\n");
    
    // Initialize test matrices
    float embedding[MATRIX_SIZE][MATRIX_SIZE] = {{1.0, 2.0}, {3.0, 4.0}};
    double self_attention_result[MATRIX_SIZE][MATRIX_SIZE];
    
    // Initialize attention matrices with test values
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            k_matrix[i][j] = 0.5;
            q_matrix[i][j] = 0.5;
            v_matrix[i][j] = 0.5;
        }
    }
    
    // Compute self attention
    compute_self_attention(embedding, k_matrix, q_matrix, v_matrix, MATRIX_SIZE, self_attention_result);
    
    // Check if result matrix is not all zeros
    int all_zeros = 1;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            if (fabs(self_attention_result[i][j]) > 1e-6) {
                all_zeros = 0;
                break;
            }
        }
        if (!all_zeros) break;
    }
    
    if (all_zeros) {
        printf("FAIL: Self attention computation resulted in zero matrix\n");
        return;
    }
    
    printf("PASS: Attention mechanism test\n");
}

// Test gradient clipping
void test_gradient_clipping() {
    printf("Testing gradient clipping...\n");
    
    double large_gradient = 150.0;
    double clipped = clip_gradient_transformer(large_gradient);
    
    if (fabs(clipped) > CLIP_THRESHOLD) {
        printf("FAIL: Gradient not properly clipped\n");
        return;
    }
    
    double small_gradient = 50.0;
    clipped = clip_gradient_transformer(small_gradient);
    
    if (fabs(clipped - small_gradient) > 1e-6) {
        printf("FAIL: Small gradient was incorrectly clipped\n");
        return;
    }
    
    printf("PASS: Gradient clipping test\n");
}

int main() {
    printf("Starting Transformer Block Tests...\n\n");
    
    test_positional_encoding();
    test_matrix_operations();
    test_attention();
    test_gradient_clipping();
    
    printf("\nAll tests completed.\n");
    return 0;
}

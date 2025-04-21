#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "../include/transformer_block.h"

// Helper function to compare two matrices
int compare_matrices(double A[MATRIX_SIZE][MATRIX_SIZE], double B[MATRIX_SIZE][MATRIX_SIZE], double tolerance){
    for(int i = 0; i < MATRIX_SIZE; i++){
        for(int j = 0; j < MATRIX_SIZE; j++){
            if(fabs(A[i][j] - B[i][j]) > tolerance){
                return 0;
            }
        }
    }
    return 1;
}

// Test positional encoding
void test_positional_encoding(){
    printf("Testing positional encoding...\n");
    
    int index = 0;
    int vector_size = 2;
    double* encoding = positional_encoding(index, vector_size);
    
    // Test basic properties
    assert(encoding != NULL);
    assert(vector_size == 2);
    
    // Test values are within expected range
    for(int i = 0; i < vector_size; i++){
        assert(encoding[i] >= -1.0 && encoding[i] <= 1.0);
    }
    
    free(encoding);
    printf("Positional encoding test passed!\n");
}

// Test matrix operations
void test_matrix_operations(){
    printf("Testing matrix operations...\n");
    
    // Test matrices
    double A[MATRIX_SIZE][MATRIX_SIZE] = {{1.0, 2.0}, {3.0, 4.0}};
    double B[MATRIX_SIZE][MATRIX_SIZE] = {{5.0, 6.0}, {7.0, 8.0}};
    double result[MATRIX_SIZE][MATRIX_SIZE];
    
    // Test dot product
    dot_product(A, B, result);
    double expected_dot[MATRIX_SIZE][MATRIX_SIZE] = {{19.0, 22.0}, {43.0, 50.0}};
    assert(compare_matrices(result, expected_dot, 0.0001));
    
    // Test transpose
    double transposed[MATRIX_SIZE][MATRIX_SIZE];
    transpose(A, transposed);
    double expected_transpose[MATRIX_SIZE][MATRIX_SIZE] = {{1.0, 3.0}, {2.0, 4.0}};
    assert(compare_matrices(transposed, expected_transpose, 0.0001));
    
    printf("Matrix operations test passed!\n");
}

// Test self-attention computation
void test_self_attention(){
    printf("Testing self-attention computation...\n");
    
    // Initialize test matrices
    float embedding_matrix[MATRIX_SIZE][MATRIX_SIZE] = {{1.0, 2.0}, {3.0, 4.0}};
    double k_matrix[MATRIX_SIZE][MATRIX_SIZE] = {{0.5, 0.5}, {0.5, 0.5}};
    double q_matrix[MATRIX_SIZE][MATRIX_SIZE] = {{0.5, 0.5}, {0.5, 0.5}};
    double v_matrix[MATRIX_SIZE][MATRIX_SIZE] = {{0.5, 0.5}, {0.5, 0.5}};
    double self_attention_matrix[MATRIX_SIZE][MATRIX_SIZE];
    
    // Compute self-attention
    compute_self_attention(embedding_matrix, k_matrix, q_matrix, v_matrix, MATRIX_SIZE, self_attention_matrix);
    
    // Verify output matrix properties
    for(int i = 0; i < MATRIX_SIZE; i++){
        for(int j = 0; j < MATRIX_SIZE; j++){
            assert(!isnan(self_attention_matrix[i][j]));
            assert(!isinf(self_attention_matrix[i][j]));
        }
    }
    
    printf("Self-attention test passed!\n");
}

// Test gradient clipping
void test_gradient_clipping(){
    printf("Testing gradient clipping...\n");
    
    // Test values above threshold
    double large_gradient = 150.0;
    double clipped = clip_gradient_transformer(large_gradient);
    assert(clipped == CLIP_THRESHOLD);
    
    // Test values below threshold
    double small_gradient = 50.0;
    clipped = clip_gradient_transformer(small_gradient);
    assert(clipped == small_gradient);
    
    // Test negative values
    double negative_gradient = -150.0;
    clipped = clip_gradient_transformer(negative_gradient);
    assert(clipped == -CLIP_THRESHOLD);
    
    printf("Gradient clipping test passed!\n");
}

// Test matrix initialization
void test_matrix_initialization(){
    printf("Testing matrix initialization...\n");
    
    // Initialize matrices from files
    initialize_matrices_from_files();
    
    // Verify matrices are initialized
    assert(k_matrix != NULL);
    assert(q_matrix != NULL);
    assert(v_matrix != NULL);
    
    // Print matrices for visual inspection
    print_matrix("Key Matrix", k_matrix);
    print_matrix("Query Matrix", q_matrix);
    print_matrix("Value Matrix", v_matrix);
    
    printf("Matrix initialization test passed!\n");
}

int main(){
    printf("Starting transformer tests...\n\n");
    
    test_positional_encoding();
    test_matrix_operations();
    test_self_attention();
    test_gradient_clipping();
    test_matrix_initialization();
    
    printf("\nAll transformer tests passed successfully!\n");
    return 0;
}

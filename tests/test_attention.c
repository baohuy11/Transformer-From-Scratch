#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../src/self_attention_layer.c"

// Test dot product function
void test_dot_product() {
    printf("Testing dot_product...\n");
    
    float a[3] = {1.0f, 2.0f, 3.0f};
    float b[3] = {4.0f, 5.0f, 6.0f};
    float result = dot_product(a, b, 3);
    
    assert(result == 32.0f); // 1*4 + 2*5 + 3*6 = 32
    printf("dot_product test passed!\n\n");
}

// Test softmax function
void test_softmax() {
    printf("Testing softmax...\n");
    
    float input[3] = {1.0f, 2.0f, 3.0f};
    float output[3];
    softmax(input, output, 3);
    
    float sum = 0.0f;
    for(int i = 0; i < 3; i++) {
        sum += output[i];
    }
    
    assert(fabs(sum - 1.0f) < 1e-6); // Sum should be approximately 1
    printf("softmax test passed!\n\n");
}

// Test matrix multiplication
void test_matrix_multiply() {
    printf("Testing matrix_multiply...\n");
    
    float A[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};
    float B[EMBEDDING_DIM][EMBEDDING_DIM] = {0};
    float C[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};
    
    // Initialize test matrices
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < EMBEDDING_DIM; j++) {
            A[i][j] = (i + 1) * (j + 1) * 0.1f;
            B[j][j] = 1.0f; // Identity matrix
        }
    }
    
    matrix_multiply(A, B, C, 2, EMBEDDING_DIM, EMBEDDING_DIM);
    
    // Since B is identity matrix, C should equal A
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < EMBEDDING_DIM; j++) {
            assert(fabs(C[i][j] - A[i][j]) < 1e-6);
        }
    }
    
    printf("matrix_multiply test passed!\n\n");
}

// Test self-attention computation
void test_self_attention() {
    printf("Testing self_attention...\n");
    
    float input[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};
    float output[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};
    
    // Initialize input with simple values
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < EMBEDDING_DIM; j++) {
            input[i][j] = (i + 1) * (j + 1) * 0.1f;
        }
    }
    
    self_attention(input, output, 2);
    
    // Check if output has valid values
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < EMBEDDING_DIM; j++) {
            assert(!isnan(output[i][j]));
            assert(!isinf(output[i][j]));
        }
    }
    
    printf("self_attention test passed!\n\n");
}

// Test layer normalization
void test_layer_normalization() {
    printf("Testing layer_normalization...\n");
    
    float input[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};
    float output[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};
    
    // Initialize input with simple values
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < EMBEDDING_DIM; j++) {
            input[i][j] = (i + 1) * (j + 1) * 0.1f;
        }
    }
    
    layer_normalization(input, output, 2);
    
    // Check if output has valid values
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < EMBEDDING_DIM; j++) {
            assert(!isnan(output[i][j]));
            assert(!isinf(output[i][j]));
        }
    }
    
    printf("layer_normalization test passed!\n\n");
}

// Test feed-forward network
void test_feed_forward() {
    printf("Testing feed_forward...\n");
    
    float input[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};
    float output[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};
    
    // Initialize input with simple values
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < EMBEDDING_DIM; j++) {
            input[i][j] = (i + 1) * (j + 1) * 0.1f;
        }
    }
    
    feed_forward(input, output, 2);
    
    // Check if output has valid values
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < EMBEDDING_DIM; j++) {
            assert(!isnan(output[i][j]));
            assert(!isinf(output[i][j]));
        }
    }
    
    printf("feed_forward test passed!\n\n");
}

int main() {
    printf("Starting self-attention layer tests...\n\n");
    
    // Set random seed for reproducibility
    srand(42);
    
    test_dot_product();
    test_softmax();
    test_matrix_multiply();
    test_self_attention();
    test_layer_normalization();
    test_feed_forward();
    
    printf("All tests completed successfully!\n");
    return 0;
}


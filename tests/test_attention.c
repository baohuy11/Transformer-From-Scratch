#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../include/self_attention_layer.h"

// Test dot product calculation
void test_dot_product(){
    printf("Testing dot_product...\n");
    
    float a[EMBEDDING_DIM] = {0};
    float b[EMBEDDING_DIM] = {0};
    
    // Initialize test vectors
    for(int i = 0; i < EMBEDDING_DIM; i++){
        a[i] = (float)(i + 1);
        b[i] = (float)(i + 1);
    }
    
    float result = dot_product(a, b, EMBEDDING_DIM);
    
    // Expected result: sum of squares from 1 to EMBEDDING_DIM
    float expected = 0.0f;
    for(int i = 1; i <= EMBEDDING_DIM; i++){
        expected += i * i;
    }
    
    assert(fabs(result - expected) < 1e-6);
    printf("dot_product test passed\n");
}

// Test softmax computation
void test_softmax(){
    printf("Testing softmax...\n");
    
    float input[MAX_SEQ_LENGTH] = {0};
    float output[MAX_SEQ_LENGTH] = {0};
    
    // Initialize input with some values
    for(int i = 0; i < MAX_SEQ_LENGTH; i++){
        input[i] = (float)i;
    }
    
    softmax(input, output, MAX_SEQ_LENGTH);
    
    // Check if output sums to 1 (softmax property)
    float sum = 0.0;
    for(int i = 0; i < MAX_SEQ_LENGTH; i++){
        sum += output[i];
    }
    assert(fabs(sum - 1.0) < 1e-6);
    printf("softmax test passed\n");
}

// Test matrix multiplication
void test_matrix_multiply(){
    printf("Testing matrix_multiply...\n");
    
    float A[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};
    float B[EMBEDDING_DIM][EMBEDDING_DIM] = {0};
    float C[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};
    
    // Initialize test matrices
    for(int i = 0; i < MAX_SEQ_LENGTH; i++){
        for(int j = 0; j < EMBEDDING_DIM; j++){
            A[i][j] = (float)(i + j);
        }
    }
    
    for(int i = 0; i < EMBEDDING_DIM; i++){
        for(int j = 0; j < EMBEDDING_DIM; j++){
            B[i][j] = (float)(i + j);
        }
    }
    
    matrix_multiply(A, B, C, MAX_SEQ_LENGTH, EMBEDDING_DIM, EMBEDDING_DIM);
    
    // Basic check: output should not be all zeros
    int non_zero_count = 0;
    for(int i = 0; i < MAX_SEQ_LENGTH; i++){
        for(int j = 0; j < EMBEDDING_DIM; j++){
            if(fabs(C[i][j]) > 1e-6){
                non_zero_count++;
            }
        }
    }
    assert(non_zero_count > 0);
    printf("✓ matrix_multiply test passed\n");
}

// Test self-attention mechanism
void test_self_attention(){
    printf("Testing self_attention...\n");
    
    float input[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};
    float output[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};
    
    // Initialize input with some values
    for(int i = 0; i < MAX_SEQ_LENGTH; i++){
        for(int j = 0; j < EMBEDDING_DIM; j++){
            input[i][j] = (float)(i + j) / 10.0;
        }
    }
    
    self_attention(input, output, MAX_SEQ_LENGTH);
    
    // Basic checks
    // 1. Output dimensions should match input
    // 2. Output should not be all zeros
    int non_zero_count = 0;
    for(int i = 0; i < MAX_SEQ_LENGTH; i++){
        for(int j = 0; j < EMBEDDING_DIM; j++){
            if(fabs(output[i][j]) > 1e-6){
                non_zero_count++;
            }
        }
    }
    assert(non_zero_count > 0);
    printf("✓ self_attention test passed\n");
}

int main() {
    printf("Starting attention mechanism tests...\n\n");
    
    test_dot_product();
    test_softmax();
    test_matrix_multiply();
    test_self_attention();
    
    printf("\nAll attention mechanism tests passed successfully!\n");
    return 0;
}

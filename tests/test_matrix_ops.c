#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../include/transformer_block.h"

// Test matrix multiplication
void test_matrix_multiply(){
    printf("Testing matrix multiplication...\n");
    
    double A[MATRIX_SIZE][MATRIX_SIZE] = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    double B[MATRIX_SIZE][MATRIX_SIZE] = {
        {5.0, 6.0},
        {7.0, 8.0}
    };
    
    double result[MATRIX_SIZE][MATRIX_SIZE] = {0};
    
    matrix_multiply(A, B, result, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);
    
    // Expected result:
    // [1*5 + 2*7, 1*6 + 2*8]
    // [3*5 + 4*7, 3*6 + 4*8]
    // = [19, 22]
    //   [43, 50]
    
    assert(fabs(result[0][0] - 19.0) < 1e-6);
    assert(fabs(result[0][1] - 22.0) < 1e-6);
    assert(fabs(result[1][0] - 43.0) < 1e-6);
    assert(fabs(result[1][1] - 50.0) < 1e-6);
    
    printf("Matrix multiplication test passed\n");
}

// Test matrix addition
void test_matrix_addition(){
    printf("Testing matrix addition...\n");
    
    float A[MATRIX_SIZE][MATRIX_SIZE] = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    double B[MATRIX_SIZE][MATRIX_SIZE] = {
        {5.0, 6.0},
        {7.0, 8.0}
    };
    
    double result[MATRIX_SIZE][MATRIX_SIZE] = {0};
    
    add_matrices(A, B, result, MATRIX_SIZE, MATRIX_SIZE);
    
    // Expected result:
    // [1+5, 2+6]
    // [3+7, 4+8]
    // = [6, 8]
    //   [10, 12]
    
    assert(fabs(result[0][0] - 6.0) < 1e-6);
    assert(fabs(result[0][1] - 8.0) < 1e-6);
    assert(fabs(result[1][0] - 10.0) < 1e-6);
    assert(fabs(result[1][1] - 12.0) < 1e-6);
    
    printf("Matrix addition test passed\n");
}

// Test matrix transpose
void test_matrix_transpose(){
    printf("Testing matrix transpose...\n");
    
    double A[MATRIX_SIZE][MATRIX_SIZE] = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    double result[MATRIX_SIZE][MATRIX_SIZE] = {0};
    
    transpose(A, result);
    
    // Expected result:
    // [1, 3]
    // [2, 4]
    
    assert(fabs(result[0][0] - 1.0) < 1e-6);
    assert(fabs(result[0][1] - 3.0) < 1e-6);
    assert(fabs(result[1][0] - 2.0) < 1e-6);
    assert(fabs(result[1][1] - 4.0) < 1e-6);
    
    printf("Matrix transpose test passed\n");
}

// Test dot product
void test_dot_product(){
    printf("Testing dot product...\n");
    
    double A[MATRIX_SIZE][MATRIX_SIZE] = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    double B[MATRIX_SIZE][MATRIX_SIZE] = {
        {5.0, 6.0},
        {7.0, 8.0}
    };
    
    double result[MATRIX_SIZE][MATRIX_SIZE] = {0};
    
    dot_product(A, B, result);
    
    // Expected result:
    // [1*5 + 2*7, 1*6 + 2*8]
    // [3*5 + 4*7, 3*6 + 4*8]
    // = [19, 22]
    //   [43, 50]
    
    assert(fabs(result[0][0] - 19.0) < 1e-6);
    assert(fabs(result[0][1] - 22.0) < 1e-6);
    assert(fabs(result[1][0] - 43.0) < 1e-6);
    assert(fabs(result[1][1] - 50.0) < 1e-6);
    
    printf("Dot product test passed\n");
}

int main(){
    printf("Starting matrix operations tests...\n\n");
    
    test_matrix_multiply();
    test_matrix_addition();
    test_matrix_transpose();
    test_dot_product();
    
    printf("\nAll matrix operations tests passed successfully!\n");
    return 0;
}

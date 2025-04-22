#include <stdio.h>
#include <stdlib.h>
#include "include/Data_Preprocessing.h"

void print_matrix(float matrix[EMBEDDING_SIZE][MATRIX_SIZE], int rows) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < MATRIX_SIZE; j++) {
            printf("%.4f ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main() {
    // Test min_max_normalize_and_scale
    float test_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    size_t size = sizeof(test_data) / sizeof(test_data[0]);
    
    printf("Original data: ");
    for(size_t i = 0; i < size; i++) {
        printf("%.2f ", test_data[i]);
    }
    printf("\n");
    
    min_max_normalize_and_scale(test_data, size, 0.0f, 1.0f);
    
    printf("Normalized data: ");
    for(size_t i = 0; i < size; i++) {
        printf("%.2f ", test_data[i]);
    }
    printf("\n\n");
    
    // Test get_meaningful_length
    float test_data2[] = {0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f};
    size_t size2 = sizeof(test_data2) / sizeof(test_data2[0]);
    size_t meaningful_length = get_meaningful_length(test_data2, size2);
    printf("Meaningful length: %zu\n\n", meaningful_length);
    
    // Test scale_matrix and Add_Positional_Encoding
    float embedding_matrix[EMBEDDING_SIZE][MATRIX_SIZE] = {0};
    
    // Initialize with some test values
    for(int i = 0; i < 5; i++) {
        embedding_matrix[i][0] = (float)i;
        embedding_matrix[i][1] = (float)(i * 2);
    }
    
    printf("Original embedding matrix (first 5 rows):\n");
    print_matrix(embedding_matrix, 5);
    
    scale_matrix(embedding_matrix);
    printf("\nScaled embedding matrix (first 5 rows):\n");
    print_matrix(embedding_matrix, 5);
    
    Add_Positional_Encoding(embedding_matrix, 5);
    printf("\nEmbedding matrix with positional encoding (first 5 rows):\n");
    print_matrix(embedding_matrix, 5);
    
    return 0;
} 
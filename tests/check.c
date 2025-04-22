#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/feed_forward_layer.h"

#define NUM_WEIGHTS 10  // Example number of weights
#define TEST_PATH "weights"  // Path to weight files

void test_read_weights() {
    printf("Testing read_weights function...\n");
    
    // Allocate memory for weights
    double* weights = (double*)malloc(NUM_WEIGHTS * sizeof(double));
    if (weights == NULL) {
        printf("Memory allocation failed\n");
        return;
    }
    
    // Initialize weights to 0
    memset(weights, 0, NUM_WEIGHTS * sizeof(double));
    
    // Read weights from files
    read_weights(TEST_PATH, weights, NUM_WEIGHTS);
    
    // Print the read weights
    printf("Read weights:\n");
    for (int i = 0; i < NUM_WEIGHTS; i++) {
        printf("Weight %d: %lf\n", i, weights[i]);
    }
    
    // Clean up
    free(weights);
}

void test_feed_forward() {
    printf("\nTesting feed forward layer...\n");
    
    // Example input vector
    double input[4] = {1.0, 2.0, 3.0, 4.0};
    int input_size = 4;
    int hidden_size = 8;
    int output_size = 4;
    
    // Create feed forward layer
    FeedForwardLayer* layer = create_feed_forward_layer(input_size, hidden_size, output_size);
    if (layer == NULL) {
        printf("Failed to create feed forward layer\n");
        return;
    }
    
    // Forward pass
    double* output = feed_forward_forward(layer, input);
    if (output == NULL) {
        printf("Forward pass failed\n");
        free_feed_forward_layer(layer);
        return;
    }
    
    // Print output
    printf("Feed forward output:\n");
    for (int i = 0; i < output_size; i++) {
        printf("Output %d: %lf\n", i, output[i]);
    }
    
    // Clean up
    free(output);
    free_feed_forward_layer(layer);
}

int main() {
    printf("Starting feed forward layer tests...\n");
    
    test_read_weights();
    test_feed_forward();
    
    printf("\nAll tests completed.\n");
    return 0;
} 
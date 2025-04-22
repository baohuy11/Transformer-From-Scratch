#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../include/backprop.h"

#define EPSILON 1e-6

// Test MSE calculation
void test_calculate_mse() {
    printf("Testing MSE calculation...\n");
    
    double output[] = {1.0, 2.0, 3.0};
    double expected[] = {1.5, 2.5, 3.5};
    int size = 3;
    
    double mse = calculate_mse(output, expected, size);
    
    // Expected MSE = ((1.0-1.5)² + (2.0-2.5)² + (3.0-3.5)²) / 3
    // = (0.25 + 0.25 + 0.25) / 3 = 0.25
    assert(fabs(mse - 0.25) < 1e-6);
    
    printf("MSE calculation test passed\n");
}

// Test gradient clipping
void test_clip_gradient() {
    printf("Testing gradient clipping...\n");
    
    double clip_threshold = 1.0;
    
    // Test positive value above threshold
    assert(fabs(clip_gradient_backpropagation(2.0, clip_threshold) - 1.0) < 1e-6);
    
    // Test negative value below threshold
    assert(fabs(clip_gradient_backpropagation(-2.0, clip_threshold) + 1.0) < 1e-6);
    
    // Test value within threshold
    assert(fabs(clip_gradient_backpropagation(0.5, clip_threshold) - 0.5) < 1e-6);
    
    printf("Gradient clipping test passed\n");
}

// Test weight updates
void test_weight_updates() {
    printf("Testing weight updates...\n");
    
    // Initialize test arrays
    double final_layer_weights[4] = {0.1, 0.2, 0.3, 0.4};
    double semi_final_layer_weights[2] = {0.5, 0.6};
    double loss = 0.1;
    double learning_rate = 0.01;
    double clip_threshold = 1.0;
    
    // Save original weights for comparison
    double original_final_weights[4];
    double original_semi_final_weights[2];
    for(int i = 0; i < 4; i++) original_final_weights[i] = final_layer_weights[i];
    for(int i = 0; i < 2; i++) original_semi_final_weights[i] = semi_final_layer_weights[i];
    
    // Update weights
    update_weights_last_layer(loss, learning_rate, final_layer_weights, 
                            semi_final_layer_weights, 2, 2, clip_threshold);
    update_semi_final_layer_weights(loss, learning_rate, semi_final_layer_weights, 
                                  2, clip_threshold);
    
    // Verify weights were updated
    for(int i = 0; i < 4; i++) {
        assert(fabs(final_layer_weights[i] - original_final_weights[i]) > 1e-6);
    }
    for(int i = 0; i < 2; i++) {
        assert(fabs(semi_final_layer_weights[i] - original_semi_final_weights[i]) > 1e-6);
    }
    
    printf("Weight updates test passed\n");
}

int main() {
    printf("Starting backpropagation tests...\n\n");
    
    test_calculate_mse();
    test_clip_gradient();
    test_weight_updates();
    
    printf("\nAll backpropagation tests passed successfully!\n");
    return 0;
} 
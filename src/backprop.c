#include <math.h>
#include <stdio.h>

#include "../include/backprop.h"

// FUNCTION TO CALCULATE MEAN SQUARED ERROR (MSE)
double calculate_mse(double output_array[], double expected_output_array[], int size){
    double mse = 0.0;
    for(int i = 0; i < size; i++){
        double difference = output_array[i] - expected_output_array[i];
        mse += difference * difference;
    }
    return mse / size;
}

// FUNCTION TO CLIP GRADIENTS
double clip_gradient_backpropagation(double gradient, double clip_threshold) {
    if(gradient > clip_threshold){
        return clip_threshold;
    }else if(gradient < -clip_threshold){
        return -clip_threshold;
    }
    return gradient;
}

// FUNCTION TO UPDATE WEIGHTS IN THE FINAL LAYER
void update_weights_last_layer(double loss, double learning_rate, double final_layer_weights[], double semi_final_layer_weights[], int final_layer_size, int semi_final_layer_size, double clip_threshold){
    // Update weights for the first node in final layer
    for(int i = 0; i < semi_final_layer_size; i++){
        double gradient_node_1 = semi_final_layer_weights[i] * loss;
        gradient_node_1 = clip_gradient_backpropagation(gradient_node_1, clip_threshold);
        final_layer_weights[i] -= learning_rate * gradient_node_1;
    }

    // Update weights for the second node in final layer
    for(int i = 0; i < semi_final_layer_size; i++){
        double gradient_node_2 = semi_final_layer_weights[i] * loss;
        gradient_node_2 = clip_gradient_backpropagation(gradient_node_2, clip_threshold);
        final_layer_weights[i + semi_final_layer_size] -= learning_rate * gradient_node_2;
    }
}

// FUNCTION TO UPDATE WEIGHTS IN THE SEMI-FINAL LAYER
void update_semi_final_layer_weights(double loss, double learning_rate, double semi_final_layer_weights[], int semi_final_layer_size, double clip_threshold){
    // Update each weight in the semi-final layer
    for(int i = 0; i < semi_final_layer_size; i++){
        double gradient = loss * semi_final_layer_weights[i];
        gradient = clip_gradient_backpropagation(gradient, clip_threshold);
        semi_final_layer_weights[i] -= learning_rate * gradient;
    }
}
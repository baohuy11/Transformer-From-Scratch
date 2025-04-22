#ifndef BACKPROP_H
#define BACKPROP_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double calculate_mse(double output_array[], double expected_output_array[], int size);

double clip_gradient_backpropagation(double gradient, double clip_threshold);

void update_weights_last_layer(double loss, double learning_rate, double final_layer_weights[], 
                             double semi_final_layer_weights[], int final_layer_size, 
                             int semi_final_layer_size, double clip_threshold);

void update_semi_final_layer_weights(double loss, double learning_rate, 
                                   double semi_final_layer_weights[], 
                                   int semi_final_layer_size, double clip_threshold);

#endif // BACKPROP_H
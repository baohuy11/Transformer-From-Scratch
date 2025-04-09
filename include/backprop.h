#ifndef BACKPROP_H
#define BACKPROP_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double calculate_mse(double* output, double* target, int size);

void update_weights_last_layer(double loss, double learning_rate, double* final_weights, double* semi_final_weights, int final_size, int semi_final_size, double clip_threshold);

void update_weights_hidden_layer(double loss, double learning_rate, double* semi_final_weights, int semi_final_size, double clip_threshold);

void update_weights_self_attention(double gradient, double clip_threshold);

#endif // BACKPROP_H
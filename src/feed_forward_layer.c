#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "../include/feed_forward_layer.h"

// READ THE WEIGHTS FROM THE TEXT FILES
void read_weights(const char* path, double* weights, int num_weights) {
    char file_name[256];
    FILE* file;
    double weight;

    for(int i = 1; i < num_weights; i++) {
        snprintf(file_name, sizeof(file_name), "%s/weight_%d.txt", path, i);
        file = fopen(file_name, "r");

        if(file == NULL) {
            fprintf(stderr, "Error opening file %s\n", file_name);
            exit(EXIT_FAILURE);
        }

        if(fscanf(file, "%lf", &weight) != 1) {
            fprintf(stderr, "Error reading weight from file %s\n", file_name);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        weights[i - 1] = weight;
        fclose(file);
    }
}

// Create a new feed forward layer
FeedForwardLayer* create_feed_forward_layer(int input_size, int hidden_size, int output_size) {
    FeedForwardLayer* layer = (FeedForwardLayer*)malloc(sizeof(FeedForwardLayer));
    if (layer == NULL) return NULL;

    layer->input_size = input_size;
    layer->hidden_size = hidden_size;
    layer->output_size = output_size;

    // Allocate memory for weights and biases
    layer->weights1 = (double*)malloc(input_size * hidden_size * sizeof(double));
    layer->weights2 = (double*)malloc(hidden_size * output_size * sizeof(double));
    layer->bias1 = (double*)malloc(hidden_size * sizeof(double));
    layer->bias2 = (double*)malloc(output_size * sizeof(double));

    if (layer->weights1 == NULL || layer->weights2 == NULL || 
        layer->bias1 == NULL || layer->bias2 == NULL) {
        free_feed_forward_layer(layer);
        return NULL;
    }

    // Initialize weights with small random values
    for (int i = 0; i < input_size * hidden_size; i++) {
        layer->weights1[i] = (double)rand() / RAND_MAX * 0.1;
    }
    for (int i = 0; i < hidden_size * output_size; i++) {
        layer->weights2[i] = (double)rand() / RAND_MAX * 0.1;
    }

    // Initialize biases to zero
    memset(layer->bias1, 0, hidden_size * sizeof(double));
    memset(layer->bias2, 0, output_size * sizeof(double));

    return layer;
}

// Free the feed forward layer
void free_feed_forward_layer(FeedForwardLayer* layer) {
    if (layer == NULL) return;
    
    free(layer->weights1);
    free(layer->weights2);
    free(layer->bias1);
    free(layer->bias2);
    free(layer);
}

// ReLU activation function
static double relu(double x) {
    return x > 0 ? x : 0;
}

// Forward pass through the feed forward layer
double* feed_forward_forward(FeedForwardLayer* layer, const double* input) {
    if (layer == NULL || input == NULL) return NULL;

    // Allocate memory for hidden layer and output
    double* hidden = (double*)malloc(layer->hidden_size * sizeof(double));
    double* output = (double*)malloc(layer->output_size * sizeof(double));
    
    if (hidden == NULL || output == NULL) {
        free(hidden);
        free(output);
        return NULL;
    }

    // First layer: input -> hidden
    for (int i = 0; i < layer->hidden_size; i++) {
        hidden[i] = layer->bias1[i];
        for (int j = 0; j < layer->input_size; j++) {
            hidden[i] += input[j] * layer->weights1[j * layer->hidden_size + i];
        }
        hidden[i] = relu(hidden[i]);  // Apply ReLU activation
    }

    // Second layer: hidden -> output
    for (int i = 0; i < layer->output_size; i++) {
        output[i] = layer->bias2[i];
        for (int j = 0; j < layer->hidden_size; j++) {
            output[i] += hidden[j] * layer->weights2[j * layer->output_size + i];
        }
    }

    free(hidden);
    return output;
}
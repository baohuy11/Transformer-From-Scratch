#ifndef FEED_FORWARD_LAYER_H
#define FEED_FORWARD_LAYER_H

#include <stdlib.h>

// Structure to hold the feed forward layer parameters
typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    double* weights1;  // First layer weights
    double* weights2;  // Second layer weights
    double* bias1;     // First layer bias
    double* bias2;     // Second layer bias
} FeedForwardLayer;

// Function to read weights from files
void read_weights(const char* path, double* weights, int num_weights);

// Create a new feed forward layer
FeedForwardLayer* create_feed_forward_layer(int input_size, int hidden_size, int output_size);

// Free the feed forward layer
void free_feed_forward_layer(FeedForwardLayer* layer);

// Forward pass through the feed forward layer
double* feed_forward_forward(FeedForwardLayer* layer, const double* input);

#endif /* FEED_FORWARD_LAYER_H */
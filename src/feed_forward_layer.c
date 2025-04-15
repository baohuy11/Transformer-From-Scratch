#include <stdlib.h>
#include <stdio.h>

#include "../include/feed_forward_layer.h"

void read_weights(const char* path, double* semi_final_layer_weights, int num_weights){
    char file_name[256];

    FILE* file;

    double weight;

    // Iterate from 1 to the Specified number of weights
    for(int i = 1; i < num_weights; i++){

        // Create the file name
        snprintf(file_name, sizeof(file_name), "%weight_%d.txt", path, i);
        // Open the file for reading
        file = fopen(file_name, "r");

        if(file == NULL){
            fprintf(stderr, "Error opening file %s\n", file_name);
            exit(EXIT_FAILURE);
        }

        // Read the double value from the file
        if(fscanf(file, "%lf", &weight) != 1){
            fprintf(stderr, "Error reading weight from file %s\n", file_name);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        // Store the value in the array
        semi_final_layer_weights[i - 1] = weight;

        // Close the file
        fclose(file);
    }
}
#include <stdlib.h>
#include <stdio.h>

#include "../include/feed_forward_layer.h"

// READ THE WEIGHTS FROM THE TEXT FILES
void read_weights(const char* path, double* semi_final_layer_weights, int num_weights){
    char file_name[256];  // Buffer to store the full file path
    FILE* file;           // File pointer for reading
    double weight;        // Temporary variable to store each weight

    // Iterate through each weight file (from 1 to num_weights-1)
    for(int i = 1; i < num_weights; i++){
        // Construct the file name using the base path and weight index
        // Format: "path/weight_X.txt" where X is the weight index
        snprintf(file_name, sizeof(file_name), "%weight_%d.txt", path, i);
        
        // Open the weight file for reading
        file = fopen(file_name, "r");

        // Check if file was opened successfully
        if(file == NULL){
            fprintf(stderr, "Error opening file %s\n", file_name);
            exit(EXIT_FAILURE);
        }

        // Read a single double value from the file
        if(fscanf(file, "%lf", &weight) != 1){
            fprintf(stderr, "Error reading weight from file %s\n", file_name);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        // Store the weight in the array at index (i-1)
        semi_final_layer_weights[i - 1] = weight;

        // Close the file after reading
        fclose(file);
    }
}
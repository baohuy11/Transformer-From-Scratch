#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/self_attention_layer.h"

#define VOCAB_SIZE 1000
#define EMBEDDING_DIM 512
#define MAX_SEQ_LEN 128

// Function to compute the dot product of two matrices
float dot_product(float *a, float *b, int dim){
    float result = 0.0f;
    for(int i = 0; i < dim; i++){
        result += a[i] * b[i];
    }
    return result;
}

// Function to compute the softmax
void softmax(float *input, float *output, int length){
    float max_val = input[0];
    for(int i = 1; i < length; i++){
        if(input[i] > max_val){
            max_val = input[i];
        }
    }

    float sum = 0.0f;
    for(int i = 0; i < length; i++){
        output[i] = exp(input[i] - max_val); // Subtract max for numerical stability
        sum += output[i];
    }

    for(int i = 0; i < length; i++){
        output[i] /= sum; // Normalize to get probabilities
    }
}

// Function to initialize the token embedding matrix
void initialize_embedding(float embedding[VOCAB_SIZE][EMBEDDING_DIM]){
    printf("Initializing tone embedding matrix:\n\n");

    for(int i = 0; i < VOCAB_SIZE; i++){
        printf("Embedding vector for token index %d: [", i);

        for(int j = 0; j < EMBEDDING_DIM; j++){/
            embedding[i][j] = ((float)rand() / RAND_MAX) - 0.5; // Value between -0.5 dan 0.5

            printf("%f", embedding[i][j]);
            if(j < EMBEDDING_DIM - 1){
                printf(", ");
            }
        }
    }
}

// Function to generate the positional encoding matrix
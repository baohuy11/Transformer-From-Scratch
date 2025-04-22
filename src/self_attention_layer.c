#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "../include/self_attention_layer.h"

// Model hyperparameters
#define VOCAB_SIZE 1000        // Size of the vocabulary
#define EMBEDDING_DIM 512      // Dimension of token embeddings
#define MAX_SEQ_LENGTH 128     // Maximum sequence length
#define EPSILON 1e-6          // Small value for numerical stability
#define FF_DIM 2048          // Feed-forward network dimension

// FUNCTION TO COMPUTE THE DOT PRODUCT OF TWO VECTORS
float dot_product(float *a, float *b, int dim){
    assert(a != NULL && b != NULL && dim > 0);
    float result = 0.0f;
    for(int i = 0; i < dim; i++){
        result += a[i] * b[i];
    }
    return result;
}

// FUNCTION TO APPLY SOFTMAX TO A VECTOR OF SCORES
void softmax(float *input, float *output, int length){
    assert(input != NULL && output != NULL && length > 0);
    
    // Find maximum value for numerical stability
    float max_val = input[0];
    for(int i = 1; i < length; i++){
        if(input[i] > max_val){
            max_val = input[i];
        }
    }

    // Compute exponentials and sum
    float sum = 0.0f;
    for(int i = 0; i < length; i++){
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }

    // Normalize to get probabilities
    for(int i = 0; i < length; i++){
        output[i] /= sum;
    }
}

// FUNCTION TO INITIALIZE THE TOKEN EMBEDDING MATRIX
void initialize_embedding(float embedding[VOCAB_SIZE][EMBEDDING_DIM]){
    assert(embedding != NULL);
    
    printf("Initializing token embedding matrix:\n\n");
    for(int i = 0; i < VOCAB_SIZE; i++){
        printf("Embedding vector for token index %d: [", i);
        for(int j = 0; j < EMBEDDING_DIM; j++){
            // Initialize with random values between -0.5 and 0.5
            embedding[i][j] = ((float)rand() / RAND_MAX) - 0.5f;
            printf("%f", embedding[i][j]);
            if(j < EMBEDDING_DIM - 1) {
                printf(", ");
            }
        }
        printf("]\n\n");
    }
    printf("Token embedding matrix initialized successfully.\n\n");
}

// FUNCTION TO GENERATE POSITIONAL ENCODING
void generate_positional_encoding(float positional_encoding[MAX_SEQ_LENGTH][EMBEDDING_DIM]){
    assert(positional_encoding != NULL);
    
    printf("Generating positional encoding matrix:\n\n");
    for(int pos = 0; pos < MAX_SEQ_LENGTH; pos++){
        for(int i = 0; i < EMBEDDING_DIM; i++){
            if(i % 2 == 0){
                positional_encoding[pos][i] = sin(pos / pow(10000, (2.0 * i / EMBEDDING_DIM)));  
            } else {
                positional_encoding[pos][i] = cos(pos / pow(10000, (2.0 * (i - 1) / EMBEDDING_DIM)));  
            }
        }
        printf("Positional Encoding for position %d: [", pos);
        for(int j = 0; j < EMBEDDING_DIM; j++){
            printf("%f", positional_encoding[pos][j]);
            if(j < EMBEDDING_DIM - 1){
                printf(", ");
            }
        }
        printf("]\n\n");
    }
    printf("Positional Encoding Generated!\n\n");
}

// FUNCTION TO LOOKUP EMBEDDING FOR A GIVEN TOKEN INDEX
void lookup_embedding(float embedding[VOCAB_SIZE][EMBEDDING_DIM], int token_index, float *output){
    printf("Looking up Embedding for Token Index %d:\n\n", token_index);
    for(int i = 0; i < EMBEDDING_DIM; i++){
        output[i] = embedding[token_index][i];
        printf("output[%d] = %f\n", i, output[i]);
    }

    printf("\nEmbedding Lookup Completed!\n\n");
}

// FUNCTION TO CONCATENATE TOKEN EMBEDDING AND POSITIONAL EMBEDDING
void concatenate_embeddings(float token_embedding[EMBEDDING_DIM], float positional_embedding[EMBEDDING_DIM], float *output){
    printf("Concatenating Token and Positional Embedding:\n\n");
    
    for(int i = 0; i < EMBEDDING_DIM; i++){
        output[i] = token_embedding[i] + positional_embedding[i];  
        printf("Concatenated output[%d] = %f\n", i, output[i]);
    }

    printf("\nConcatenation Completed!\n\n");
}

// FUNCTION TO MULTIPLY TWO MATRICES
void matrix_multiply(float A[MAX_SEQ_LENGTH][EMBEDDING_DIM], float B[EMBEDDING_DIM][EMBEDDING_DIM], float C[MAX_SEQ_LENGTH][EMBEDDING_DIM], int rows_A, int cols_A, int cols_B){
    for(int i = 0; i < rows_A; i++){
        for(int j = 0; j < cols_B; j++){
            C[i][j] = 0.0f;
            for (int k = 0; k < cols_A; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// FUNCTION TO INITIALIZE WEIGHT MATRICES
void initialize_weight_matrix(float weight[EMBEDDING_DIM][EMBEDDING_DIM]){
    for(int i = 0; i < EMBEDDING_DIM; i++){
        for(int j = 0; j < EMBEDDING_DIM; j++){
            weight[i][j] = ((float) rand() / (float)(RAND_MAX)) - 0.5; // Random values between -0.5 and 0.5
        }
    }
}

// FUNCTION TO COMPUTE SELF-ATTENTION WITH TRAINABLE K, Q, V
void self_attention(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM], float output[MAX_SEQ_LENGTH][EMBEDDING_DIM], int seq_length){
    float W_Q[EMBEDDING_DIM][EMBEDDING_DIM]; // Weight matrix for Q
    float W_K[EMBEDDING_DIM][EMBEDDING_DIM]; // Weight matrix for K
    float W_V[EMBEDDING_DIM][EMBEDDING_DIM]; // Weight matrix for V

    // Initialize weight matrices
    initialize_weight_matrix(W_Q);
    initialize_weight_matrix(W_K);
    initialize_weight_matrix(W_V);

    float Q[MAX_SEQ_LENGTH][EMBEDDING_DIM]; // Query matrix
    float K[MAX_SEQ_LENGTH][EMBEDDING_DIM]; // Key matrix
    float V[MAX_SEQ_LENGTH][EMBEDDING_DIM]; // Value matrix

    // Compute Q, K, V by multiplying input with the weight matrices
    matrix_multiply(input, W_Q, Q, seq_length, EMBEDDING_DIM, EMBEDDING_DIM);
    matrix_multiply(input, W_K, K, seq_length, EMBEDDING_DIM, EMBEDDING_DIM);
    matrix_multiply(input, W_V, V, seq_length, EMBEDDING_DIM, EMBEDDING_DIM);

    float attention_scores[MAX_SEQ_LENGTH][MAX_SEQ_LENGTH] = {0};
    float attention_weights[MAX_SEQ_LENGTH][MAX_SEQ_LENGTH] = {0};

    // Compute attention scores
    for(int i = 0; i < seq_length; i++){
        for (int j = 0; j < seq_length; j++){
            attention_scores[i][j] = dot_product(Q[i], K[j], EMBEDDING_DIM) / sqrt(EMBEDDING_DIM);
        }
    }

    // Compute attention weights using softmax
    for(int i = 0; i < seq_length; i++){
        softmax(attention_scores[i], attention_weights[i], seq_length);
    }

    // Compute output of self-attention
    for(int i = 0; i < seq_length; i++){
        for (int j = 0; j < EMBEDDING_DIM; j++){
            output[i][j] = 0.0f;
            for(int k = 0; k < seq_length; k++){
                output[i][j] += attention_weights[i][k] * V[k][j];
            }
        }
    }
}

void feed_forward(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM], float output[MAX_SEQ_LENGTH][EMBEDDING_DIM], int seq_length) {
    // Allocate memory for weights and intermediate values
    float (*W1)[FF_DIM] = malloc(sizeof(float[EMBEDDING_DIM][FF_DIM]));
    float (*W2)[EMBEDDING_DIM] = malloc(sizeof(float[FF_DIM][EMBEDDING_DIM]));
    float (*intermediate)[FF_DIM] = malloc(sizeof(float[MAX_SEQ_LENGTH][FF_DIM]));
    
    if (!W1 || !W2 || !intermediate) {
        printf("Memory allocation failed in feed_forward\n");
        if (W1) free(W1);
        if (W2) free(W2);
        if (intermediate) free(intermediate);
        return;
    }
    
    // Initialize weights
    for(int i = 0; i < EMBEDDING_DIM; i++) {
        for(int j = 0; j < FF_DIM; j++) {
            W1[i][j] = ((float) rand() / (float)(RAND_MAX)) - 0.5;
        }
    }
    
    for(int i = 0; i < FF_DIM; i++) {
        for(int j = 0; j < EMBEDDING_DIM; j++) {
            W2[i][j] = ((float) rand() / (float)(RAND_MAX)) - 0.5;
        }
    }
    
    // First linear transformation with ReLU
    for(int i = 0; i < seq_length; i++) {
        for(int j = 0; j < FF_DIM; j++) {
            intermediate[i][j] = 0;
            for(int k = 0; k < EMBEDDING_DIM; k++) {
                intermediate[i][j] += input[i][k] * W1[k][j];
            }
            intermediate[i][j] = fmax(0, intermediate[i][j]); // ReLU activation
        }
    }
    
    // Second linear transformation
    for(int i = 0; i < seq_length; i++) {
        for(int j = 0; j < EMBEDDING_DIM; j++) {
            output[i][j] = 0;
            for(int k = 0; k < FF_DIM; k++) {
                output[i][j] += intermediate[i][k] * W2[k][j];
            }
        }
    }
    
    // Free allocated memory
    free(W1);
    free(W2);
    free(intermediate);
}

void layer_normalization(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM], float output[MAX_SEQ_LENGTH][EMBEDDING_DIM], int seq_length){
    for(int i = 0; i < seq_length; i++){
        float mean = 0.0f;
        float variance = 0.0f;
        
        // Compute the mean
        for(int j = 0; j < EMBEDDING_DIM; j++){
            mean += input[i][j];
        }
        mean /= EMBEDDING_DIM;

        // Compute the variance
        for(int j = 0; j < EMBEDDING_DIM; j++){
            variance += (input[i][j] - mean) * (input[i][j] - mean);
        }
        variance /= EMBEDDING_DIM;

        // Compute the normalization
        for(int j = 0; j < EMBEDDING_DIM; j++){
            output[i][j] = (input[i][j] - mean) / sqrt(variance + EPSILON); // Add epsilon for numerical stability
        }
    }
}